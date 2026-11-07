"""Frozen paired quality/key-budget gate. One model; one fresh cache per branch."""
import contextlib,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from final_router_counter import CountingRouter
from zha_l6_controlled import LAYERS,initial_references
from zha_l6_reference import installed

root=Path(os.environ['FINAL_W_ROOT']);source_root=Path(__file__).resolve().parent
split=sys.argv[1];assert split in ('dev','final')
protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))['cases']
lock=json.loads((root/'source_lock.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,expected in lock['source_sha256'].items():assert sha(source_root/name)==expected,('source changed',name)
trained_path=Path('W:/Codex-ZHA-L6-20261104/R16/router.pt');saved_path=Path('W:/Codex-ZHA-L6-20261104/R16/checkpoint.pt')
assert sha(trained_path)==lock['trained_router_sha256'] and sha(saved_path)==lock['training_checkpoint_sha256']
assert sha(root/'protocol.json')==lock['protocol_sha256'] and sha(root/'manifest.json')==lock['manifest_sha256']
out=root/f'{split}_results.jsonl';done=set()
if out.exists():
 for s in out.read_text(encoding='utf8').splitlines():
  r=json.loads(s);key=(r['task'],r['index'],r['mode']);assert key not in done;done.add(key)
def guard():
 free=shutil.disk_usage('C:/').free
 if free<26*2**30:raise RuntimeError(f'DISK_GUARD_C {free}')
 if shutil.disk_usage(root).free<10*2**30:raise RuntimeError('DISK_GUARD_W')
guard();torch.manual_seed(121);torch.set_num_threads(4)
payload=torch.load(trained_path,map_location='cpu',weights_only=True)
saved=torch.load(saved_path,map_location='cpu',weights_only=True)
reconstructed=initial_references(16);trained=payload['refs'];initial=trained.clone().zero_()
for layer in LAYERS:
 assert torch.equal(payload['initial'][layer],reconstructed[layer])
 assert torch.equal(saved['initial'][layer],reconstructed[layer])
 assert torch.equal(saved['refs'][str(layer)],trained[layer])
 initial[layer]=payload['initial'][layer]
init_hash=hashlib.sha256(b''.join(initial[l].numpy().tobytes() for l in LAYERS)).hexdigest()
assert init_hash==lock['initial_tensor_sha256']
trained=trained.cuda();initial=initial.cuda()
t=AutoTokenizer.from_pretrained('model',local_files_only=True)
m=Qwen3_5ForConditionalGeneration.from_pretrained('model',dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True).eval()
items=[x for x in manifest if x['split']==split]
assert len(items)==(4 if split=='dev' else 96)
for item in items:
 task,index=item['task'],item['index'];casepath=root/'cases'/f'{split}_{task}'/'validation.jsonl'
 assert sha(casepath)==item['case_file_sha256']
 cases=[json.loads(s) for s in casepath.read_text(encoding='utf8').splitlines()];case=cases[index]
 assert hashlib.sha256(case['input'].encode()).hexdigest()==item['input_sha256']
 seq=t(case['input']+case['answer_prefix'],add_special_tokens=False).input_ids;N=len(seq)
 assert hashlib.sha256(str(seq).encode()).hexdigest()==item['token_sha256'] and N==item['input_tokens']
 assert case['outputs']==item['expected'] and case['outputs']
 for mode in protocol['configurations']:
  if (task,index,mode) in done:continue
  guard();print('CASE_START',split,task,index,mode,'tokens',N,flush=True)
  refs=None if mode=='dense' else (initial if mode.startswith('init') else trained)
  threshold=None if refs is None else (.9 if mode.endswith('090') else .95)
  r=CountingRouter(refs,p=threshold,disk_guard_gib=26,measure_mass=False) if refs is not None else None
  ctx=installed(r) if r else contextlib.nullcontext();cache=None;times=[];tokens=[]
  try:
   with torch.no_grad(),ctx:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for pos in range(0,N-1,512):
     guard();end=min(N-1,pos+512);x=torch.tensor([seq[pos:end]],device='cuda');mask=torch.ones((1,end),device='cuda',dtype=torch.long)
     z=m(input_ids=x,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
     cache=z.past_key_values;del x,mask,z
     if end%4096==0:print('PREFILL',split,task,index,mode,end,N,'C_free',shutil.disk_usage('C:/').free,flush=True)
    torch.cuda.synchronize();prefill=time.perf_counter()-start
    expected_pre=128*(N-1)*N//2;expected_pre_queries=128*(N-1)
    pre_calls=len(r.calls) if r else 0
    if r:
     assert pre_calls==8*math.ceil((N-1)/512) and not r.fallbacks
     assert r.available_keys==expected_pre and r.query_heads==expected_pre_queries
     pre_selected,pre_available=r.selected_keys,r.available_keys
    else:pre_selected=pre_available=expected_pre
    next_in=seq[-1]
    for j in range(item['generation_cap']):
     guard();x=torch.tensor([[next_in]],device='cuda');mask=torch.ones((1,N+j),device='cuda',dtype=torch.long)
     torch.cuda.synchronize();start=time.perf_counter()
     z=m(input_ids=x,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
     torch.cuda.synchronize();times.append(time.perf_counter()-start)
     cache=z.past_key_values;next_in=int(z.logits[:,-1].argmax());tokens.append(next_in);del x,mask,z
     if (j+1)%20==0:print('DECODE',split,task,index,mode,j+1,flush=True)
     if next_in==t.eos_token_id:break
   count=len(tokens);expected_dec=128*(count*N+count*(count-1)//2)
   if r:
    assert len(r.calls)==pre_calls+8*count and not r.fallbacks
    dec_selected=r.selected_keys-pre_selected;dec_available=r.available_keys-pre_available
    assert dec_available==expected_dec and r.query_heads-expected_pre_queries==128*count
   else:dec_selected=dec_available=expected_dec
   assert 0<pre_selected<=pre_available and 0<dec_selected<=dec_available
   text=t.decode(tokens);found=[a.lower() in text.lower() for a in case['outputs']]
   row={'split':split,'task':task,'index':index,'mode':mode,'threshold':threshold,
    'checkpoint_sha256':lock['trained_router_sha256'] if refs is not None else None,
    'reference_tensor_sha256':init_hash if mode.startswith('init') else lock['trained_reference_tensor_sha256'] if refs is not None else None,
    'input_tokens':N,'input_sha256':item['input_sha256'],'token_sha256':item['token_sha256'],
    'expected':case['outputs'],'generated':text,'score':sum(found)/len(found),'all_expected_recovered':all(found),
    'prefill_selected':pre_selected,'prefill_available':pre_available,'prefill_key_share':pre_selected/pre_available,
    'decode_selected':dec_selected,'decode_available':dec_available,'decode_key_share':dec_selected/dec_available,
    'prefill_s':prefill,'decode_s_total':sum(times),'generated_tokens':count,
    'ended_on_eos':tokens[-1]==t.eos_token_id,'fallback_count':len(r.fallbacks) if r else 0,
    'routed_calls':len(r.calls) if r else 0,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
   with out.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
   done.add((task,index,mode));print('CASE_COMPLETE',json.dumps({k:row[k] for k in ('split','task','index','mode','score','all_expected_recovered','prefill_key_share','decode_key_share','prefill_s','decode_s_total','peak_allocated_gib')}),flush=True)
  finally:cache=None;torch.cuda.empty_cache()
