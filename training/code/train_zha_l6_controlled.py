"""Controlled fixed-L6 R4/R16 frozen-teacher training; writes to W: only.

Both sizes use identical document order, sampled query positions, WSD schedule,
and nested reference initialization. Each document has fresh recurrent state.
"""
import hashlib,json,os,random,shutil,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from r16_multilevel_loss import objective
from zha_l6_controlled import LAYERS,initial_references,wsd_multiplier

if len(sys.argv)<2 or sys.argv[1] not in ('4','16'):raise ValueError('specify reference count 4 or 16')
R=int(sys.argv[1]);pilot='--pilot' in sys.argv
root=Path(os.environ['ZHA_W_ROOT']);assert str(root).upper().startswith('W:')
out=root/f'R{R}';out.mkdir(parents=True,exist_ok=True)
data_root=Path('results/plan_boundary');manifest_path=data_root/'dataset_manifest.json';manifest=json.loads(manifest_path.read_text())
rows=manifest['documents'];pg=[x for x in rows if x['source']=='pg19' and x['split']=='train'];wiki=[x for x in rows if x['source']=='wikitext2' and x['split']=='train']
valid=[x for x in rows if (x['source']=='pg19' and x['split']=='validation') or (x['source']=='wikitext2' and x['split']=='valid')]
assert len(pg)==len(wiki)==64 and len(valid)==8
train=[x for pair in zip(pg,wiki) for x in pair];total=2*len(train)
torch.manual_seed(94121);torch.set_num_threads(4)
init=initial_references(R);params={l:torch.nn.Parameter(init[l].to('cuda')) for l in LAYERS}
opts={l:torch.optim.AdamW([params[l]],lr=.032,betas=(.9,.95),eps=1e-15,weight_decay=0) for l in LAYERS}
checkpoint=out/'checkpoint.pt';progress=[];before=None;manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if checkpoint.exists() and not pilot:
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert state['dataset_sha256']==manifest_hash and state['R']==R
 for l in LAYERS:params[l].data.copy_(state['refs'][str(l)].to('cuda'));opts[l].load_state_dict(state['optimizers'][str(l)])
 progress=state['progress'];before=state['validation_before']
t=AutoTokenizer.from_pretrained('model',local_files_only=True)
m=Qwen3_5ForConditionalGeneration.from_pretrained('model',dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True).eval()
iface=qm.ALL_ATTENTION_FUNCTIONS;old=iface['sdpa'];raw_cache={}
def guard():
 free=shutil.disk_usage('C:/').free
 if free<24*2**30:print('DISK_GUARD_C',free,flush=True);sys.exit(3)
 if shutil.disk_usage(root).free<10*2**30:raise RuntimeError('W: reserve below 10 GiB')
def tokens(row):
 if row['source']=='pg19':
  blob=Path(row['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==row['sha256'];text=blob.decode('utf8',errors='ignore')
 else:
  split=row['split']
  if split not in raw_cache:raw_cache[split]=Path(f'data/wikitext2_{split}.txt').read_text(encoding='utf8')
  text=raw_cache[split][row['start_char']:row['end_char']];assert hashlib.sha256(text.encode()).hexdigest()==row['sha256']
 ids=t(text,add_special_tokens=False).input_ids[:row['window_tokens']];assert len(ids)==row['window_tokens']
 seed=int.from_bytes(hashlib.sha256(('queries-v1:'+row['document_id']).encode()).digest()[:8],'big')
 positions=sorted(random.Random(seed).sample(range(128,len(ids)),64))
 return ids,positions
def process(row,training,step=0):
 guard();ids,positions=tokens(row);pos=torch.tensor(positions,device='cuda');values={};start=time.perf_counter()
 def hook(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
  guard();lid=module.layer_idx;assert lid in params and lid not in values
  data=(key[0].detach(),query[0,:,pos,:].detach(),pos,scaling)
  with torch.enable_grad() if training else torch.no_grad():
   if training:
    opts[lid].zero_grad(set_to_none=True)
    for group in opts[lid].param_groups:group['lr']=.032*wsd_multiplier(step,total)
   loss,details=objective(data,params[lid],max_levels=6)
   if not torch.isfinite(loss):raise RuntimeError('nonfinite KL')
   if training:
    loss.backward()
    if not torch.isfinite(params[lid].grad).all():raise RuntimeError('nonfinite gradient')
    torch.nn.utils.clip_grad_norm_([params[lid]],1.);opts[lid].step()
   values[lid]={'sum_kl':float(loss.detach()),'levels':details}
  return old(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
 iface._local_mapping['sdpa']=hook
 try:
  with torch.no_grad():
   torch.cuda.empty_cache();x=torch.tensor([ids],device='cuda')
   result=m(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1)
   torch.cuda.synchronize();del x,result
 finally:iface._local_mapping.pop('sdpa',None)
 assert tuple(values)==LAYERS
 return {'document_id':row['document_id'],'source':row['source'],'length':len(ids),'input_sha256':hashlib.sha256(str(ids).encode()).hexdigest(),'sampled_positions':positions,'losses':values,'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
def validate():return [process(row,False) for row in valid]
if pilot:
 torch.cuda.reset_peak_memory_stats()
 for step,row in enumerate((pg[0],wiki[0])):
  z=process(row,True,step);print('PILOT',json.dumps({k:z[k] for k in ('document_id','length','seconds','peak_allocated_gib')}),flush=True)
 sys.exit(0)
if before is None:before=validate();print('VALIDATION_BEFORE_COMPLETE',len(before),flush=True)
for epoch in range(2):
 for index,row in enumerate(train):
  step=epoch*len(train)+index
  if step<len(progress):continue
  if step!=len(progress):raise RuntimeError('checkpoint order mismatch')
  torch.cuda.reset_peak_memory_stats();result=process(row,True,step);result.update({'epoch':epoch,'index':index,'lr':.032*wsd_multiplier(step,total)});progress.append(result)
  state={'R':R,'refs':{str(l):params[l].detach().cpu() for l in LAYERS},'initial':init,'optimizers':{str(l):opts[l].state_dict() for l in LAYERS},'progress':progress,'validation_before':before,'dataset_sha256':manifest_hash}
  tmp=str(checkpoint)+'.partial';torch.save(state,tmp);os.replace(tmp,checkpoint)
  print('TRAIN',json.dumps({k:result[k] for k in ('document_id','source','length','epoch','index','lr','seconds','peak_allocated_gib')}),flush=True)
after=validate();refs=torch.zeros(32,4,R,256)
for layer in LAYERS:refs[layer]=params[layer].detach().cpu()
tmp=str(out/'router.pt')+'.partial';torch.save({'refs':refs,'initial':init,'R':R},tmp);os.replace(tmp,out/'router.pt')
meta={'R':R,'L':6,'C':4,'dataset_sha256':manifest_hash,'unique_train_documents':len(train),'unique_train_tokens':sum(x['window_tokens'] for x in train),'epochs':2,'teacher_token_exposures':2*sum(x['window_tokens'] for x in train),'sampled_positions_per_document_layer':64,'sampled_layer_query_rows':len(progress)*8*64,'updates_per_layer':len(progress),'optimizer':'independent layerwise AdamW beta=(.9,.95), eps=1e-15, wd=0, clip=1; peak LR .032, 20 warmup updates, last 20% linear to 1%','initialization':'seed 94121; shared normal std .2; R4 is first four of R16 for each layer/KV head','limitations':'single-document batches, 4K/16K contexts, repeated positions, 128 documents; not paper 32K/1B data','validation_before':before,'validation_after':after,'progress':progress}
tmp=str(out/'training.json')+'.partial';Path(tmp).write_text(json.dumps(meta,indent=2),encoding='utf8');os.replace(tmp,out/'training.json')
print('COMPLETE',json.dumps({'R':R,'passes':len(progress),'tokens':meta['unique_train_tokens'],'exposures':meta['teacher_token_exposures']}),flush=True)
