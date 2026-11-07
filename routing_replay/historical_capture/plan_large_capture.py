import json,hashlib,random,torch,os,shutil
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm

torch.manual_seed(2411);torch.set_num_threads(4);t=AutoTokenizer.from_pretrained('model',local_files_only=True)
paths={'train':'data/wikitext2_train.txt','valid':'data/wikitext2_valid.txt'}
docs=[('train',600000+50000*i) for i in range(16)]+[('valid',220000),('valid',240000)]
length=16384;sample=64;layers=(3,7,11,15,19,23,27,31);base='results/plan_large';metadata=base+'/teacher_metadata.json';meta=[]
if os.path.exists(metadata):
 oldmeta=json.load(open(metadata));meta=oldmeta['docs']
for i in range(len(meta)):
 assert meta[i]['split']==docs[i][0] and meta[i]['offset']==docs[i][1] and len(meta[i]['positions'])==sample
expected=18*8*(4*length*256*2+16*sample*256*2)
print('estimated key+sampled-query payload bytes',expected,'reserve bytes',20*2**30,flush=True)
m=Qwen3_5ForConditionalGeneration.from_pretrained('model',dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True).eval()
iface=qm.ALL_ATTENTION_FUNCTIONS;old=iface['sdpa'];token_cache={}
try:
 for doc_id,(split,offset) in enumerate(docs):
  if doc_id<len(meta):continue
  if shutil.disk_usage('.').free < 20*2**30:raise RuntimeError('disk reserve guard: below 20 GiB')
  if split not in token_cache:
   raw=open(paths[split],encoding='utf8').read();token_cache[split]=t(raw,add_special_tokens=False).input_ids
  ids=token_cache[split][offset:offset+length]
  if len(ids)!=length:raise ValueError('insufficient tokens')
  rng=random.Random(2411+doc_id);positions=sorted(rng.sample(range(128,length),sample));pos=torch.tensor(positions,device='cuda')
  def hook(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
   q=query[0,:,pos,:].detach();k=key[0].detach();path=f'{base}/teacher/doc{doc_id}_layer{module.layer_idx}.pt';tmp=path+'.partial'
   torch.save({'key':k.cpu(),'queries':q.cpu(),'positions':positions,'scale':scaling},tmp);os.replace(tmp,path)
   return old(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
  iface._local_mapping['sdpa']=hook
  with torch.no_grad():
   x=torch.tensor([ids],device='cuda');torch.cuda.reset_peak_memory_stats();out=m(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1);torch.cuda.synchronize();del out,x
  iface._local_mapping.pop('sdpa',None)
  assert all(os.path.exists(f'{base}/teacher/doc{doc_id}_layer{l}.pt') for l in layers)
  info={'doc_id':doc_id,'role':'train' if doc_id<16 else 'validation' if doc_id==16 else 'test','split':split,'offset':offset,'length':length,'positions':positions,'token_sha256':hashlib.sha256(str(ids).encode()).hexdigest(),'source_sha256':hashlib.sha256(open(paths[split],'rb').read()).hexdigest(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
  meta.append(info);z={'model':'Qwen/Qwen3.5-4B-Base','model_revision':'1001bb4d826a52d1f399e183466143f4da7b741b','teacher':'frozen dense SDPA prefill','dtype':'bfloat16','format':'full keys plus sampled queries; targets recomputed from Q/K','docs':meta};tmp=metadata+'.partial';json.dump(z,open(tmp,'w'),indent=2);os.replace(tmp,metadata);print('captured',doc_id,info['role'],offset,'free',shutil.disk_usage('.').free,flush=True)
finally:iface._local_mapping.pop('sdpa',None)
