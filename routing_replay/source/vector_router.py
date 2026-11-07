"""Batched hierarchical router with tiled SDPA, experimental text-only inference.

All candidates at each tree level are scored in parallel; attention is dense
SDPA with a sparse selection mask. This is not a subquadratic sparse kernel.
Tree summaries are cached between completions and rebuilt at chunk boundaries.
"""
import contextlib,torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
DENSE=qm.eager_attention_forward

def rotate(refs,positions,rotary_dim,theta):
 # refs [KV,R,D], positions [N]; output [KV,N,R,D]
 inv=theta**(-torch.arange(0,rotary_dim,2,device=refs.device,dtype=torch.float32)/rotary_dim)
 f=positions.float()[:,None]*inv;cos=torch.cat((f.cos(),f.cos()),-1)[None,:,None,:];sin=torch.cat((f.sin(),f.sin()),-1)[None,:,None,:]
 x=refs[:,None,:,:rotary_dim];a,b=x.chunk(2,-1)
 return torch.cat((x*cos+torch.cat((-b,a),-1)*sin,refs[:,None,:,rotary_dim:].expand(-1,len(positions),-1,-1)),-1)

def tree_build(keys,refs,chunk,scale,rotary_dim,theta):
 # keys [KV,T,D], exclude token zero; [KV,N,R,D] and [KV,N,R]
 kv,T,d=keys.shape;N=(T-1)//chunk
 if not N:return []
 ref=rotate(refs,torch.arange(1,1+N*chunk,chunk,device=keys.device)+chunk-1,rotary_dim,theta)
 chunks=keys[:,1:1+N*chunk].float().reshape(kv,N,chunk,d)
 logits=torch.einsum('knrd,kncd->knrc',ref,chunks)*scale
 prob=logits.softmax(-1);mean=torch.einsum('knrc,kncd->knrd',prob,chunks)
 bias=-(prob*prob.clamp_min(1e-30).log()).sum(-1);levels=[(mean,bias)]
 width=chunk
 while N>=chunk:
  parents=N//chunk;width*=chunk
  child=mean[:,:parents*chunk].reshape(kv,parents,chunk,refs.shape[1],d).permute(0,1,3,2,4)
  cb=bias[:,:parents*chunk].reshape(kv,parents,chunk,refs.shape[1]).permute(0,1,3,2)
  ref=rotate(refs,torch.arange(1,1+parents*width,width,device=keys.device)+width-1,rotary_dim,theta)
  logits=(child*ref[:,:,:,None,:]).sum(-1)*scale+cb
  p=logits.softmax(-1);mean=(p[...,None]*child).sum(-2);bias=-(p*p.clamp_min(1e-30).log()).sum(-1)+(p*cb).sum(-1)
  levels.append((mean,bias));N=parents
 return levels

def select_mask(query,levels,end,chunk,p,groups,scale):
 # query [H,Q,D]; end [Q] exclusive. Return selected raw-key boolean [H,Q,Kmax].
 H,Q,d=query.shape;Kmax=int(end[-1]);local=1+((end-2).clamp_min(0)//chunk)*chunk
 local=torch.where(end>1,local,torch.ones_like(local));selected=None
 for level in reversed(range(len(levels))):
  mean,bias=levels[level];N=mean.shape[1];width=chunk**(level+1)
  eligible=(1+(torch.arange(N,device=query.device)+1)*width)[None,:]<=local[:,None]
  candidate=eligible[None,:,:].expand(H,-1,-1)
  if selected is not None:
   parent_count=(local-1)//(width*chunk)
   idx=torch.arange(N,device=query.device)//chunk
   expanded=selected[:,:,idx.clamp(max=selected.shape[-1]-1)]
   orphan=idx[None,:]>=parent_count[:,None]
   candidate=candidate & (expanded|orphan[None,:,:])
  perhead=mean.repeat_interleave(groups,0);b=bias.repeat_interleave(groups,0)
  scores=torch.einsum('hqd,hnrd->hqnr',query.float(),perhead)*scale+b[:,None,:,:]
  scores=scores.amax(-1).masked_fill(~candidate,float('-inf'))
  sorted_scores,order=scores.sort(dim=-1,descending=True)
  valid=torch.isfinite(sorted_scores);safe=torch.where(valid,sorted_scores,torch.full_like(sorted_scores,-1e30))
  probs=safe.softmax(-1)*valid
  before=probs.cumsum(-1)-probs
  chosen=valid & (before < p if p<1 else valid)
  selected=torch.zeros_like(chosen).scatter(-1,order,chosen)
 if selected is None:leaves=torch.zeros((H,Q,0),device=query.device,dtype=torch.bool)
 else:leaves=selected.repeat_interleave(chunk,-1)
 indices=torch.arange(Kmax,device=query.device)
 keep=(indices[None,:]==0)|((indices[None,:]>=local[:,None])&(indices[None,:]<end[:,None]))
 keep=keep[None,:,:].expand(H,-1,-1).clone()
 usable=min(Kmax-1,leaves.shape[-1])
 if usable>0:keep[:,:,1:1+usable]|=leaves[:,:,:usable]
 return keep & (indices[None,None,:]<end[None,:,None])

class VectorRouter:
 def __init__(self,refs,chunk=4,p=.9,rotary_dim=64,theta=1e7,tile=16):
  self.refs,self.chunk,self.p,self.rotary_dim,self.theta,self.tile=refs,chunk,p,rotary_dim,theta,tile;self.states={};self.stats=None
 def reset(self):self.states.clear()
 def forward(self,module,query,key,value,mask,scaling,dropout=0.,**kwargs):
  if module.training or dropout:raise ValueError('inference only')
  B,H,Q,D=query.shape;K=key.shape[-2];groups=module.num_key_value_groups
  supported=B==1 and mask is not None
  if supported:
   # One vectorized check for exact zero causal prefix and masked future.
   ends=torch.arange(K-Q+1,K+1,device=query.device)
   pos=torch.arange(K,device=query.device)
   expected=pos[None,:]<ends[:,None];actual=mask[0,0,:,:K]
   supported=bool(torch.all(torch.where(expected,actual==0,actual < -1e4)))
  if not supported:
   self.reset();return DENSE(module,query,key,value,mask,scaling,dropout=dropout,**kwargs)
  lid=module.layer_idx;state=self.states.get(lid);old=0 if state is None else state['length']
  same=state is not None and K>old and torch.equal(key[:,:,:old],state['keys'])
  complete=(K-1)//self.chunk
  if not same or state['complete']!=complete:
   levels=tree_build(key[0],self.refs[lid],self.chunk,scaling,self.rotary_dim,self.theta)
  else:levels=state['levels']
  self.states[lid]={'length':K,'complete':complete,'keys':key.detach(),'levels':levels}
  k=key.repeat_interleave(groups,1);v=value.repeat_interleave(groups,1);parts=[]
  for begin in range(0,Q,self.tile):
   endidx=min(Q,begin+self.tile);ends=torch.arange(K-Q+begin+1,K-Q+endidx+1,device=query.device)
   keep=select_mask(query[0,:,begin:endidx],levels,ends,self.chunk,self.p,groups,scaling)
   kk=k[:,:,:int(ends[-1])];vv=v[:,:,:int(ends[-1])]
   y=F.scaled_dot_product_attention(query[:,:,begin:endidx],kk,vv,attn_mask=keep.unsqueeze(0),dropout_p=0.,scale=scaling)
   parts.append(y)
   if self.stats is not None:
    logits=(query[0,:,begin:endidx].float()@kk[0].float().transpose(-1,-2))*scaling
    pos=torch.arange(kk.shape[-2],device=query.device);logits=logits.masked_fill(pos[None,None,:]>=ends[None,:,None],float('-inf'))
    mass=(logits.softmax(-1)*keep).sum(-1);count=keep.sum(-1)
    self.stats.append((mass.detach().cpu(),count.detach().cpu(),ends.detach().cpu()))
  return torch.cat(parts,2).transpose(1,2).contiguous(),None

@contextlib.contextmanager
def installed(router):
 old=qm.eager_attention_forward;router.reset();qm.eager_attention_forward=router.forward
 try:yield
 finally:qm.eager_attention_forward=old;router.reset()
