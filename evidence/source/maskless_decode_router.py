"""Batched selective tree traversal and gathered sparse KV attention.

Experimental eager hook. Descends only selected parents plus parentless nodes;
selected KV blocks are gathered per query and head for SDPA. Uses PyTorch ops,
not a fused kernel; ragged compaction synchronizes once per tree level per tile.
"""
import contextlib,torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from vector_router import tree_build
from incremental_tree import update_tree
from sparse_kernel import gathered_attention,node_scores
DENSE=qm.ALL_ATTENTION_FUNCTIONS['sdpa']

def select_indices(query,levels,end,chunk,p,groups,scale,window=128):
 H,Q,D=query.shape;device=query.device;local=1+((end-window-1).clamp_min(0)//chunk)*chunk
 local=torch.where(end>1,local,torch.ones_like(local));selected=None
 for level in reversed(range(len(levels))):
  mean,bias=levels[level];N=mean.shape[1];width=chunk**(level+1)
  eligible=(local-1)//width
  if selected is None:
   candidates=torch.arange(N,device=device)[None,None,:].expand(H,Q,-1)
  else:
   off=torch.arange(chunk,device=device)
   kids=(selected[:,:,:,None]*chunk+off).reshape(H,Q,-1)
   kids=torch.where((selected[:,:,:,None]>=0).expand(-1,-1,-1,chunk).reshape(H,Q,-1),kids,-1)
   orphan_start=((local-1)//(width*chunk))*chunk
   orphan=orphan_start[:,None]+torch.arange(chunk-1,device=device)[None,:]
   orphan=torch.where(orphan<eligible[:,None],orphan,-1)[None,:,:].expand(H,-1,-1)
   candidates=torch.cat((kids,orphan),dim=-1)
  valid=(candidates>=0)&(candidates<eligible[None,:,None])&(candidates<N)
  candidates=torch.where(valid,candidates,-1)
  scores=node_scores(query,mean,bias,candidates,groups,scale)
  sorted_scores,order=scores.sort(-1,descending=True);sorted_idx=candidates.gather(-1,order)
  valid_sorted=torch.isfinite(sorted_scores);safe=torch.where(valid_sorted,sorted_scores,torch.full_like(sorted_scores,-1e30))
  probs=safe.softmax(-1)*valid_sorted;before=probs.cumsum(-1)-probs
  keep=valid_sorted & ((before<p) if p<1 else valid_sorted)
  max_keep=int(keep.sum(-1).max())
  selected=torch.where(keep[:,:,:max_keep],sorted_idx[:,:,:max_keep],-1)
 # Convert selected leaf blocks plus sink and local chunk to token indices.
 if selected is None:
  leaf=torch.empty(H,Q,0,device=device,dtype=torch.long)
 else:
  offsets=torch.arange(chunk,device=device)
  leaf=(1+selected[:,:,:,None]*chunk+offsets).reshape(H,Q,-1)
  leaf=torch.where((selected[:,:,:,None]>=0).expand(-1,-1,-1,chunk).reshape(H,Q,-1),leaf,-1)
 offsets=torch.arange(window+chunk,device=device)
 local_tokens=local[:,None]+offsets
 local_tokens=torch.where(local_tokens<end[:,None],local_tokens,-1)[None,:,:].expand(H,-1,-1)
 sink=torch.zeros(H,Q,1,device=device,dtype=torch.long)
 ids=torch.cat((sink,leaf,local_tokens),-1)
 return ids

class WindowFusedRouter:
 def __init__(self,refs,chunk=4,p=.9,rotary_dim=64,theta=1e7,tile=16,window=128):
  self.refs,self.chunk,self.p,self.rotary_dim,self.theta,self.tile=refs,chunk,p,rotary_dim,theta,tile;self.window=window;self.states={};self.stats=None
 def reset(self):self.states.clear()
 def forward(self,module,query,key,value,mask,scaling,dropout=0.,**kwargs):
  if module.training or dropout:raise ValueError('inference only')
  B,H,Q,D=query.shape;K=key.shape[-2];groups=module.num_key_value_groups
  # Narrow contract: a missing SDPA mask and one query mean every cached key is visible.
  # Explicit masks, prefill, padded batches and unsupported modes use native SDPA.
  supported=B==1 and Q==1 and mask is None
  if not supported:
   self.reset();return DENSE(module,query,key,value,mask,scaling=scaling,dropout=dropout,**kwargs)
  lid=module.layer_idx;state=self.states.get(lid);old=0 if state is None else state['length']
  same=state is not None and K>old and torch.equal(key[:,:,:old],state['keys']);complete=(K-1)//self.chunk
  if not same:levels=tree_build(key[0],self.refs[lid],self.chunk,scaling,self.rotary_dim,self.theta)
  elif state['complete']!=complete:levels=update_tree(state['levels'],key[0],self.refs[lid],self.chunk,scaling,self.rotary_dim,self.theta)
  else:levels=state['levels']
  self.states[lid]={'length':K,'complete':complete,'keys':key.detach(),'levels':levels}
  parts=[];kvindex=(torch.arange(H,device=query.device)//groups)[:,None,None]
  for begin in range(0,Q,self.tile):
   stop=min(Q,begin+self.tile);ends=torch.arange(K-Q+begin+1,K-Q+stop+1,device=query.device)
   ids=select_indices(query[0,:,begin:stop],levels,ends,self.chunk,self.p,groups,scaling,self.window)
   valid=ids>=0;idx=ids.clamp(min=0,max=K-1)
   if getattr(self,'budget_log',None) is not None and not any(item['layer']==lid for item in self.budget_log):
    count=valid.sum(-1).float();self.budget_log.append({'layer':lid,'key_length':K,'head_min':int(count.min()),'head_max':int(count.max()),'head_mean':float(count.mean()),'padded_slots_per_head':ids.shape[-1]})
   y=gathered_attention(query[0,:,begin:stop],key[0],value[0],ids,scaling)
   parts.append(y)
   if self.stats is not None:
    dense=(query[0,:,begin:stop].float()@key[0].repeat_interleave(groups,0).float().transpose(-1,-2))*scaling
    pos=torch.arange(K,device=query.device);dense=dense.masked_fill(pos[None,None,:]>=ends[None,:,None],float('-inf'))
    probs=dense.softmax(-1);mass=probs.gather(-1,idx).masked_fill(~valid,0).sum(-1)
    self.stats.append((mass.detach().cpu(),valid.sum(-1).detach().cpu(),ends.detach().cpu()))
  return torch.cat(parts,1).unsqueeze(0).transpose(1,2).contiguous(),None

@contextlib.contextmanager
def installed(router):
 iface=qm.ALL_ATTENTION_FUNCTIONS
 old=iface._local_mapping.get('sdpa');router.reset();iface._local_mapping['sdpa']=router.forward
 try:yield
 finally:
  if old is None:iface._local_mapping.pop('sdpa',None)
  else:iface._local_mapping['sdpa']=old
  router.reset()


