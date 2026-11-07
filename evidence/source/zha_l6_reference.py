"""Slow, explicit C4/L6 adaptive-top-p reference for Qwen text attention.

The native backend remains the fallback for batches, padding, explicit unusual
masks and training. This is a correctness implementation, not a fast kernel.
"""
import contextlib
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from vector_router import tree_build

DENSE=qm.ALL_ATTENTION_FUNCTIONS['sdpa']

def make_tree(keys,refs,scale,rotary_dim=64,theta=1e7,chunk=4,max_levels=6):
    # Truncation supplies the exact fixed L6 geometry. Reference implementation
    # may construct extra higher levels before discarding them; it is slow.
    return tree_build(keys,refs,chunk,scale,rotary_dim,theta)[:max_levels]

def selected_indices(query,levels,end,kv_head,scale,p=.9,chunk=4):
    """Independent per-query-head top-p over complete reachable nodes."""
    if not 0<p<=1:raise ValueError('p must be in (0,1]')
    if end<1:raise ValueError('at least one visible key required')
    # Sink is index 0. Retain visible keys in the current leaf chunk exactly.
    local=1+((end-2)//chunk)*chunk if end>1 else 1
    selected=None
    for level in range(len(levels)-1,-1,-1):
        means,bias=levels[level];width=chunk**(level+1)
        eligible=min(means.shape[1],(local-1)//width)
        if selected is None:
            candidates=list(range(eligible))
        else:
            parent_complete=(local-1)//(width*chunk)
            candidates=[x*chunk+j for x in selected for j in range(chunk) if x*chunk+j<eligible]
            candidates += list(range(parent_complete*chunk,eligible))
        if not candidates:
            selected=[];continue
        assert len(candidates)==len(set(candidates))
        idx=torch.tensor(candidates,device=query.device,dtype=torch.long)
        score=(torch.einsum('nrd,d->nr',means[kv_head,idx],query.float())*scale+bias[kv_head,idx]).amax(-1)
        order=torch.argsort(score,descending=True,stable=True)
        probs=torch.softmax(score[order].float(),dim=0)
        n=len(candidates) if p==1 else int((probs.cumsum(0)-probs < p).sum())
        selected=idx[order[:n]].tolist()
    ids=[0]+list(range(local,end))
    if selected is not None:
        for leaf in selected:ids.extend(range(1+leaf*chunk,1+(leaf+1)*chunk))
    result=sorted(ids)
    assert len(result)==len(set(result)) and result[0]==0 and result[-1]<end
    return torch.tensor(result,device=query.device,dtype=torch.long)

def _simple_causal_ends(mask,B,Q,K,device):
    """Return visible prefix ends, or None for any unsupported mask."""
    if B!=1:return None
    ends=torch.arange(K-Q+1,K+1,device=device)
    if bool((ends<1).any()):return None
    if mask is None:return ends
    if mask.ndim!=4 or mask.shape[0]!=1 or mask.shape[1]!=1 or mask.shape[-2] not in (1,Q) or mask.shape[-1]<K:return None
    rows=mask[0,0,:,:K]
    if rows.shape[0]==1:rows=rows.expand(Q,-1)
    allowed=torch.arange(K,device=device)[None,:]<ends[:,None]
    if mask.dtype==torch.bool:
        # PyTorch SDPA boolean masks use True for visible positions.
        if not bool(torch.equal(rows,allowed)):return None
    elif not bool(torch.all(torch.where(allowed,rows==0,rows < -1e4))):return None
    return ends

class L6Router:
    def __init__(self,refs,p=.9,chunk=4,rotary_dim=64,theta=1e7):
        self.refs=refs;self.p=p;self.chunk=chunk;self.rotary_dim=rotary_dim;self.theta=theta;self.records=[]
    def reset(self):pass
    def forward(self,module,query,key,value,mask,scaling,dropout=0.,**kwargs):
        if module.training or dropout:raise ValueError('inference only')
        B,H,Q,D=query.shape;K=key.shape[-2];groups=module.num_key_value_groups
        ends=_simple_causal_ends(mask,B,Q,K,query.device)
        if ends is None:return DENSE(module,query,key,value,mask,scaling=scaling,dropout=dropout,**kwargs)
        assert H==key.shape[1]*groups
        levels=make_tree(key[0],self.refs[module.layer_idx],scaling,self.rotary_dim,self.theta,self.chunk,6)
        out=torch.empty_like(query)
        for t in range(Q):
            end=int(ends[t])
            for h in range(H):
                kv=h//groups;ids=selected_indices(query[0,h,t],levels,end,kv,scaling,self.p,self.chunk)
                logits=(key[0,kv,ids].float()@query[0,h,t].float())*scaling
                probs=logits.softmax(-1)
                out[0,h,t]=(probs@value[0,kv,ids].float()).to(out.dtype)
                self.records.append({'layer':module.layer_idx,'query':t,'head':h,'end':end,'keys':len(ids)})
        return out.transpose(1,2).contiguous(),None

@contextlib.contextmanager
def installed(router):
    iface=qm.ALL_ATTENTION_FUNCTIONS;old=iface._local_mapping.get('sdpa');iface._local_mapping['sdpa']=router.forward
    try:yield
    finally:
        if old is None:iface._local_mapping.pop('sdpa',None)
        else:iface._local_mapping['sdpa']=old
