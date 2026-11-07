import torch
from vector_router import rotate

def update_tree(levels, keys, refs, chunk, scale, rotary_dim, theta):
    """Append completed nodes; existing keys and reference parameters must be unchanged."""
    kv,T,d=keys.shape
    target=(T-1)//chunk
    old=levels[0][0].shape[1] if levels else 0
    if target<old: raise ValueError('tree cannot shrink')
    if target==old: return levels
    result=list(levels)
    positions=torch.arange(old,target,device=keys.device)*chunk+chunk
    ref=rotate(refs,positions,rotary_dim,theta)
    blocks=keys[:,1+old*chunk:1+target*chunk].float().reshape(kv,target-old,chunk,d)
    logits=torch.einsum('knrd,kncd->knrc',ref,blocks)*scale
    p=logits.softmax(-1)
    mean=torch.einsum('knrc,kncd->knrd',p,blocks)
    bias=-(p*p.clamp_min(1e-30).log()).sum(-1)
    if result: result[0]=(torch.cat((result[0][0],mean),1),torch.cat((result[0][1],bias),1))
    else: result.append((mean,bias))
    level=1; width=chunk
    while result[level-1][0].shape[1]>=chunk:
        width*=chunk
        childmean,childbias=result[level-1]
        count=childmean.shape[1]//chunk
        prev=result[level][0].shape[1] if level<len(result) else 0
        if count>prev:
            child=childmean[:,prev*chunk:count*chunk].reshape(kv,count-prev,chunk,refs.shape[1],d).permute(0,1,3,2,4)
            cb=childbias[:,prev*chunk:count*chunk].reshape(kv,count-prev,chunk,refs.shape[1]).permute(0,1,3,2)
            ref=rotate(refs,torch.arange(prev,count,device=keys.device)*width+width,rotary_dim,theta)
            logits=(child*ref[:,:,:,None,:]).sum(-1)*scale+cb
            p=logits.softmax(-1)
            mean=(p[...,None]*child).sum(-2)
            bias=-(p*p.clamp_min(1e-30).log()).sum(-1)+(p*cb).sum(-1)
            if level<len(result): result[level]=(torch.cat((result[level][0],mean),1),torch.cat((result[level][1],bias),1))
            else: result.append((mean,bias))
        level+=1
    return result
