"""Triton gathered-KV attention, one program per query/head, inference only."""
import torch,triton,triton.language as tl

@triton.jit
def _attention(Q,K,V,I,O, H, N, S, D:tl.constexpr, GROUP:tl.constexpr,
               qs0,qs1,qs2,
               ks0,ks1,ks2,
               vs0,vs1,vs2,
               is0,is1,is2,
               SCALE:tl.constexpr, BK:tl.constexpr, BD:tl.constexpr):
 h=tl.program_id(0);q=tl.program_id(1);kv=h//GROUP
 d=tl.arange(0,BD); offs=tl.arange(0,BK)
 qv=tl.load(Q+h*qs0+q*qs1+d*qs2,mask=d<D,other=0).to(tl.float32)
 m=tl.full((),-float('inf'),tl.float32);l=tl.full((),0.,tl.float32);acc=tl.full((BD,),0.,tl.float32)
 for block in range(tl.cdiv(S,BK)):
  slot=block*BK+offs
  ids=tl.load(I+h*is0+q*is1+slot*is2,mask=slot<S,other=-1)
  valid=(slot<S)&(ids>=0)&(ids<N);safe=tl.maximum(ids,0)
  kvs=tl.load(K+kv*ks0+safe[:,None]*ks1+d[None,:]*ks2,mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
  score=tl.sum(kvs*qv[None,:],axis=1)*SCALE;score=tl.where(valid,score,-float('inf'))
  new_m=tl.maximum(m,tl.max(score,axis=0));alpha=tl.exp(m-new_m);p=tl.exp(score-new_m)
  vals=tl.load(V+kv*vs0+safe[:,None]*vs1+d[None,:]*vs2,mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
  acc=acc*alpha+tl.sum(p[:,None]*vals,axis=0);l=l*alpha+tl.sum(p,axis=0);m=new_m
 tl.store(O+(h*tl.num_programs(1)+q)*D+d,acc/l,mask=d<D)

def gathered_attention(query,key,value,ids,scale,block=16):
 # query [H,Q,D], key/value [KV,N,D], ids [H,Q,S], -1 padding.
 H,Q,D=query.shape;KV,N,_=key.shape
 assert D<=256 and H%KV==0 and ids.shape[:2]==(H,Q)
 out=torch.empty_like(query, memory_format=torch.contiguous_format)
 _attention[(H,Q)](query,key,value,ids,out,H,N,ids.shape[-1],D,H//KV,*query.stride(),*key.stride(),*value.stride(),*ids.stride(),scale,block,triton.next_power_of_2(D),num_warps=4)
 return out


@triton.jit
def _node_scores(Q,M,B,I,O, N, CAND, D:tl.constexpr, R:tl.constexpr, GROUP:tl.constexpr,
                 qs0,qs1,qs2,ms0,ms1,ms2,ms3,bs0,bs1,bs2,is0,is1,is2,
                 SCALE:tl.constexpr, BN:tl.constexpr, BD:tl.constexpr):
 h=tl.program_id(0);q=tl.program_id(1);block=tl.program_id(2);kv=h//GROUP
 d=tl.arange(0,BD);slot=block*BN+tl.arange(0,BN)
 ids=tl.load(I+h*is0+q*is1+slot*is2,mask=slot<CAND,other=-1);valid=(slot<CAND)&(ids>=0)&(ids<N);safe=tl.maximum(ids,0)
 qv=tl.load(Q+h*qs0+q*qs1+d*qs2,mask=d<D,other=0).to(tl.float32)
 best=tl.full((BN,),-float('inf'),tl.float32)
 for r in tl.static_range(R):
  v=tl.load(M+kv*ms0+safe[:,None]*ms1+r*ms2+d[None,:]*ms3,mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
  bias=tl.load(B+kv*bs0+safe*bs1+r*bs2,mask=valid,other=0).to(tl.float32)
  score=tl.sum(v*qv[None,:],axis=1)*SCALE+bias;best=tl.maximum(best,score)
 tl.store(O+(h*tl.num_programs(1)+q)*CAND+slot,tl.where(valid,best,-float('inf')),mask=slot<CAND)

def node_scores(query,mean,bias,candidates,groups,scale,block=8):
 H,Q,D=query.shape;KV,N,R,_=mean.shape;C=candidates.shape[-1]
 out=torch.empty((H,Q,C),device=query.device,dtype=torch.float32)
 if C:_node_scores[(H,Q,triton.cdiv(C,block))](query,mean,bias,candidates,out,N,C,D,R,groups,*query.stride(),*mean.stride(),*bias.stride(),*candidates.stride(),scale,block,triton.next_power_of_2(D),num_warps=4)
 return out
