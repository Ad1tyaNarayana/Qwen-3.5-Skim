"""Count exact causal head-query key opportunities without changing routing."""
from zha_l6_sparse import L6SparseRouter
from zha_l6_reference import _simple_causal_ends

class CountingRouter(L6SparseRouter):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.available_keys=0
 def forward(self,module,query,key,value,mask,scaling,dropout=0.,**kwargs):
  B,H,Q,D=query.shape;ends=_simple_causal_ends(mask,B,Q,key.shape[-2],query.device)
  result=super().forward(module,query,key,value,mask,scaling,dropout=dropout,**kwargs)
  if ends is not None:self.available_keys+=H*int(ends.sum())
  return result
