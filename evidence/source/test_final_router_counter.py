"""Tiny hybrid Qwen checks exact prefill/decode causal counters."""
import torch
from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
from final_router_counter import CountingRouter
from zha_l6_reference import installed
torch.manual_seed(6202)
c=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention']*3+['full_attention'],max_position_embeddings=128)
c._attn_implementation='sdpa';m=Qwen3_5ForCausalLM(c).eval().cuda()
refs=torch.randn(4,2,3,8,device='cuda')*.2;x=torch.randint(1,64,(1,37),device='cuda');mask=torch.ones_like(x)
for p in (1.,.7):
 r=CountingRouter(refs,p=p,rotary_dim=2,theta=10000.,tile=8)
 with torch.no_grad(),installed(r):
  a=m(x[:,:36],attention_mask=mask[:,:36],use_cache=True)
  pre=(r.selected_keys,r.available_keys,r.query_heads)
  m(x[:,36:],attention_mask=mask,past_key_values=a.past_key_values,use_cache=True)
 assert pre[1]==4*sum(range(1,37)) and pre[2]==4*36
 assert r.available_keys-pre[1]==4*37 and r.query_heads-pre[2]==4
 assert 0<pre[0]<=pre[1] and 0<r.selected_keys-pre[0]<=r.available_keys-pre[1]
 if p==1:assert r.selected_keys==r.available_keys
 assert not r.fallbacks
print('causal prefill/decode counters and full-selection control PASS')
