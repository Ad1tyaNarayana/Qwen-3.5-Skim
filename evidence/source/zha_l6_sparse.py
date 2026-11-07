"""Grouped, gathered-KV six-level routing for prefill and decode.

Inference correctness prototype: tree rebuilds and dynamic selection remain
expensive. Unsupported masks use the native dense backend.
"""
import shutil
import torch
from zha_l6_reference import DENSE, _simple_causal_ends, make_tree
from maskless_decode_router import select_indices
from sparse_kernel import gathered_attention


class L6SparseRouter:
    def __init__(self, refs, p=.9, chunk=4, rotary_dim=64, theta=1e7,
                 tile=16, disk_guard_gib=None, measure_mass=False):
        if not 0 < p <= 1:
            raise ValueError('p must be in (0, 1]')
        self.refs, self.p, self.chunk = refs, p, chunk
        self.rotary_dim, self.theta, self.tile = rotary_dim, theta, tile
        self.disk_guard_gib = disk_guard_gib
        self.calls = []
        self.fallbacks = []
        self.selected_keys = 0
        self.query_heads = 0
        self.measure_mass = measure_mass
        self.mass_sum = 0.0
        self.mass_queries = 0

    def reset(self):
        pass

    def _guard(self):
        if self.disk_guard_gib is not None and shutil.disk_usage('C:\\').free < self.disk_guard_gib * 2**30:
            raise RuntimeError('DISK_GUARD: C: free space below configured reserve')

    def forward(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        if module.training or dropout:
            raise ValueError('inference only')
        B, H, Q, D = query.shape
        K = key.shape[-2]
        groups = module.num_key_value_groups
        ends = _simple_causal_ends(mask, B, Q, K, query.device)
        if ends is None:
            self.fallbacks.append({'layer':module.layer_idx,'Q':Q,'K':K,
                'mask_shape':None if mask is None else list(mask.shape),
                'mask_dtype':None if mask is None else str(mask.dtype),
                'mask_min':None if mask is None else float(mask.min()),
                'mask_max':None if mask is None else float(mask.max())})
            return DENSE(module, query, key, value, mask, scaling=scaling, dropout=dropout, **kwargs)
        assert H == key.shape[1] * groups
        group_outputs = []
        for kv in range(key.shape[1]):
            self._guard()
            keys, values = key[0, kv:kv+1], value[0, kv:kv+1]
            levels = make_tree(keys, self.refs[module.layer_idx][kv:kv+1], scaling,
                               self.rotary_dim, self.theta, self.chunk, 6)
            tiles = []
            for begin in range(0, Q, self.tile):
                self._guard()
                stop = min(Q, begin + self.tile)
                q = query[0, kv*groups:(kv+1)*groups, begin:stop]
                ids = select_indices(q, levels, ends[begin:stop], self.chunk,
                                     self.p, groups, scaling, window=1)
                self.selected_keys += int((ids >= 0).sum())
                self.query_heads += ids.shape[0] * ids.shape[1]
                if self.measure_mass and begin == 0:
                    # Compare to true dense mass for the first query in this
                    # forward call, on the same live Q/K trajectory.
                    logits = torch.einsum('hd,nd->hn', q[:,0].float(), keys[0].float()) * scaling
                    positions = torch.arange(K, device=query.device)
                    logits = logits.masked_fill(positions[None,:] >= ends[begin], float('-inf'))
                    probabilities = logits.softmax(-1)
                    sample = ids[:,0]
                    mass = probabilities.gather(-1, sample.clamp(min=0,max=K-1)).masked_fill(sample < 0,0).sum(-1)
                    self.mass_sum += float(mass.sum())
                    self.mass_queries += mass.numel()
                tiles.append(gathered_attention(q, keys, values, ids, scaling))
            group_outputs.append(torch.cat(tiles, dim=1))
        self.calls.append((module.layer_idx, Q, K))
        return torch.cat(group_outputs, dim=0).unsqueeze(0).transpose(1, 2).contiguous(), None


class SinkCurrentRouter:
    """Simple sparse baseline: sink plus the visible current C4 chunk."""
    def __init__(self, chunk=4):
        self.chunk = chunk
        self.calls = []
        self.fallbacks = []

    def reset(self):
        pass

    def forward(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        if module.training or dropout:
            raise ValueError('inference only')
        B,H,Q,D=query.shape; K=key.shape[-2]
        ends=_simple_causal_ends(mask,B,Q,K,query.device)
        if ends is None:
            self.fallbacks.append((module.layer_idx,Q,K))
            return DENSE(module,query,key,value,mask,scaling=scaling,dropout=dropout,**kwargs)
        local=torch.where(ends>1,1+((ends-2)//self.chunk)*self.chunk,torch.ones_like(ends))
        ids=local[:,None]+torch.arange(self.chunk,device=query.device)[None,:]
        ids=torch.where(ids<ends[:,None],ids,-1)
        ids=torch.cat((torch.zeros((Q,1),dtype=ids.dtype,device=ids.device),ids),dim=-1)
        ids=ids.unsqueeze(0).expand(H,-1,-1).contiguous()
        out=gathered_attention(query[0],key[0],value[0],ids,scaling)
        self.calls.append((module.layer_idx,Q,K))
        return out.unsqueeze(0).transpose(1,2).contiguous(),None
