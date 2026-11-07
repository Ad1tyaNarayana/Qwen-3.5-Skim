"""Sampled dense-teacher multilevel KL with exact sink and incomplete chunk.

This is a bounded Qwen pilot, not a full reproduction of Zyphra training.
The same R16 references are shared across levels; complete C4 nodes use the
existing RoPE-corrected reference-query tree. The model and teacher use exactly
the same sink and incomplete-current-chunk logits at each level.
"""
import torch
from vector_router import tree_build

def load(path, device):
    d = torch.load(path, map_location=device, weights_only=True)
    return d['key'], d['queries'], torch.tensor(d['positions'], device=device), d['scale']

def objective(data, refs, chunk=4, rotary_dim=64, theta=1e7, max_levels=None):
    key, q, positions, scale = data
    heads, samples, _ = q.shape
    groups = heads // key.shape[0]
    assert heads == key.shape[0] * groups and q.shape[1] == len(positions)
    ends = positions + 1
    expanded = key.repeat_interleave(groups, 0)
    logits = torch.matmul(q.float(), expanded.float().transpose(-1, -2)) * scale
    levels = tree_build(key, refs, chunk, scale, rotary_dim, theta)
    if max_levels is not None:
        levels = levels[:max_levels]
    losses, details = [], {}
    for level, (mean, bias) in enumerate(levels):
        width = chunk ** (level + 1)
        n = mean.shape[1]
        if not n:
            continue
        pooled = logits[:, :, 1:1+n*width].reshape(heads, samples, n, width).logsumexp(-1)
        done = (ends-1) // width
        valid = torch.arange(n, device=q.device)[None, :] < done[:, None]
        good = valid.sum(-1) >= 1
        if not bool(good.any()):
            continue
        far_target = pooled.masked_fill(~valid[None], float('-inf'))
        mh = mean.repeat_interleave(groups, 0)
        bh = bias.repeat_interleave(groups, 0)
        far_model = (torch.einsum('hsd,hnrd->hsnr', q.float(), mh)*scale + bh[:, None]).amax(-1)
        far_model = far_model.masked_fill(~valid[None], float('-inf'))
        indices = torch.arange(key.shape[1], device=q.device)
        tail_start = 1 + done * width
        tail_valid = (indices[None, :] >= tail_start[:, None]) & (indices[None, :] < ends[:, None])
        tail = logits.masked_fill(~tail_valid[None], float('-inf')).logsumexp(-1, keepdim=True)
        sink = logits[:, :, :1]
        target_logits = torch.cat((sink, far_target, tail), dim=-1)[:, good]
        model_logits = torch.cat((sink, far_model, tail), dim=-1)[:, good]
        target_logp = target_logits.log_softmax(-1).detach()
        model_logp = model_logits.log_softmax(-1)
        target_p = target_logp.exp()
        # Avoid 0 * (-inf - -inf) in masked slots.
        difference = torch.where(target_p > 0, target_logp - model_logp, 0.)
        kl = (target_p * difference).sum(-1).mean()
        losses.append(kl)
        details[str(width)] = float(kl.detach())
    if not losses:
        raise ValueError('no complete nodes at any level')
    return torch.stack(losses).sum(), details
