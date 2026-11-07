"""Shared initialization and WSD schedule for fixed-L6 R4/R16 controls."""
import torch

LAYERS=(3,7,11,15,19,23,27,31)

def initial_references(R,seed=94121):
    if R not in (4,16):raise ValueError('R must be 4 or 16')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    full={layer:torch.randn(4,16,256,generator=generator)*.2 for layer in LAYERS}
    return {layer:full[layer][:,:R].clone() for layer in LAYERS}

def wsd_multiplier(step,total,warmup=20):
    """20 warmup updates, final 20% linear decay to 1% of peak."""
    if not 0<=step<total or total<=warmup:raise ValueError('invalid update')
    decay_start=int(total*.8)
    if step<warmup:return (step+1)/warmup
    if step<decay_start:return 1.
    return 1.-.99*(step-decay_start)/(total-1-decay_start)
