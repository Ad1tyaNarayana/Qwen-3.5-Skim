REAL ROUTING DATA FOR BOTH FIGURES

teacher/doc0_layer3.pt: saved dense keys [4,16384,256], sampled queries [16,64,256], 64 positions and attention scale .0625. BF16 tensors, ~34 MB. No values tensor; sufficient for routing and true retained mass, not attention outputs or generated answers.
checkpoints/router.pt: trained R16 references AND exact original initialization.
source/: matching tree-builder and CPU reference selector.
original_replays/: both original GIFs/posters, full trace.json, capture and rendering source. Trace includes every candidate/selected node, selected original key index and retained mass.

Run: python replay_from_tensors.py
Requires compatible PyTorch and Transformers (original: PyTorch 2.14.1+cu130, Transformers 5.21.0, Python 3.14.2). Runs on CPU, no base-model download or GPU. Writes recomputed/trace.json; validates selections against the original CPU reference. The local teacher_file provenance path differs after relocation; hashes and routing decisions should match. Numerical variation may occur across environments.

historical_capture/plan_large_capture.py is the ORIGINAL tensor-capture script. Doc0 uses the WikiText-2 TRAIN token window beginning at offset 600000, length 16384; 64 sampled queries, seed 2411. It captures all eight full-attention layers; only layer3 is bundled because both figures use it. Full capture metadata is included. Rerunning capture requires the original model/corpus files and adapted paths.

The animations show the final eight sampled query positions, layer3/head0 zero-based; these are NOT consecutive generation steps. The tree figure uses the last snapshot. This is an earlier TRAIN-split cache that may span article boundaries, not held-out data or the final 96-prompt benchmark. It is diagnostic material, not new quality evidence. Timing/node spacing may be schematic; keep actual selected identities unchanged. Dense retained mass is neither answer quality nor the estimated p threshold.

The historical rendering script has absolute output/font paths; adapt them for your renderer. You can recolor the existing trace without rerunning capture. No base-model weights or corpus text are redistributed.
