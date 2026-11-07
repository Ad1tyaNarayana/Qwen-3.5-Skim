# Hierarchical attention in Qwen3.5-4B: methods, results and evidence

## 1. What we did and what we can conclude

We built an experimental, coarse-to-fine attention backend for the full-attention layers of **Qwen3.5-4B-Base**, trained small learned routing modules against the frozen model, and compared them with the unmodified model. The work was inspired by [Zyphra Hierarchical Attention (ZHA)](https://arxiv.org/html/2610.42112v2), but it is an adaptation to a different model and much smaller training budget, not a reproduction of the paper.

**Main finding:** we did not establish a general improvement in model quality or speed. The most recent directly paired test compared the previous R16 router with a checkpoint given one additional 32K-document update. They tied on partial-credit score in all six fresh cases. That is evidence about those six cases, not proof that the checkpoints are equivalent. Earlier, on 200 longer previously inspected cases using dense prefill and sparse decode, our learned router scored below dense in every task/length cell. An attention-mass oracle improved those scores at the tested key budgets, suggesting that selection quality can matter; it does not prove that our router can learn the oracle's choices.

**What is actually built:** a runtime hook and separate reference-query checkpoints, not a fused standalone model. A reference version and an experimental gathered-KV path implement adaptive top-p tree traversal with sparse prefill and decode for the supported text inputs. Unsupported masks fall back to dense. No optimized end-to-end speedup or full paper reproduction is demonstrated.

The accompanying package uses paths such as `code/...`, `data/...`, and `checkpoints/...` below. `SHA256SUMS.txt` records file hashes. The machine-readable `data/.../results.jsonl` files contain one record per branch, including task, target length, case index, expected answer, generated text and score. They are the source for the tables, rather than reconstructed illustrative examples. The package does not include the base model weights, teacher text corpora, software environment, credentials or unrelated personal files.

## 2. Background: what the hierarchy retrieves

Ordinary causal attention compares each query with every visible key, assigns softmax weights, and combines the corresponding values. Our system still stores the original KV cache. It adds a tree of summaries so a query can first compare coarse spans, then inspect only selected child spans, and finally attend to selected original keys and values.

In the C4, L6, R16 configuration, each node summarizes four children, the tree has at most six levels, and 16 learned reference queries per KV head produce alternative summaries. At each level, a query head scores the available complete nodes and retains the smallest highest-scoring prefix whose **estimated** probability mass reaches p; we used p=0.9 in the later gate. The first token and visible tokens in the current four-token chunk are kept separately. The same query head then evaluates selected children plus complete nodes whose parent is incomplete. Finally, one softmax is computed over the selected original keys. Code: `code/zha_l6_reference.py:13-48`, `code/maskless_decode_router.py:14-48`, `code/zha_l6_sparse.py:36-80`, `code/sparse_kernel.py:5-31`.

An approximate node score is a maximum of affine lower bounds on a chunk's log unnormalized attention mass. The reference queries are learned by matching the frozen model's dense attention distribution pooled into chunks at each level. The positional correction evaluates child summaries using the parent's rotated reference query. The important practical consequence: **p=0.9 is a target fraction of the router's estimated mass among candidates, not a guarantee that 90% of the model's true dense attention mass was retained.** See [ZHA sections III-A–III-C](https://arxiv.org/html/2610.42112v2#S3); implementation `code/vector_router.py:12-36`, `code/r16_multilevel_loss.py:15-58`.

This is different from [OptMem](https://github.com/VictorTaelin/OptMem), an external agent memory system organized around selected textual notes. OptMem does not by itself insert a learned hierarchy over the model's original KV cache. The user's requested model-level retrieval motivated the internal attention experiment; we did not establish that either approach is universally better.

## 3. Backbone, hardware and execution

The frozen [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) revision was `1001bb4d826a52d1f399e183466143f4da7b741b`. Its text configuration has 32 blocks: 24 Gated DeltaNet linear-attention blocks and eight full-attention blocks at zero-based indices **3, 7, 11, 15, 19, 23, 27, 31**. Each full-attention block has 16 query heads, four KV heads, dimension 256 and partial RoPE on 64 dimensions; output gating is enabled. `data/model/config.json` is the captured configuration. Only the full-attention backend was replaced; original model projections, normalization, RoPE, attention gates, recurrent blocks and weights remained active.

The actual machine used Windows, NVIDIA RTX 5070 Ti with 16,303 MiB reported dedicated VRAM, NVIDIA driver 617.42, Python 3.14.2, PyTorch 2.14.1+cu130, CUDA 13.0, Transformers 5.21.0 and Triton 3.8.0. The runtime used reference PyTorch fallbacks for Qwen's Gated DeltaNet operations because the optional optimized packages were absent. Results are specific to this environment. PyTorch allocation/reservation counters under Windows WDDM are not the same as dedicated GPU memory usage.

For scale, raw BF16 K/V across the eight full-attention layers costs about 32 KiB per token: 1 GiB at 32K or 2 GiB at 64K before model weights, tree summaries, recurrent state and allocator overhead. R16 FP32 summaries add substantial memory. These are arithmetic estimates, not measured peaks.

## 4. Design relative to Zyphra's paper

The paper's principal deep geometry is C4/L6/R16. It shares reference queries across tree levels and KV-sharing query heads, uses per-query-head adaptive top-p, retains the sink and current chunk, and trains only the router against frozen dense attention. Those mechanisms informed the later local reference. Shared references across levels are **not** a local deviation. [Paper architecture](https://arxiv.org/html/2610.42112v2#S3).

Confirmed differences and limitations:

* The paper's main Qwen experiments use Qwen3.6-35B-A3B; ours use the hybrid 4B model above. It also evaluates another backbone. Results do not transfer automatically.
* The paper reports 1,900 steps, batch 16, 32,768-token document-masked rows and 0.996 billion teacher-input tokens. Our controlled R4/R16 experiment used 128 documents, batch one, 4K/16K windows, 1.31 million unique tokens and two exposures. Our later 32K continuation was **one document and one update per layer**. [Paper training details](https://arxiv.org/html/2610.42112v2#A2.SS1).
* The paper does not specify enough to confirm identical reference initialization, sampled-query policy, clipping scope or all numerical details. We use a documented seeded normal initialization and 64 sampled positions per document/layer; these are local choices, not claimed paper details.
* Our later hook rebuilds FP32 summaries, uses dynamic PyTorch sorting and per-head Triton gathers, and keeps separate per-head masks. It does not implement the paper's efficient GQA-union loads or its incremental serving design. Unsupported padding, unusual masks and batches use dense fallback. The implementation covers a narrow batch-one text contract, not all multimodal, position, beam or cache scenarios.
* The paper says optimized kernels are not released and its attention-layer performance comparisons use a roofline model. Its H100 projections are **not measured end-to-end 5070 Ti speedups**. [Paper speed analysis](https://arxiv.org/html/2610.42112v2#S5).

Earlier project stages used other choices: dynamically deeper trees, aligned 128-token recent windows, fixed-beam/fixed-budget selectors, and dense-prefill/sparse-decode evaluation. Their results should not be mixed with the later C4/L6 adaptive sparse-prefill gate. `code/zha_l6_reference.py:13-16` computes the full tree then slices it to the fixed maximum, a correct reference geometry but extra construction work. `data/l6/training_protocol.json` and the checkpoint hashes record the later controlled setup.

## 5. What we tested first

The project progressed from synthetic tree mathematics through a Qwen runtime hook, selective node traversal and a Triton gathered-attention path. In 40 earlier related synthetic fact-insertion probes, dense SDPA answered 33 correctly, the original selective router 35, and a small WikiText-trained router 34. These probes shared templates and were already inspected; they are exploratory and not evidence of a general improvement. On eight 512-token WikiText test windows, 4,088 scored tokens gave mean NLL 2.41029 dense, 2.41540 original selective and 2.41317 WikiText-trained selective. See `data/historical/SELECTIVE_REPORT.md` for definitions, numerical controls and source-file names.

A separate propagation diagnostic used 12 short WikiText-based inputs of 108–258 tokens, including six related fact prompts. It replayed identical dense Q/K/V through the router, enabled routing in individual full-attention layers and then all eight, and measured downstream blocks while retaining all recurrent layers. Median direct retained mass by layer in prefill was 91.3–93.5%; routed attention-output relative L2 was 1.61–8.36%, compared with 0.23–0.79% in its p=1 numerical controls. The downstream Gated DeltaNet outputs did change, but these controls **do not prove that recurrent layers uniquely amplify routing errors**: residuals, MLPs, differing norms and p=1 numerical differences complicate attribution. These short prompts are not a long-context benchmark. See `data/historical/ABLATION_REPORT.md` and `data/historical/ablation_summary.json`.

## 6. The 200-case long-context study and causal oracle

The 200 cases use the pinned [NVIDIA RULER](https://github.com/NVIDIA/RULER) generator at commit `db43f92ff634bbce1c33d1fb15858aec4bba509a`: variable tracking (VT) and common-word extraction (CWE), with 40 cases per 16K/32K cell and 20 per 64K cell. They used the earlier R4 router, an aligned recent window of at least 128 tokens, dense prefill and sparse decode. Scores are partial credit from case-insensitive expected-answer substring matches; they are not binary exact-match accuracy. Source: `data/oracle/200_manifest.json`, `data/oracle/200_results.jsonl`, `code/plan_200_eval.py`, `data/oracle/200_summary.md`.

We then reran the same already-inspected cases with three live decode branches: the learned router, an unrestricted token oracle and a C4-leaf oracle. At each autoregressive step the oracle computed dense attention from **its own current Q/K**, with no answer labels or future keys, using the learned run's corresponding-step per-head selected-count schedule. The token oracle selects the highest-mass far tokens; the leaf oracle selects highest-mass complete C4 leaves globally and bypasses ancestor traversal. Both preserve sink/window and use one softmax. Their key identities, Q/K trajectories and output lengths can differ. This is a mass-selection diagnostic, not a deployable oracle and not a constrained tree oracle. `data/oracle/complete_manifest.json`; `code/oracle_complete_router.py:14-40`, `code/plan_oracle_complete_eval.py:18-60`.

| Task | Context | Cases | Dense | Learned | Token oracle | C4-leaf oracle |
|---|---:|---:|---:|---:|---:|---:|
| VT | 16K | 40 | 1.000 | 0.850 | 1.000 | 1.000 |
| VT | 32K | 40 | 0.990 | 0.780 | 0.995 | 0.995 |
| VT | 64K | 20 | 1.000 | 0.650 | 1.000 | 1.000 |
| CWE | 16K | 40 | 0.885 | 0.768 | 0.880 | 0.860 |
| CWE | 32K | 40 | 0.805 | 0.733 | 0.823 | 0.792 |
| CWE | 64K | 20 | 0.480 | 0.350 | 0.560 | 0.565 |

All 200 cases and 600 oracle branch rows completed; 227,256 corresponding layer/step budget vectors were checked. The complete raw data, including per-head records, are `data/oracle/complete_results.jsonl`; the paired counts, exploratory unadjusted bootstrap intervals and diagnostics are in `data/oracle/stream_summary.md`. On the branches' **different** live trajectories, mean per-case visible-step true mass was 0.76088 learned, 0.88881 token and 0.81885 leaf. A same-Q/K, equal-local-count subset showed mean mass differences +0.11641 for token and +0.05223 for leaf; those many head/step observations are correlated, not independent cases. These cases had been examined already, and neither mass optimality at this budget nor a favorable answer proves learnability or performance at other budgets.

## 7. Data boundaries and the 96-case fixed-beam gate

Earlier training pilots had important flaws: some TinyStories were concatenated without document masks and some WikiText windows could cross article headings. We preserved those results, then made a document-isolated data manifest. It selected 64 PG-19 train books of 16,384 tokens each and 64 WikiText train articles of 4,096 tokens each, with separate validation and test documents. Each teacher forward resets KV and recurrent state. Exact IDs and hashes were disjoint across the relevant source splits; the maximum measured exact 13-word shingle overlap in selected holdout windows was 0.274%. This is not semantic deduplication and cannot rule out pretraining contamination. Sources: [PG-19](https://huggingface.co/datasets/deepmind/pg19), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), `data/training/dataset_manifest.json`, `data/training/split_overlap.json`.

The 96-case gate had eight cases in each of four task variants (noise NIAH, distractor NIAH, VT, CWE) at 16K, 32K and 64K. It compared dense prefill plus dense decode with two fixed-beam sparse-decode routers. The R4 and R16 routers had different training histories. Each head used the same 128 C4 far-leaf target plus sink and aligned 128-token window when available; this matches per-head counts, **not measured physical GPU traffic**, and is not adaptive top-p. Five 64K R16 cases used a separately validated grouped-FP32 implementation; changed operation order prevents claiming universal bitwise equivalence. All 96 three-way comparisons completed, with R16 better/tied/worse than R4 on 8/78/10. Mixed task-level scores and small n do not demonstrate a general improvement. Full per-cell means: `data/boundary/gate_summary.md`; raw primary and five grouped completion records: `data/boundary/gate_results.jsonl`, `data/boundary/grouped_completion.jsonl`. The fixed-beam 96-case results are not the later C4/L6 top-p experiment.

## 8. Controlled C4/L6 R4 versus R16 training

The later controlled training used the same document-isolated 128-document selection for both routers. Nested initialization ensured the R4 references were the first four of the corresponding R16 initialization; both used the same order, 64 sampled query positions per document/layer, objective and optimizer. The audit checked all 256 document passes for identical IDs, input hashes, positions, epoch and learning rates. Code: `code/zha_l6_controlled.py`, `code/train_zha_l6_controlled.py:20-95`, `code/plan_zha_l6_training_summary.py`; settings and source hashes: `data/l6/training_protocol.json`.

Each run processed 1,310,720 unique input tokens twice: 2,621,440 token exposures, 256 optimizer updates per layer and 131,072 sampled layer-query rows. AdamW had beta (0.9, 0.95), epsilon 1e-15, zero weight decay, per-layer clipping 1, peak learning rate 0.032, 20 warmup updates and a final 20% linear decay to 1% of peak. Batch size was one document. Sampled positions were repeated across epochs. The eight validation examples are four held-out books and four held-out articles.

| Router | PG-19 teacher KL, before → after | WikiText teacher KL, before → after |
|---|---:|---:|
| R4 | 3.228 → 1.439 | 2.516 → 1.099 |
| R16 | 2.984 → 0.931 | 2.315 → 0.634 |

The metric is the document- and layer-averaged sum of per-level KL to the dense teacher, not an answer score. R16's lower validation KL means it fit this sampled teacher target better; it does not establish that R16 is better at generation. The logged document-work totals were 583.45 seconds R4 and 621.95 seconds R16, excluding some setup/validation. `data/l6/training_summary.json`, `data/l6/R4_training.json` and `data/l6/R16_training.json` contain the per-document records and sampled positions.

## 9. Correctness and the discovered prefill bug

Direct reference tests cover top-p traversal, multiple top-level roots, partial chunks, causal visibility, GQA and the RoPE lower-bound construction. Across eight frozen-teacher layers, 768 real Q/K head-query selections matched the vectorized selector. Tiny hybrid-Qwen tests cover p=1, sparse prefill, cached decode, future-token causality and padded fallback. `code/test_zha_l6_reference.py`, `code/test_zha_l6_model.py`, `code/test_zha_l6_real_qk.py`, `code/test_zha_l6_sparse.py`; `data/l6/real_qk.log`.

A real issue was found during the first gate: subsequent cached prefill blocks used boolean causal masks, but the hook treated those masks as unsupported, silently falling back to dense. Those sparse gate rows were invalid for a full sparse-prefill claim and are preserved separately rather than included in the corrected results. The mask validator was fixed to accept only the exact boolean causal prefix and to reject padding or future-visible masks. Thereafter the evaluation asserted eight routed calls per prefill block and generated token and zero fallback. Evidence: `data/l6/mask_fix_note.json`, `data/l6/block_prefill_before_fix.json`, `data/l6/block_prefill_diagnostic.json`, `code/test_zha_l6_bool_mask.py`, `code/test_zha_l6_block_prefill.py`.

For a 1,025-token real-Qwen cached-block control, p=1 routed all 24 expected calls, mean sampled retained mass was 0.999999996 and the final greedy token matched dense; maximum and mean absolute final-logit differences were 0.125 and 0.0177. A separate 192-token control found max logit difference 1.6875 over all logits, mean 0.0259 and equal final top token. On identical Q/K/V, gathered-versus-native BF16 attention mean absolute per-element error ranged about 0.00015–0.00050 across layers. The native model's full versus cached final logits also differed. These observations support bounded numerical controls, **not bitwise equality** or exhaustive correctness. `data/l6/pretrained_p1_control.json`, `data/l6/bf16_diagnostic.json`.

## 10. Corrected sparse-prefill and decode gates

The fixed C4/L6 p=0.9 4K/8K gate compared dense, sink-plus-current-only, controlled R4 and controlled R16. Four cases per task/length; all 64 branches completed. Each branch had a fresh hybrid cache. Sink/current alone has a much smaller key budget and is a simple baseline, not an equal-budget comparison.

| Task | Context | Dense | Sink/current | R4 | R16 |
|---|---:|---:|---:|---:|---:|
| VT | 4K | 1.000 | 0.000 | 1.000 | 1.000 |
| VT | 8K | 1.000 | 0.000 | 1.000 | 1.000 |
| CWE | 4K | 1.000 | 0.000 | 0.975 | 0.975 |
| CWE | 8K | 0.975 | 0.000 | 0.925 | 0.900 |

R16 was better/tied/worse than R4 on 0/15/1 of these 16 cases. A separate 16K extension had two examples per task: VT mean dense/R4/R16 was 1.0/1.0/0.9; CWE 1.0/0.95/0.95. These are small partial-credit synthetic cells, not a broad conclusion. The per-branch results, prompts and protocol are in `data/l6/gate_results.jsonl`, `data/l6/gate_summary.json`, `data/l6/long_gate_results.jsonl`, `data/l6/gate_cases`, `data/l6/long_gate_cases`.

## 11. The 32K continuation and latest held-out test

A first one-shot 32K attempt hit the C: free-space guard during Windows paging and saved no continuation. The later attempt, after an active page file was configured on W:, loaded the existing R16 reference queries and optimizer state and ran one **one-shot dense-teacher** 32,768-token document forward with 64 sampled query positions per full-attention layer. It performed one update per layer at learning rate 0.00032, saving a separate checkpoint. The first 16,384 tokens were already in the prior training window; it was not 32,768 new unique tokens or a full paper-scale 32K campaign. The document work took 187.265 seconds excluding model load; peak PyTorch allocated counter was 16.024 GiB under WDDM. All eight pre-update teacher KL values and saved references were finite. Source: `code/run_zha_32k_resume.py`, `data/continuation/result.json`, `data/continuation/report.md`, `checkpoints/updated_r16_checkpoint.pt`.

We subsequently generated six new held-out RULER synthetic cases and compared dense, previous R16 and updated R16 with independent full hybrid caches and sparse prefill **and** decode. The cases did not exactly match 454 earlier saved prompts scanned; shared templates and semantic overlap remain possible. There were two cases per task at 16K and one per task at 32K. The 16K actual lengths were 15,943–16,354 tokens; the 32K prompts had 32,733 and 32,559 tokens. `data/quality/protocol.json`, `data/quality/cases`, `data/quality/results.jsonl`, `code/plan_32k_quality_eval.py:40-89`.

| Task and target | Cases | Dense | Previous R16 | Updated R16 |
|---|---:|---:|---:|---:|
| VT, 16K | 2 | 1.00 | 1.00 | 1.00 |
| CWE, 16K | 2 | 0.90 | 0.95 | 0.95 |
| VT, 32K | 1 | 1.00 | 1.00 | 1.00 |
| CWE, 32K | 1 | 1.00 | 1.00 | 1.00 |

The update was better/tied/worse on 0/6/0 case scores. Text was exactly identical in five old/new pairs and differed in one tied-score 16K CWE pair. Overall unweighted means are 0.9667 dense and 0.9833 for each router, but tiny, mixed cells and ceiling effects make these descriptive only: the apparent advantage over dense is not evidence of general improvement. The full 18 branch runs took approximately 107.6 minutes from log creation to final result. Recorded prefill-plus-decode sums were 2.3 minutes dense, 48.2 minutes old R16, 56.9 minutes updated R16; instrumentation and unoptimized traversal make these **not a controlled speed benchmark**. `data/quality/summary.json`, `data/quality/checkpoint_provenance.json`.

## 12. Interpretation, unknowns and next steps

**Established in these measurements:** the model-level hook operates on the eight full-attention layers for supported inputs; checked causal and p=1 controls pass within stated numerical limits; the trained R16 fits sampled teacher attention better than R4; the latest one-document 32K continuation has no observed score gain on six cases; and an exact dense-mass selector can recover quality in many cases of the earlier oracle test at its tested budget.

**Reasonable inference, not proof:** router selection error is a plausible contributor to the earlier long-context losses, because oracle selection raised true retained mass and often answer score at matched per-head count schedules. Different live trajectories, output lengths, nonindependent head observations and answer correctness prevent attributing every loss to one component. Later lower KL did not guarantee better answer quality.

**Unknown:** generalization to more tasks or documents, whether more or different training would improve the router, whether another budget or threshold would help, true speed after an optimized implementation, multimodal and broad batching correctness, and quality at the model's maximum context. Nothing here establishes a universal hierarchy advantage, a failure of Zyphra's paper, or a reproduction of that paper.

To improve confidence, the next experiment should first freeze a representative untouched suite with many independent examples, include 32K and longer retrieval plus general language/reasoning tasks, compare dense and both router checkpoints with uncertainty intervals, and log true attention retention on matched Q/K when feasible. A further training study should compare controlled longer, more diverse documents with explicitly recorded boundaries and token exposures; avoid training or selecting on the final test. Only after quality is convincing should an incremental GQA-sharing kernel and matched performance/traffic benchmarking be prioritized. The project is currently a research prototype.

## 13. Reproduction and audit

The standalone `verify_results.py` in this package uses only Python's standard library to check file hashes and regenerate the core paired tables from the packaged raw records. Run `python verify_results.py` after extracting the archive. `MANIFEST.json` lists source paths, sizes and hashes; raw oracle data are large but included. Scores are recomputed from recorded rows, not substituted from the prose.

Reproducing GPU runs requires the specified Qwen model revision and its tokenizer, the documented compatible Python/PyTorch/Transformers/Triton environment, sufficient GPU and host resources, a supported local compiler and writeable scratch space. The base model, cached teacher tensors, source corpora and environment are not redistributed in the package. The source scripts retain their original relative paths, W:-based output locations, and disk guards; they are included to make methods inspectable, not a turnkey cross-platform installer. Some tests and training/evaluation scripts overwrite their configured result files. Use isolated output directories and inspect the scripts before rerunning. On the original setup, the CPU control commands were `python test_zha_l6_reference.py`, `python test_zha_l6_model.py`, `python test_zha_l6_controlled.py`, and `python test_r16_multilevel_loss.py` with the workspace dependencies on `PYTHONPATH`; the GPU test `python test_zha_l6_sparse.py` and the real 4B controls require the local model, CUDA/Triton and the same safety checks. `test_zha_l6_real_qk.py` additionally requires the local teacher Q/K files, which are not included.

Primary external sources: [ZHA paper](https://arxiv.org/html/2610.42112v2), [Zyphra overview](https://www.zyphra.com/our-work/zyphra-hierarchical-attention), [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base), [RULER](https://github.com/NVIDIA/RULER), [PG-19](https://huggingface.co/datasets/deepmind/pg19), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), and [OptMem](https://github.com/VictorTaelin/OptMem). The source comparison uses the paper itself where an overview page was inaccessible; no claim depends on having accessed the X post.
