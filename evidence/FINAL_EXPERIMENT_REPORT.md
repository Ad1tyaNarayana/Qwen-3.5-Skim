# Frozen Qwen hierarchical-router experiment — final results

Completed 2026-11-05. All 480 branches (96 prompts × 5 configurations) and all 20 separate development branches completed. The original frozen analysis and figures are preserved. This report and the readable two-panel figure are post-run presentation artifacts, not changes to the protocol.

## Finding

At p=0.90, the controlled trained R16 router improved answer quality over its exact original initialization while using fewer selected keys in both prefill and decode. This supports a useful training benefit on this fixed 16K synthetic test set. At p=0.95, training also reduced selected keys, but its answer-score advantage has a paired interval crossing zero. Near-equality with dense in aggregate is not proof of equivalence or general superiority.

## Aggregate results

| Configuration | Partial-credit score | All expected recovered | Prefill key share | Decode key share |
|---|---:|---:|---:|---:|
| dense | 95.069% | 77/96 | 100.000% | 100.000% |
| init_090 | 80.538% | 39/96 | 10.542% | 2.674% |
| trained_090 | 92.465% | 69/96 | 7.948% | 2.111% |
| init_095 | 92.448% | 66/96 | 21.854% | 6.543% |
| trained_095 | 95.000% | 75/96 | 16.993% | 5.002% |

Shares are total selected keys divided by total causally available keys over head-query evaluations, separately by phase. They are not physical memory traffic, attention probability retained, cache compression, or measured speedup. Different generated trajectories and lengths contribute to these totals.

## Paired training effect

- p=.90: trained minus untrained score +11.927 percentage points; paired 95% interval [+7.309, +16.684]. Better/tied/worse: 35/54/7. Aggregate decode key share falls 21.1% relative to untrained.
- p=.95: trained minus untrained score +2.552 percentage points; paired 95% interval [-0.382, +5.590]. Better/tied/worse: 17/71/8. Aggregate decode key share falls 23.5% relative to untrained.

Intervals use the frozen 10,000 whole-prompt paired resamples stratified by task, seed 4721. They reflect this sampled prompt population, not model-seed variability. No multiple-comparison adjustment was predeclared. No noninferiority/equivalence margin was predeclared.

## Task scores (24 prompts each)

| Task | Dense | Init .90 | Trained .90 | Init .95 | Trained .95 |
|---|---:|---:|---:|---:|---:|
| vt | 100.00% | 65.83% | 100.00% | 95.83% | 100.00% |
| cwe | 94.17% | 85.83% | 87.92% | 91.67% | 92.50% |
| niah_distractor | 86.11% | 77.78% | 81.94% | 83.33% | 87.50% |
| niah_multikey | 100.00% | 92.71% | 100.00% | 98.96% | 100.00% |

The largest p=.90 gain is variable tracking; the task gains are not uniform. All four task types have equal numbers of prompts.

## Method and controls

Qwen/Qwen3.5-4B-Base revision 1001bb4d826a52d1f399e183466143f4da7b741b; BF16, batch one, RTX 5070 Ti 16 GB, Windows. Corrected C4/L6 grouped router in all eight full-attention layers, with recurrent layers retained. Sparse prefill and decode; 512-token prefill blocks, sinks and current chunk, adaptive p=.90/.95. Fresh KV and recurrent state per branch; greedy generation, frozen task caps; measure_mass disabled.

The trained router is W:/Codex-ZHA-L6-20261104/R16/router.pt, before the one-document 32K continuation. Its saved original initialization was verified bitwise against seed 94121 and the saved optimizer checkpoint across all eight full-attention layers. No further training was performed for this experiment.

Pinned RULER generator commit db43f92ff634bbce1c33d1fb15858aec4bba509a. Four development prompts and 96 final prompts were generated and hashed before evaluation. Final tasks: variable tracking, common-word extraction, needle retrieval with distractors, and multi-key needle retrieval, 24 each. No exact prompt overlaps with the 460 earlier checked prompts; semantic overlap was not excluded. Actual input lengths: 15962–16354 tokens. Scoring is the frozen case-insensitive expected-substring fraction, plus whether every expected item appeared. It does not penalize extra answers or independently grade reasoning.

Post-run integrity audit passed: 480 unique branches, 96 complete five-way prompt groups, all 13 frozen source files and snapshots unchanged, protocol/manifest/checkpoint hashes unchanged, scores recomputed from answers, exact causal-availability counts, selected-key bounds and ratios, 279688 routed layer calls verified, zero dense routing fallbacks. A routing fallback is distinct from the unchanged reference implementations used for the recurrent layers.

## Runtime and resources

Total measured prefill plus decode time: 15.542 hours across 480 branches. The supervisor completed at 2026-11-05 11:16:44 UTC. The sweep began around 2026-11-04 19:44 UTC. These measurements are operational timings, not a controlled repeated speed benchmark.

| Configuration | Mean prefill s | Mean decode s | Mean generated tokens | Peak allocated GiB |
|---|---:|---:|---:|---:|
| dense | 3.71 | 2.09 | 60.14 | 9.343 |
| init_090 | 128.12 | 14.01 | 58.61 | 9.326 |
| trained_090 | 127.34 | 14.84 | 62.10 | 9.326 |
| init_095 | 133.00 | 13.82 | 57.39 | 9.328 |
| trained_095 | 131.96 | 13.95 | 58.07 | 9.327 |

The sparse reference implementation is much slower than the dense baseline despite selecting fewer keys. Generated lengths differ, so total decode seconds are not a matched-token throughput comparison. Peak PyTorch allocation is not total GPU memory use. Dense uses SDPA; recurrent operations use the same installed reference implementations for every configuration. No production speedup is established.

The development workers exited unexpectedly after 18 saved branches; those rows were preserved and the remaining two resumed under an independent supervisor. All 480 final branches completed without a logged evaluation error. Outputs and TEMP/TMP/Triton cache were on W:. C: >=26 GiB and W: >=10 GiB guards stayed enabled. A ~2.24 GiB C: decline occurred during the run, but no guard failed; its cause was not established by the read-only investigation. No files were deleted and no system settings changed.

## Limits and interpretation

This is one initialization/checkpoint and four synthetic task families at approximately 16K, not broad language quality, length extrapolation, a paper reproduction, or evidence of gains beyond dense Qwen. Prompts share templates. The thresholds were fixed rather than matched for exact key budgets. Nevertheless, at p=.90 training achieved both better aggregate answers and lower aggregate key shares; its benefit is not explained by merely selecting more keys in these measurements. The p=.95 trained score (95.000%) is close to dense (95.069%), but all-items recovery is 75/96 versus 77/96, and equivalence was not established.

## Evidence and reproduction

All paths below are relative to W:/Codex-Qwen-Final-20261104/.

- `final_results.jsonl`: all 480 answers, expectations, per-branch counts, times and hashes.
- `final_summary.json`: frozen paired bootstrap results and per-task metrics.
- `final_integrity_audit.json`: independent recomputation, counters, hashes and timing summaries.
- `protocol.json`, `manifest.json`, `source_lock.json`, `cases/`: frozen definitions, seeds, prompts and provenance.
- `source/`: frozen evaluation, router, counter and analysis code.
- `dev_results.jsonl`, `dev_summary.json`, `development_feasibility.json`, `counter_test.log`: development and correctness evidence.
- `final_eval.log`, `supervisor_resume.log`, `supervisor_status.json`: execution evidence.
- `quality_vs_decode_key_share.png/.pdf`: original frozen figure (labels crowded).
- `quality_vs_decode_key_share_readable.png/.pdf`: same values and intervals with a magnified panel.

To reproduce the frozen summary/figure with the existing environment, from the original workspace:

```powershell
$env:FINAL_W_ROOT='W:\Codex-Qwen-Final-20261104'
$env:PYTHONPATH='W:\Codex-Qwen-Final-20261104\plot_deps;C:\Users\Admin\Documents\Codex\2026-11-04\task\.deps'
& 'C:\Users\Admin\qwenputer-games-work\.venv\Scripts\python.exe' final_experiment_analyze.py final
```

The archive contains frozen source, prompts, outputs, logs, analysis, plots, audit code and the controlled router/optimizer checkpoints. It excludes base-model weights, Python environment, installed plotting dependencies and temporary compiler caches. Checkpoint paths in the scripts reflect the original local layout. Re-running inference requires that model/environment/layout or an explicitly documented port; re-running it is not needed to inspect these results.

Raw-result SHA256: `a9379597f0ef531a22e180d7d0468b547af173dcf131b127d7d34f3c81e631ba`. Frozen-summary SHA256: `a6115611aac07964babc0eb70ae7155f70438678266a874e01d3758a5b63b36c`.
