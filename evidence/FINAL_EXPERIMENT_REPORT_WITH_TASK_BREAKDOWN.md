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


# Supplemental task-level and dense comparisons

Supplemental post-hoc comparisons requested after final results; frozen outputs unchanged.

10000 whole-prompt paired bootstrap draws stratified by task, seed 4721; reuse same draws across modes. Task intervals use that task block of each global draw. Percentile bounds with same indexing as frozen script; no multiplicity adjustment or equivalence margin.

Scores and key shares below are percentages; differences and intervals are percentage points. Each task has 24 paired prompts. All-recovered counts require every expected substring, not exact answer-string equality.

## all (n=96)

| Configuration | Score [95% CI] | All recovered | Prefill key share | Decode key share |
|---|---:|---:|---:|---:|
| dense | 95.069 [93.194, 96.875] | 77/96 | 100.000% | 100.000% |
| init_090 | 80.538 [76.250, 84.792] | 39/96 | 10.542% | 2.674% |
| trained_090 | 92.465 [90.069, 94.792] | 69/96 | 7.948% | 2.111% |
| init_095 | 92.448 [89.826, 94.896] | 66/96 | 21.854% | 6.543% |
| trained_095 | 95.000 [93.056, 96.806] | 75/96 | 16.993% | 5.002% |

| Paired comparison | Score difference [95% CI] | Better/tied/worse |
|---|---:|---:|
| trained_090 minus init_090 | +11.927 [+7.309, +16.684] | 35/54/7 |
| trained_095 minus init_095 | +2.552 [-0.382, +5.590] | 17/71/8 |
| init_090 minus dense | -14.531 [-19.149, -10.052] | 6/44/46 |
| trained_090 minus dense | -2.604 [-5.139, -0.139] | 5/74/17 |
| init_095 minus dense | -2.622 [-5.260, -0.069] | 7/70/19 |
| trained_095 minus dense | -0.069 [-2.222, +2.118] | 7/79/10 |

## vt (n=24)

| Configuration | Score [95% CI] | All recovered | Prefill key share | Decode key share |
|---|---:|---:|---:|---:|
| dense | 100.000 [100.000, 100.000] | 24/24 | 100.000% | 100.000% |
| init_090 | 65.833 [54.167, 77.500] | 7/24 | 9.726% | 0.607% |
| trained_090 | 100.000 [100.000, 100.000] | 24/24 | 6.988% | 0.419% |
| init_095 | 95.833 [89.167, 100.000] | 22/24 | 20.645% | 1.742% |
| trained_095 | 100.000 [100.000, 100.000] | 24/24 | 15.487% | 1.052% |

| Paired comparison | Score difference [95% CI] | Better/tied/worse |
|---|---:|---:|
| trained_090 minus init_090 | +34.167 [+22.500, +45.833] | 17/7/0 |
| trained_095 minus init_095 | +4.167 [+0.000, +10.833] | 2/22/0 |
| init_090 minus dense | -34.167 [-45.833, -22.500] | 0/7/17 |
| trained_090 minus dense | +0.000 [+0.000, +0.000] | 0/24/0 |
| init_095 minus dense | -4.167 [-10.833, +0.000] | 0/22/2 |
| trained_095 minus dense | +0.000 [+0.000, +0.000] | 0/24/0 |

## cwe (n=24)

| Configuration | Score [95% CI] | All recovered | Prefill key share | Decode key share |
|---|---:|---:|---:|---:|
| dense | 94.167 [90.417, 97.500] | 15/24 | 100.000% | 100.000% |
| init_090 | 85.833 [80.833, 90.000] | 3/24 | 11.648% | 4.100% |
| trained_090 | 87.917 [82.083, 92.917] | 9/24 | 8.974% | 3.166% |
| init_095 | 91.667 [88.750, 94.583] | 8/24 | 23.758% | 9.924% |
| trained_095 | 92.500 [88.750, 95.833] | 12/24 | 18.810% | 7.388% |

| Paired comparison | Score difference [95% CI] | Better/tied/worse |
|---|---:|---:|
| trained_090 minus init_090 | +2.083 [-1.667, +5.833] | 7/14/3 |
| trained_095 minus init_095 | +0.833 [-2.500, +4.167] | 8/12/4 |
| init_090 minus dense | -8.333 [-12.917, -4.167] | 2/6/16 |
| trained_090 minus dense | -6.250 [-10.833, -1.667] | 2/10/12 |
| init_095 minus dense | -2.500 [-5.833, +0.833] | 4/9/11 |
| trained_095 minus dense | -1.667 [-4.583, +1.250] | 2/16/6 |

## niah_distractor (n=24)

| Configuration | Score [95% CI] | All recovered | Prefill key share | Decode key share |
|---|---:|---:|---:|---:|
| dense | 86.111 [79.167, 93.056] | 14/24 | 100.000% | 100.000% |
| init_090 | 77.778 [69.444, 86.111] | 10/24 | 11.105% | 4.004% |
| trained_090 | 81.944 [75.000, 88.889] | 12/24 | 8.647% | 3.060% |
| init_095 | 83.333 [75.000, 90.278] | 13/24 | 22.396% | 10.060% |
| trained_095 | 87.500 [80.556, 93.056] | 15/24 | 17.887% | 8.113% |

| Paired comparison | Score difference [95% CI] | Better/tied/worse |
|---|---:|---:|
| trained_090 minus init_090 | +4.167 [-6.944, +16.667] | 6/14/4 |
| trained_095 minus init_095 | +4.167 [-5.556, +13.889] | 6/14/4 |
| init_090 minus dense | -8.333 [-19.444, +2.778] | 4/12/8 |
| trained_090 minus dense | -4.167 [-13.889, +4.167] | 3/16/5 |
| init_095 minus dense | -2.778 [-9.722, +5.556] | 3/16/5 |
| trained_095 minus dense | +1.389 [-6.944, +9.722] | 5/15/4 |

## niah_multikey (n=24)

| Configuration | Score [95% CI] | All recovered | Prefill key share | Decode key share |
|---|---:|---:|---:|---:|
| dense | 100.000 [100.000, 100.000] | 24/24 | 100.000% | 100.000% |
| init_090 | 92.708 [85.417, 97.917] | 19/24 | 9.713% | 0.480% |
| trained_090 | 100.000 [100.000, 100.000] | 24/24 | 7.207% | 0.445% |
| init_095 | 98.958 [96.875, 100.000] | 23/24 | 20.653% | 1.256% |
| trained_095 | 100.000 [100.000, 100.000] | 24/24 | 15.829% | 0.966% |

| Paired comparison | Score difference [95% CI] | Better/tied/worse |
|---|---:|---:|
| trained_090 minus init_090 | +7.292 [+2.083, +14.583] | 5/19/0 |
| trained_095 minus init_095 | +1.042 [+0.000, +3.125] | 1/23/0 |
| init_090 minus dense | -7.292 [-14.583, -2.083] | 0/19/5 |
| trained_090 minus dense | +0.000 [+0.000, +0.000] | 0/24/0 |
| init_095 minus dense | -1.042 [-3.125, +0.000] | 0/23/1 |
| trained_095 minus dense | +0.000 [+0.000, +0.000] | 0/24/0 |

## Key-share aggregation

For each branch and phase: sum the selected causal KV positions over every attention head, query, and routed full-attention layer; divide by the sum of all causally available positions over the same evaluations. The model has eight full-attention layers and 16 query heads each (128 layer-head evaluations per query position). The counter counts gathered key indices, not GPU bytes or the dense probability mass on those indices.

Let N be the prompt token count and T the generated-token count. The first N−1 prompt positions are processed in prefill. The last prompt token is fed during the first one-token generation step and is counted under decode. Prefill availability = 128 × (N−1) × N / 2. Decode availability = 128 × [T × N + T × (T−1)/2]. All 480 records matched these formulas. This boundary convention is identical in all configurations.

For an aggregate over prompts, key share = sum(branch selected keys) / sum(branch available keys), not the arithmetic mean of branch shares, not divided by final context length. Per-task values use only those 24 prompts; global values pool all 96. Thus prompt lengths and generation lengths affect weights. Quality scores instead give each prompt equal weight (and tasks equal weight because there are 24 each). Every bootstrap draw resamples complete prompts, pairs all configurations, and recomputes the ratio of sums.

Each configuration follows its own autoregressive trajectory; these are not identical-Q/K replay measurements or exact matched-key-budget runs. Key share omits tree-building/router work and repeated physical GQA loads.

## Audit and limitations

Independent audit: 480 unique branches, 96 complete five-way pairs; original frozen summary reproduced from raw data; 13 frozen source/snapshot hashes, protocol, manifest and both checkpoint hashes unchanged; scores recomputed from answers; exact availability and routed-call counts; zero fallbacks; 279,688 routed layer calls. Original initial tensors were checked against seed 94121 in all eight layers before inference. The final integrity audit rechecked checkpoint file hashes, not a new training run.

The supplemental bootstrap reproduces the two frozen overall trained-versus-initialized score intervals to numerical precision. All extra task and dense-comparison intervals are exploratory post-hoc outputs. There is no multiplicity correction or predeclared equivalence margin. An interval crossing zero is not evidence of equivalence. Perfect sampled task scores produce degenerate bootstrap intervals; those do not imply certainty on unseen cases.

Single router initialization/checkpoint, synthetic template-based prompts, substring scoring, fixed output caps, one context range, and only 24 cases per task limit generalization. The task effects differ substantially; broad claims should not be based solely on the pooled score. Sparse inference remained slower than dense. Key reduction is not a speedup or cache-memory reduction.

Original results, summary, report, evidence ZIP and figures remain unchanged. This supplement is saved separately.
