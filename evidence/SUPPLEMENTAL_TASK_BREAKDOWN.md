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
