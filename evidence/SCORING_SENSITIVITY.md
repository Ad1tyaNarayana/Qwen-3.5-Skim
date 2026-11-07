# Post-hoc scoring sensitivity check

This supplement was added after the frozen experiment. The original substring scores, reports, raw results, protocol, source lock and summaries remain unchanged.

The alternative scorer lowercases the output and each expected item, escapes the item literally, and requires no adjacent Unicode word character: `(?<!\w)ITEM(?!\w)`. This rejects rail within railway and fen within fence. It is an item-recall sensitivity check, not exact-match answer grading; extra answers and matches elsewhere in the output remain unpenalized.

| Configuration | Frozen substring score | Boundary score | Frozen all recovered | Boundary all recovered |
|---|---:|---:|---:|---:|
| dense | 95.069% | 95.069% | 77/96 | 77/96 |
| init_090 | 80.538% | 80.434% | 39/96 | 39/96 |
| trained_090 | 92.465% | 92.361% | 69/96 | 68/96 |
| init_095 | 92.448% | 92.344% | 66/96 | 65/96 |
| trained_095 | 95.000% | 95.000% | 75/96 | 75/96 |

## Paired training effect

| Threshold | Scorer | Trained minus initial (percentage points) | Paired 95% interval | Better/tied/worse |
|---|---|---:|---|---|
| p=.90 | primary | +11.927 | [+7.309, +16.684] | 35/54/7 |
| p=.90 | boundary | +11.927 | [+7.309, +16.701] | 34/55/7 |
| p=.95 | primary | +2.552 | [-0.382, +5.590] | 17/71/8 |
| p=.95 | boundary | +2.656 | [-0.278, +5.729] | 18/70/8 |

Intervals use 10,000 whole-prompt paired resamples stratified by task, seed 4721, with the same percentile convention as the frozen analysis. This post-hoc analysis has no multiplicity adjustment and does not establish equivalence.

## Changed item credits

| Task | Case index (zero-based) | Configuration | Expected item |
|---|---:|---|---|
| cwe | 18 | trained_090 | rail |
| cwe | 18 | init_095 | rail |
| cwe | 19 | init_090 | fen |

The p=.90 aggregate training benefit remains +11.927 percentage points with a positive paired interval. Trained p=.90 loses one all-items-recovered case (69 to 68). This sensitivity check does not overturn the main finding. Per-task scores and full affected outputs are in `scoring_sensitivity.json`.

## Reproduce

From the extracted package, using Python 3.10 or newer and no additional packages:

```text
python scoring_sensitivity.py
```

The default command recomputes and checks both supplemental files without writing. Use `python scoring_sensitivity.py --write` only to regenerate these two supplements. Neither command changes the frozen experiment files.

Raw-result SHA256: `a9379597f0ef531a22e180d7d0468b547af173dcf131b127d7d34f3c81e631ba`.
