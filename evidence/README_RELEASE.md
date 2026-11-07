# Portable evidence release

This release adds two post-run fixes: a separately labeled word-boundary scoring sensitivity analysis, and a portable read-only verifier. All frozen raw results, primary summaries, protocol, source lock, source snapshots, checkpoints and original reports are preserved.

## Verify after extracting

Use Python 3.10 or newer. No PyTorch, CUDA, model download or third-party package is required.

From the extracted folder:

```text
python final_experiment_verify.py
```

On Windows with the Python launcher, `py -3 final_experiment_verify.py` also works. The script resolves all paths relative to itself, so it can also be invoked by absolute path from another working directory. It reads bundled source files under `source/` and checkpoints under `checkpoints/R16/`. It prints a verification result and writes no files.

The verifier checks frozen file hashes, saved prompt text and expected answers, 480 unique branches grouped into 96 complete comparisons, scores, causal key-count arithmetic, recorded routing coverage, aggregate and task summaries, the two primary trained-versus-initialized bootstrap score intervals, and the supplemental scoring analysis. Checkpoints are hashed without loading their tensors. This verifies the evidence package, not a new inference run or an independent authentication of its provenance.

## Read the results

- `FINAL_EXPERIMENT_REPORT_WITH_TASK_BREAKDOWN.md`: unchanged original primary report and task breakdown.
- `SCORING_SENSITIVITY.md`: post-hoc boundary-aware scoring results and limitations.
- `scoring_sensitivity.json`: full machine-readable sensitivity results, per-task scores and affected generated answers.

The frozen substring scorer grants three item credits that fail the boundary-aware check: rail within railway in two branches, and fen within fence in one branch. The trained p=.90 boundary-aware score is 92.361% rather than 92.465%, with 68 rather than 69 prompts recovering all expected items. The aggregate p=.90 training effect remains +11.927 percentage points with a positive paired 95% interval. These supplemental results do not replace the primary metric after the fact.

Recompute and verify the sensitivity files without writing:

```text
python scoring_sensitivity.py
```

To intentionally regenerate only those two supplements, add `--write`. This does not change primary data or summaries.

## Scope of portability

Evidence verification and the new sensitivity analysis are portable. The historical inference, preparation and figure-generation scripts retain their original environment and path assumptions as frozen source evidence; this release does not claim portable GPU inference. Use this README's verification command in preference to historical reproduction snippets for the original machine.

The previous verifier is retained under `historical/final_experiment_verify_original.py` for provenance. It is not the portable entry point. The older archive files remain unchanged on the original machine.
