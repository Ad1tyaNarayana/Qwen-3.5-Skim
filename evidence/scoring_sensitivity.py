"""Post-hoc word-boundary sensitivity analysis; Python standard library only.

Default: verify saved supplemental outputs without writing. --write regenerates
only the two explicitly supplemental files, never the frozen experiment files.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

TASKS = ('vt', 'cwe', 'niah_distractor', 'niah_multikey')
MODES = ('dense', 'init_090', 'trained_090', 'init_095', 'trained_095')
JSON_NAME = 'scoring_sensitivity.json'
REPORT_NAME = 'SCORING_SENSITIVITY.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compute(root):
    raw = (root / 'final_results.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    groups = collections.defaultdict(dict)
    collisions = []
    for row in rows:
        key = (row['task'], row['index'])
        require(row['mode'] not in groups[key], f'Duplicate branch: {key}, {row["mode"]}')
        expected, output = row['expected'], row['generated']
        primary = [answer.lower() in output.lower() for answer in expected]
        # Literal expected item, case-insensitive, with no adjacent Unicode
        # word character. Punctuation inside an expected item remains literal.
        bounded = [bool(re.search(r'(?<!\w)' + re.escape(answer.lower()) + r'(?!\w)',
                                  output.lower())) for answer in expected]
        require(sum(primary) / len(expected) == row['score'], 'Frozen score mismatch')
        require(all(primary) == row['all_expected_recovered'], 'Frozen all-recovered mismatch')
        item = {'primary': row['score'], 'boundary': sum(bounded) / len(expected),
                'primary_all': all(primary), 'boundary_all': all(bounded)}
        groups[key][row['mode']] = item
        for answer, old, new in zip(expected, primary, bounded):
            if old != new:
                collisions.append({'task': key[0], 'index': key[1], 'mode': row['mode'],
                                   'expected_item': answer, 'substring_match': old,
                                   'boundary_match': new, 'generated': output})
    require(len(rows) == 480 and len(groups) == 96, 'Expected 480 branches and 96 prompts')
    require(all(set(g) == set(MODES) for g in groups.values()), 'Incomplete configurations')
    keys = sorted(groups)
    by_task = {task: [key for key in keys if key[0] == task] for task in TASKS}
    require(all(len(v) == 24 for v in by_task.values()), 'Expected 24 prompts per task')

    def summarize(selected):
        return {mode: {
            'primary_score': statistics.mean(groups[k][mode]['primary'] for k in selected),
            'boundary_score': statistics.mean(groups[k][mode]['boundary'] for k in selected),
            'primary_all_recovered': sum(groups[k][mode]['primary_all'] for k in selected),
            'boundary_all_recovered': sum(groups[k][mode]['boundary_all'] for k in selected),
        } for mode in MODES}

    paired = {}
    rng = random.Random(4721)
    draws = {(threshold, metric): [] for threshold in ('090', '095')
             for metric in ('primary', 'boundary')}
    for _ in range(10000):
        selected = [rng.choice(by_task[task]) for task in TASKS for _ in range(24)]
        for (threshold, metric), values in draws.items():
            values.append(statistics.mean(
                groups[k]['trained_' + threshold][metric] - groups[k]['init_' + threshold][metric]
                for k in selected))
    for threshold in ('090', '095'):
        paired[threshold] = {}
        for metric in ('primary', 'boundary'):
            delta = [groups[k]['trained_' + threshold][metric] -
                     groups[k]['init_' + threshold][metric] for k in keys]
            values = sorted(draws[threshold, metric])
            paired[threshold][metric] = {
                'mean_difference': statistics.mean(delta),
                'ci95': [values[249], values[9749]],
                'better_tied_worse': [sum(d > 0 for d in delta), sum(d == 0 for d in delta),
                                      sum(d < 0 for d in delta)],
            }
    return {
        'status': 'post-hoc sensitivity analysis; frozen primary scoring unchanged',
        'raw_results_sha256': hashlib.sha256(raw).hexdigest(),
        'method': 'Lowercase literal expected items bounded by (?<!\\w) and (?!\\w); Unicode word characters.',
        'bootstrap': '10000 whole-prompt paired draws stratified by task; seed 4721; percentile indices 249/9749; no multiplicity adjustment.',
        'limitations': 'Checks item boundaries only; still gives recall credit anywhere in output, does not penalize extra answers, and is not exact answer matching.',
        'n': len(keys), 'configurations': summarize(keys),
        'tasks': {task: summarize(by_task[task]) for task in TASKS},
        'paired': paired, 'changed_item_matches': collisions,
    }


def render(result):
    lines = ['# Post-hoc scoring sensitivity check', '',
             'This supplement was added after the frozen experiment. The original substring scores, reports, raw results, protocol, source lock and summaries remain unchanged.', '',
             'The alternative scorer lowercases the output and each expected item, escapes the item literally, and requires no adjacent Unicode word character: `(?<!\\w)ITEM(?!\\w)`. This rejects rail within railway and fen within fence. It is an item-recall sensitivity check, not exact-match answer grading; extra answers and matches elsewhere in the output remain unpenalized.', '',
             '| Configuration | Frozen substring score | Boundary score | Frozen all recovered | Boundary all recovered |',
             '|---|---:|---:|---:|---:|']
    for mode, d in result['configurations'].items():
        lines.append(f'| {mode} | {100*d["primary_score"]:.3f}% | {100*d["boundary_score"]:.3f}% | {d["primary_all_recovered"]}/96 | {d["boundary_all_recovered"]}/96 |')
    lines += ['', '## Paired training effect', '',
              '| Threshold | Scorer | Trained minus initial (percentage points) | Paired 95% interval | Better/tied/worse |',
              '|---|---|---:|---|---|']
    for threshold, metrics in result['paired'].items():
        for metric, d in metrics.items():
            lo, hi = d['ci95']
            counts = '/'.join(map(str, d['better_tied_worse']))
            lines.append(f'| p=.{threshold[1:]} | {metric} | {100*d["mean_difference"]:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {counts} |')
    lines += ['', 'Intervals use 10,000 whole-prompt paired resamples stratified by task, seed 4721, with the same percentile convention as the frozen analysis. This post-hoc analysis has no multiplicity adjustment and does not establish equivalence.', '',
              '## Changed item credits', '',
              '| Task | Case index (zero-based) | Configuration | Expected item |',
              '|---|---:|---|---|']
    for c in result['changed_item_matches']:
        lines.append(f'| {c["task"]} | {c["index"]} | {c["mode"]} | {c["expected_item"]} |')
    lines += ['', 'The p=.90 aggregate training benefit remains +11.927 percentage points with a positive paired interval. Trained p=.90 loses one all-items-recovered case (69 to 68). This sensitivity check does not overturn the main finding. Per-task scores and full affected outputs are in `scoring_sensitivity.json`.', '',
              '## Reproduce', '', 'From the extracted package, using Python 3.10 or newer and no additional packages:', '',
              '```text', 'python scoring_sensitivity.py', '```', '',
              'The default command recomputes and checks both supplemental files without writing. Use `python scoring_sensitivity.py --write` only to regenerate these two supplements. Neither command changes the frozen experiment files.', '',
              f'Raw-result SHA256: `{result["raw_results_sha256"]}`.', '']
    return '\n'.join(lines)


def check_saved(root, result=None):
    result = compute(root) if result is None else result
    require(json.loads((root / JSON_NAME).read_text(encoding='utf-8')) == result,
            'Saved sensitivity JSON differs from recomputation')
    require((root / REPORT_NAME).read_text(encoding='utf-8') == render(result),
            'Saved sensitivity report differs from recomputation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Regenerate supplemental JSON and Markdown only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result = compute(root)
    if args.write:
        (root / JSON_NAME).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        (root / REPORT_NAME).write_text(render(result), encoding='utf-8')
    else:
        check_saved(root, result)
    print('Sensitivity analysis PASS: 480 branches; ' + str(len(result['changed_item_matches'])) + ' changed item matches; frozen results untouched.')
