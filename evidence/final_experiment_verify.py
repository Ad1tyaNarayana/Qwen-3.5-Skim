"""Portable, read-only evidence verification using only the standard library.

Run with Python 3.10+ from any working directory. Paths are relative to this
file. Checkpoints are hashed, never deserialized. No GPU/model install needed.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    lock = read('source_lock.json')
    for name, expected in lock['source_sha256'].items():
        require(sha(ROOT / 'source' / name) == expected, f'Frozen source mismatch: {name}')
    for name in ('protocol', 'manifest'):
        require(sha(ROOT / (name + '.json')) == lock[name + '_sha256'], f'{name} hash mismatch')
    for name, key in [('router.pt', 'trained_router_sha256'), ('checkpoint.pt', 'training_checkpoint_sha256')]:
        require(sha(ROOT / 'checkpoints' / 'R16' / name) == lock[key], f'Checkpoint mismatch: {name}')
    historical_audit = read('final_integrity_audit.json')
    for filename, key in [('final_results.jsonl', 'results_sha256'), ('final_summary.json', 'summary_sha256')]:
        require(sha(ROOT / filename) == historical_audit[key], f'Frozen {filename} hash mismatch')
    protocol = read('protocol.json')
    modes = set(protocol['configurations'])
    manifest = read('manifest.json')['cases']
    case_files = {}
    for item in manifest:
        name = f"cases/{item['split']}_{item['task']}/validation.jsonl"
        require(sha(ROOT / name) == item['case_file_sha256'], f'Case file hash mismatch: {name}')
        if name not in case_files:
            case_files[name] = [json.loads(line) for line in (ROOT / name).read_text(encoding='utf-8').splitlines()]
        case = case_files[name][item['index']]
        require(hashlib.sha256(case['input'].encode()).hexdigest() == item['input_sha256'], 'Prompt hash mismatch')
        require(case['outputs'] == item['expected'], 'Prompt expected answers mismatch')
    cases = {(x['task'], x['index']): x for x in manifest if x['split'] == 'final'}
    require(len(cases) == 96, 'Expected 96 final prompts')
    rows = [json.loads(line) for line in (ROOT / 'final_results.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows) == 480, 'Expected 480 final branches')
    paired = collections.defaultdict(dict)
    by_mode = collections.defaultdict(list)
    for row in rows:
        key = row['task'], row['index']
        mode = row['mode']
        require(mode in modes and mode not in paired[key], f'Unexpected/duplicate branch: {key}, {mode}')
        paired[key][mode] = row
        case = cases[key]
        n, t = row['input_tokens'], row['generated_tokens']
        require(row['split'] == 'final', 'Unexpected split')
        for field in ('input_sha256', 'token_sha256', 'expected', 'input_tokens'):
            require(row[field] == case[field], f'Provenance mismatch: {key}, {field}')
        require(0 < t <= case['generation_cap'] and row['fallback_count'] == 0, 'Generation/fallback mismatch')
        found = [answer.lower() in row['generated'].lower() for answer in row['expected']]
        require(row['score'] == sum(found) / len(found), f'Score mismatch: {key}, {mode}')
        require(row['all_expected_recovered'] == all(found), 'All-recovered score mismatch')
        require(row['prefill_available'] == 128 * (n - 1) * n // 2, 'Prefill denominator mismatch')
        require(row['decode_available'] == 128 * (t * n + t * (t - 1) // 2), 'Decode denominator mismatch')
        require(row['routed_calls'] == (0 if mode == 'dense' else 8 * (math.ceil((n - 1) / 512) + t)), 'Routing coverage mismatch')
        for phase in ('prefill', 'decode'):
            selected, available = row[phase + '_selected'], row[phase + '_available']
            require(0 < selected <= available, 'Selected key count out of bounds')
            require(row[phase + '_key_share'] == selected / available, 'Key share mismatch')
            if mode == 'dense':
                require(selected == available, 'Dense branch did not count all keys')
        if mode != 'dense':
            require(row['checkpoint_sha256'] == lock['trained_router_sha256'], 'Branch checkpoint hash mismatch')
            tensor = 'initial_tensor_sha256' if mode.startswith('init') else 'trained_reference_tensor_sha256'
            require(row['reference_tensor_sha256'] == lock[tensor], 'Branch reference hash mismatch')
            require(row['threshold'] == (0.9 if mode.endswith('090') else 0.95), 'Threshold mismatch')
        by_mode[mode].append(row)
    require(len(paired) == 96 and all(set(v) == modes for v in paired.values()), 'Incomplete paired prompts')
    summary = read('final_summary.json')

    def check_summary(subset, recorded):
        require(math.isclose(statistics.mean(x['score'] for x in subset), recorded['score'], abs_tol=1e-12), 'Summary score mismatch')
        require(math.isclose(statistics.mean(x['all_expected_recovered'] for x in subset), recorded['all_recovered'], abs_tol=1e-12), 'Summary all-recovered mismatch')
        for phase in ('prefill', 'decode'):
            value = sum(x[phase + '_selected'] for x in subset) / sum(x[phase + '_available'] for x in subset)
            require(math.isclose(value, recorded[phase + '_key_share'], abs_tol=1e-12), 'Summary key-share mismatch')

    for mode in modes:
        require(len(by_mode[mode]) == 96, 'Wrong per-mode branch count')
        check_summary(by_mode[mode], summary['configurations'][mode])
    for task, result in summary['tasks'].items():
        for mode in modes:
            subset = [x for x in by_mode[mode] if x['task'] == task]
            require(len(subset) == 24, 'Wrong per-task branch count')
            check_summary(subset, result['modes'][mode])
    # run_path avoids bytecode caches, keeping this command read-only.
    sensitivity = runpy.run_path(str(ROOT / 'scoring_sensitivity.py'))
    result = sensitivity['compute'](ROOT)
    sensitivity['check_saved'](ROOT, result)
    for threshold in ('090', '095'):
        actual, expected = result['paired'][threshold]['primary'], summary['paired'][threshold]
        require(math.isclose(actual['mean_difference'], expected['mean_score_difference'], abs_tol=1e-12), 'Paired effect mismatch')
        require(actual['better_tied_worse'] == expected['better_tied_worse'], 'Paired counts mismatch')
        require(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(actual['ci95'], expected['intervals']['score'])), 'Paired bootstrap interval mismatch')
    print(json.dumps({
        'status': 'passed', 'rows': len(rows), 'paired_prompts': len(paired),
        'source_files_verified': len(lock['source_sha256']), 'prompt_files_verified': len(case_files),
        'checkpoint_files_verified': 2, 'routed_layer_calls': sum(x['routed_calls'] for x in rows),
        'fallbacks': sum(x['fallback_count'] for x in rows),
        'primary_paired_intervals_recomputed': True, 'scoring_sensitivity_verified': True,
        'files_written': 0,
        'limits': 'Verifies packaged evidence, hashes and recorded counter arithmetic; does not rerun inference, retokenize prompts, deserialize checkpoint tensors or independently authenticate the original run.'
    }, indent=2))


if __name__ == '__main__':
    main()
