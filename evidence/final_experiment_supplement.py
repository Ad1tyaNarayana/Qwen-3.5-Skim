"""Requested post-hoc task/dense comparisons; preserves all frozen outputs."""
import json,random,math,statistics,zipfile,hashlib,shutil
from pathlib import Path
import numpy as np
R=Path('W:/Codex-Qwen-Final-20261104')
rows=[json.loads(s) for s in (R/'final_results.jsonl').read_text().splitlines()]
summary=json.loads((R/'final_summary.json').read_text());audit=json.loads((R/'final_integrity_audit.json').read_text())
modes=list(summary['configurations']);tasks=['vt','cwe','niah_distractor','niah_multikey']
groups={}
for r in rows:groups.setdefault((r['task'],r['index']),{})[r['mode']]=r
keys=sorted(groups);by_task={t:[k for k in keys if k[0]==t] for t in tasks};rng=random.Random(4721)
key_index={k:i for i,k in enumerate(keys)}
draws=np.array([[key_index[rng.choice(by_task[t])] for t in tasks for _ in range(24)] for b in range(10000)])
pairs=[('trained_090','init_090'),('trained_095','init_095')]+[(m,'dense') for m in modes if m!='dense']
def interval(x):
    x=np.sort(x);return [float(x[int(.025*(len(x)-1))]),float(x[int(.975*(len(x)-1))])]
def metrics(mode,idx):
    z=[groups[k][mode] for k in keys]
    field=lambda name:np.array([r[name] for r in z])[idx]
    return {'score':field('score').mean(axis=-1),'all_recovered':field('all_expected_recovered').mean(axis=-1),
      'prefill_key_share':field('prefill_selected').sum(axis=-1)/field('prefill_available').sum(axis=-1),
      'decode_key_share':field('decode_selected').sum(axis=-1)/field('decode_available').sum(axis=-1)}
result={'label':'Supplemental post-hoc comparisons requested after final results; frozen outputs unchanged',
 'method':'10000 whole-prompt paired bootstrap draws stratified by task, seed 4721; reuse same draws across modes. Task intervals use that task block of each global draw. Percentile bounds with same indexing as frozen script; no multiplicity adjustment or equivalence margin.',
 'scopes':{}}
for scope in ['all']+tasks:
    idx=np.array([i for i,k in enumerate(keys) if scope=='all' or k[0]==scope])
    sampled=draws if scope=='all' else draws[:,tasks.index(scope)*24:(tasks.index(scope)+1)*24]
    point={m:metrics(m,idx) for m in modes};boot={m:metrics(m,sampled) for m in modes}
    item={'n':len(idx),'configurations':{},'paired':{}}
    for m in modes:
        item['configurations'][m]={'point':{k:float(v) for k,v in point[m].items()},'intervals':{k:interval(v) for k,v in boot[m].items()},
            'all_recovered_count':sum(groups[keys[i]][m]['all_expected_recovered'] for i in idx)}
    for left,right in pairs:
        deltas=[groups[keys[i]][left]['score']-groups[keys[i]][right]['score'] for i in idx]
        item['paired'][left+' minus '+right]={'difference':{k:float(point[left][k]-point[right][k]) for k in point[left]},
            'intervals':{k:interval(boot[left][k]-boot[right][k]) for k in point[left]},
            'better_tied_worse':[sum(d>0 for d in deltas),sum(d==0 for d in deltas),sum(d<0 for d in deltas)]}
    result['scopes'][scope]=item
for p in ('090','095'):
    new=result['scopes']['all']['paired']['trained_'+p+' minus init_'+p]
    assert np.allclose(new['intervals']['score'],summary['paired'][p]['intervals']['score'],atol=1e-14,rtol=0)
    assert new['better_tied_worse']==summary['paired'][p]['better_tied_worse']
(R/'supplemental_paired_analysis.json').write_text(json.dumps(result,indent=2))
lines=['# Supplemental task-level and dense comparisons','',result['label']+'.','',result['method'],'',
'Scores and key shares below are percentages; differences and intervals are percentage points. Each task has 24 paired prompts. All-recovered counts require every expected substring, not exact answer-string equality.','']
for scope,item in result['scopes'].items():
    lines+=['## '+scope+' (n='+str(item['n'])+')','',
      '| Configuration | Score [95% CI] | All recovered | Prefill key share | Decode key share |',
      '|---|---:|---:|---:|---:|']
    for m,d in item['configurations'].items():
        p=d['point'];ci=d['intervals']['score']
        lines.append(f"| {m} | {100*p['score']:.3f} [{100*ci[0]:.3f}, {100*ci[1]:.3f}] | {d['all_recovered_count']}/{item['n']} | {100*p['prefill_key_share']:.3f}% | {100*p['decode_key_share']:.3f}% |")
    lines+=['','| Paired comparison | Score difference [95% CI] | Better/tied/worse |','|---|---:|---:|']
    for name,d in item['paired'].items():
        ci=d['intervals']['score'];b=d['better_tied_worse']
        lines.append(f"| {name} | {100*d['difference']['score']:+.3f} [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] | {b[0]}/{b[1]}/{b[2]} |")
    lines.append('')
lines+=['## Key-share aggregation','',
'For each branch and phase: sum the selected causal KV positions over every attention head, query, and routed full-attention layer; divide by the sum of all causally available positions over the same evaluations. The model has eight full-attention layers and 16 query heads each (128 layer-head evaluations per query position). The counter counts gathered key indices, not GPU bytes or the dense probability mass on those indices.','',
'Let N be the prompt token count and T the generated-token count. The first N−1 prompt positions are processed in prefill. The last prompt token is fed during the first one-token generation step and is counted under decode. Prefill availability = 128 × (N−1) × N / 2. Decode availability = 128 × [T × N + T × (T−1)/2]. All 480 records matched these formulas. This boundary convention is identical in all configurations.','',
'For an aggregate over prompts, key share = sum(branch selected keys) / sum(branch available keys), not the arithmetic mean of branch shares, not divided by final context length. Per-task values use only those 24 prompts; global values pool all 96. Thus prompt lengths and generation lengths affect weights. Quality scores instead give each prompt equal weight (and tasks equal weight because there are 24 each). Every bootstrap draw resamples complete prompts, pairs all configurations, and recomputes the ratio of sums.','',
'Each configuration follows its own autoregressive trajectory; these are not identical-Q/K replay measurements or exact matched-key-budget runs. Key share omits tree-building/router work and repeated physical GQA loads.','',
'## Audit and limitations','',
'Independent audit: 480 unique branches, 96 complete five-way pairs; original frozen summary reproduced from raw data; 13 frozen source/snapshot hashes, protocol, manifest and both checkpoint hashes unchanged; scores recomputed from answers; exact availability and routed-call counts; zero fallbacks; 279,688 routed layer calls. Original initial tensors were checked against seed 94121 in all eight layers before inference. The final integrity audit rechecked checkpoint file hashes, not a new training run.','',
'The supplemental bootstrap reproduces the two frozen overall trained-versus-initialized score intervals to numerical precision. All extra task and dense-comparison intervals are exploratory post-hoc outputs. There is no multiplicity correction or predeclared equivalence margin. An interval crossing zero is not evidence of equivalence. Perfect sampled task scores produce degenerate bootstrap intervals; those do not imply certainty on unseen cases.','',
'Single router initialization/checkpoint, synthetic template-based prompts, substring scoring, fixed output caps, one context range, and only 24 cases per task limit generalization. The task effects differ substantially; broad claims should not be based solely on the pooled score. Sparse inference remained slower than dense. Key reduction is not a speedup or cache-memory reduction.','',
'Original results, summary, report, evidence ZIP and figures remain unchanged. This supplement is saved separately.']
(R/'SUPPLEMENTAL_TASK_BREAKDOWN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
original=(R/'FINAL_EXPERIMENT_REPORT.md').read_text(encoding='utf8')
(R/'FINAL_EXPERIMENT_REPORT_WITH_TASK_BREAKDOWN.md').write_text(original+'\n\n'+ '\n'.join(lines)+'\n',encoding='utf8')
shutil.copyfile(Path(__file__),R/'final_experiment_supplement.py')
out=R/'final_experiment_evidence_with_task_breakdown.zip'
with zipfile.ZipFile(R/'final_experiment_evidence.zip') as old,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
    for name in old.namelist():new.writestr(name,old.read(name))
    for name in ('supplemental_paired_analysis.json','SUPPLEMENTAL_TASK_BREAKDOWN.md','FINAL_EXPERIMENT_REPORT_WITH_TASK_BREAKDOWN.md','final_experiment_supplement.py'):new.write(R/name,name)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print(json.dumps({'supplement':str(R/'SUPPLEMENTAL_TASK_BREAKDOWN.md'),'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
print((R/'SUPPLEMENTAL_TASK_BREAKDOWN.md').read_text())
