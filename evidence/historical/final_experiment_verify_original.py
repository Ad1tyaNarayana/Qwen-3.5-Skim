"""Post-run integrity audit; frozen evaluator and analysis remain unchanged."""
import collections,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path('W:/Codex-Qwen-Final-20261104'); WORK=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((ROOT/'source_lock.json').read_text())
for name,h in lock['source_sha256'].items():
    assert sha(WORK/name)==h and sha(ROOT/'source'/name)==h,name
for name in ('protocol','manifest'):assert sha(ROOT/(name+'.json'))==lock[name+'_sha256']
assert sha('W:/Codex-ZHA-L6-20261104/R16/router.pt')==lock['trained_router_sha256']
assert sha('W:/Codex-ZHA-L6-20261104/R16/checkpoint.pt')==lock['training_checkpoint_sha256']
manifest=json.loads((ROOT/'manifest.json').read_text())['cases']
cases={(x['task'],x['index']):x for x in manifest if x['split']=='final'}
rows=[json.loads(s) for s in (ROOT/'final_results.jsonl').read_text().splitlines()]
summary=json.loads((ROOT/'final_summary.json').read_text()); groups=collections.defaultdict(list)
assert len(rows)==480 and len({(r['task'],r['index'],r['mode']) for r in rows})==480
assert len(cases)==96
for r in rows:
    c=cases[r['task'],r['index']];N=r['input_tokens'];T=r['generated_tokens']
    assert r['input_sha256']==c['input_sha256'] and r['token_sha256']==c['token_sha256']
    assert r['expected']==c['expected'] and N==c['input_tokens']
    assert 0<T<=c['generation_cap'] and r['fallback_count']==0
    found=[a.lower() in r['generated'].lower() for a in r['expected']]
    assert r['score']==sum(found)/len(found) and r['all_expected_recovered']==all(found)
    assert r['prefill_available']==128*(N-1)*N//2
    assert r['decode_available']==128*(T*N+T*(T-1)//2)
    assert r['routed_calls']==(0 if r['mode']=='dense' else 8*(math.ceil((N-1)/512)+T))
    for phase in ('prefill','decode'):
        assert 0<r[phase+'_selected']<=r[phase+'_available']
        assert r[phase+'_key_share']==r[phase+'_selected']/r[phase+'_available']
    if r['mode']!='dense':
        assert r['checkpoint_sha256']==lock['trained_router_sha256']
        assert r['reference_tensor_sha256']==lock['initial_tensor_sha256' if r['mode'].startswith('init') else 'trained_reference_tensor_sha256']
    groups[r['mode']].append(r)
stats={}
for mode,z in groups.items():
    assert len(z)==96
    s=summary['configurations'][mode]
    assert statistics.mean(r['score'] for r in z)==s['score']
    for phase in ('prefill','decode'):
        assert sum(r[phase+'_selected'] for r in z)/sum(r[phase+'_available'] for r in z)==s[phase+'_key_share']
    stats[mode]={'n':len(z),'all_recovered_count':sum(r['all_expected_recovered'] for r in z),
      'mean_prefill_s':statistics.mean(r['prefill_s'] for r in z),'mean_decode_s':statistics.mean(r['decode_s_total'] for r in z),
      'mean_generated_tokens':statistics.mean(r['generated_tokens'] for r in z),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in z),
      'peak_reserved_gib':max(r['peak_reserved_gib'] for r in z),'ended_on_eos':sum(r['ended_on_eos'] for r in z)}
audit={'status':'passed','rows':480,'paired_prompts':96,'tasks':4,'prompts_per_task':24,'source_files_verified':len(lock['source_sha256']),
       'fallbacks':sum(r['fallback_count'] for r in rows),'routed_layer_calls':sum(r['routed_calls'] for r in rows),
       'total_measured_inference_hours':sum(r['prefill_s']+r['decode_s_total'] for r in rows)/3600,
       'input_tokens_min':min(r['input_tokens'] for r in rows),'input_tokens_max':max(r['input_tokens'] for r in rows),
       'results_sha256':sha(ROOT/'final_results.jsonl'),'summary_sha256':sha(ROOT/'final_summary.json'),'modes':stats,
       'checks':'Frozen source/snapshot/protocol/manifest/checkpoint hashes; unique complete branches; per-case provenance; recomputed scores; exact causal availability; selected bounds and ratios; 8-layer call counts; zero fallback; aggregate summary recomputed.'}
(ROOT/'final_integrity_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
