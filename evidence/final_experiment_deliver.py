"""Presentation-only report, readable supplemental plot and evidence archive."""
import json,shutil,zipfile,hashlib
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('W:/Codex-Qwen-Final-20261104');W=Path(__file__).resolve().parent
s=json.loads((R/'final_summary.json').read_text());a=json.loads((R/'final_integrity_audit.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
styles=[('init_090','#d87522','o','Untrained p=.90'),('trained_090','#1965b0','o','Trained p=.90'),('init_095','#d87522','s','Untrained p=.95'),('trained_095','#1965b0','s','Trained p=.95')]
for ax in axes:
    for mode,col,m,label in styles:
        d=s['configurations'][mode];x=d['decode_key_share']*100;y=d['score']*100
        xi=[v*100 for v in d['intervals']['decode_key_share']];yi=[v*100 for v in d['intervals']['score']]
        ax.errorbar(x,y,xerr=[[x-xi[0]],[xi[1]-x]],yerr=[[y-yi[0]],[yi[1]-y]],fmt=m,color=col,capsize=3,label=label)
    ax.axhline(s['configurations']['dense']['score']*100,color='#555555',ls='--',label='Dense score')
    ax.set(xlabel='Decode key share (%)',ylabel='Mean partial-credit score (%)');ax.grid(alpha=.2)
axes[0].plot(100,s['configurations']['dense']['score']*100,'D',color='#555555');axes[0].set(xlim=(-2,103),ylim=(0,102),title='Full key-share range')
axes[1].set(xlim=(1,8),ylim=(72,99),title='Sparse configurations — magnified');axes[1].legend(loc='lower right',fontsize=9)
fig.suptitle('Frozen 16K test: 96 paired prompts; 95% prompt-stratified bootstrap intervals')
fig.savefig(R/'quality_vs_decode_key_share_readable.png',dpi=180);fig.savefig(R/'quality_vs_decode_key_share_readable.pdf');plt.close(fig)
lines=['# Frozen Qwen hierarchical-router experiment — final results','',
'Completed 2026-11-05. All 480 branches (96 prompts × 5 configurations) and all 20 separate development branches completed. The original frozen analysis and figures are preserved. This report and the readable two-panel figure are post-run presentation artifacts, not changes to the protocol.','',
'## Finding','',
'At p=0.90, the controlled trained R16 router improved answer quality over its exact original initialization while using fewer selected keys in both prefill and decode. This supports a useful training benefit on this fixed 16K synthetic test set. At p=0.95, training also reduced selected keys, but its answer-score advantage has a paired interval crossing zero. Near-equality with dense in aggregate is not proof of equivalence or general superiority.','',
'## Aggregate results','',
'| Configuration | Partial-credit score | All expected recovered | Prefill key share | Decode key share |',
'|---|---:|---:|---:|---:|']
for mode in s['configurations']:
    d=s['configurations'][mode];lines.append(f"| {mode} | {100*d['score']:.3f}% | {a['modes'][mode]['all_recovered_count']}/96 | {100*d['prefill_key_share']:.3f}% | {100*d['decode_key_share']:.3f}% |")
lines+=['','Shares are total selected keys divided by total causally available keys over head-query evaluations, separately by phase. They are not physical memory traffic, attention probability retained, cache compression, or measured speedup. Different generated trajectories and lengths contribute to these totals.','',
'## Paired training effect','']
for p in ('090','095'):
    d=s['paired'][p];ci=d['intervals']['score'];b=d['better_tied_worse'];red=1-s['configurations']['trained_'+p]['decode_key_share']/s['configurations']['init_'+p]['decode_key_share']
    lines.append(f"- p=.{p[1:]}: trained minus untrained score {100*d['mean_score_difference']:+.3f} percentage points; paired 95% interval [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}]. Better/tied/worse: {b[0]}/{b[1]}/{b[2]}. Aggregate decode key share falls {100*red:.1f}% relative to untrained.")
lines+=['','Intervals use the frozen 10,000 whole-prompt paired resamples stratified by task, seed 4721. They reflect this sampled prompt population, not model-seed variability. No multiple-comparison adjustment was predeclared. No noninferiority/equivalence margin was predeclared.','',
'## Task scores (24 prompts each)','',
'| Task | Dense | Init .90 | Trained .90 | Init .95 | Trained .95 |','|---|---:|---:|---:|---:|---:|']
for task,t in s['tasks'].items():lines.append('| '+task+' | '+' | '.join(f"{100*t['modes'][m]['score']:.2f}%" for m in s['configurations'])+' |')
lines+=['','The largest p=.90 gain is variable tracking; the task gains are not uniform. All four task types have equal numbers of prompts.','',
'## Method and controls','',
'Qwen/Qwen3.5-4B-Base revision 1001bb4d826a52d1f399e183466143f4da7b741b; BF16, batch one, RTX 5070 Ti 16 GB, Windows. Corrected C4/L6 grouped router in all eight full-attention layers, with recurrent layers retained. Sparse prefill and decode; 512-token prefill blocks, sinks and current chunk, adaptive p=.90/.95. Fresh KV and recurrent state per branch; greedy generation, frozen task caps; measure_mass disabled.','',
'The trained router is W:/Codex-ZHA-L6-20261104/R16/router.pt, before the one-document 32K continuation. Its saved original initialization was verified bitwise against seed 94121 and the saved optimizer checkpoint across all eight full-attention layers. No further training was performed for this experiment.','',
'Pinned RULER generator commit db43f92ff634bbce1c33d1fb15858aec4bba509a. Four development prompts and 96 final prompts were generated and hashed before evaluation. Final tasks: variable tracking, common-word extraction, needle retrieval with distractors, and multi-key needle retrieval, 24 each. No exact prompt overlaps with the 460 earlier checked prompts; semantic overlap was not excluded. Actual input lengths: '+str(a['input_tokens_min'])+'–'+str(a['input_tokens_max'])+' tokens. Scoring is the frozen case-insensitive expected-substring fraction, plus whether every expected item appeared. It does not penalize extra answers or independently grade reasoning.','',
'Post-run integrity audit passed: 480 unique branches, 96 complete five-way prompt groups, all 13 frozen source files and snapshots unchanged, protocol/manifest/checkpoint hashes unchanged, scores recomputed from answers, exact causal-availability counts, selected-key bounds and ratios, '+str(a['routed_layer_calls'])+' routed layer calls verified, zero dense routing fallbacks. A routing fallback is distinct from the unchanged reference implementations used for the recurrent layers.','',
'## Runtime and resources','',
f"Total measured prefill plus decode time: {a['total_measured_inference_hours']:.3f} hours across 480 branches. The supervisor completed at 2026-11-05 11:16:44 UTC. The sweep began around 2026-11-04 19:44 UTC. These measurements are operational timings, not a controlled repeated speed benchmark.",'',
'| Configuration | Mean prefill s | Mean decode s | Mean generated tokens | Peak allocated GiB |','|---|---:|---:|---:|---:|']
for mode,d in a['modes'].items():lines.append(f"| {mode} | {d['mean_prefill_s']:.2f} | {d['mean_decode_s']:.2f} | {d['mean_generated_tokens']:.2f} | {d['peak_allocated_gib']:.3f} |")
lines+=['','The sparse reference implementation is much slower than the dense baseline despite selecting fewer keys. Generated lengths differ, so total decode seconds are not a matched-token throughput comparison. Peak PyTorch allocation is not total GPU memory use. Dense uses SDPA; recurrent operations use the same installed reference implementations for every configuration. No production speedup is established.','',
'The development workers exited unexpectedly after 18 saved branches; those rows were preserved and the remaining two resumed under an independent supervisor. All 480 final branches completed without a logged evaluation error. Outputs and TEMP/TMP/Triton cache were on W:. C: >=26 GiB and W: >=10 GiB guards stayed enabled. A ~2.24 GiB C: decline occurred during the run, but no guard failed; its cause was not established by the read-only investigation. No files were deleted and no system settings changed.','',
'## Limits and interpretation','',
'This is one initialization/checkpoint and four synthetic task families at approximately 16K, not broad language quality, length extrapolation, a paper reproduction, or evidence of gains beyond dense Qwen. Prompts share templates. The thresholds were fixed rather than matched for exact key budgets. Nevertheless, at p=.90 training achieved both better aggregate answers and lower aggregate key shares; its benefit is not explained by merely selecting more keys in these measurements. The p=.95 trained score (95.000%) is close to dense (95.069%), but all-items recovery is 75/96 versus 77/96, and equivalence was not established.','',
'## Evidence and reproduction','',
'All paths below are relative to W:/Codex-Qwen-Final-20261104/.','',
'- `final_results.jsonl`: all 480 answers, expectations, per-branch counts, times and hashes.','- `final_summary.json`: frozen paired bootstrap results and per-task metrics.','- `final_integrity_audit.json`: independent recomputation, counters, hashes and timing summaries.','- `protocol.json`, `manifest.json`, `source_lock.json`, `cases/`: frozen definitions, seeds, prompts and provenance.','- `source/`: frozen evaluation, router, counter and analysis code.','- `dev_results.jsonl`, `dev_summary.json`, `development_feasibility.json`, `counter_test.log`: development and correctness evidence.','- `final_eval.log`, `supervisor_resume.log`, `supervisor_status.json`: execution evidence.','- `quality_vs_decode_key_share.png/.pdf`: original frozen figure (labels crowded).','- `quality_vs_decode_key_share_readable.png/.pdf`: same values and intervals with a magnified panel.','',
'To reproduce the frozen summary/figure with the existing environment, from the original workspace:','',
'```powershell',"$env:FINAL_W_ROOT='W:\\Codex-Qwen-Final-20261104'", "$env:PYTHONPATH='W:\\Codex-Qwen-Final-20261104\\plot_deps;C:\\Users\\Admin\\Documents\\Codex\\2026-11-04\\task\\.deps'", "& 'C:\\Users\\Admin\\qwenputer-games-work\\.venv\\Scripts\\python.exe' final_experiment_analyze.py final",'```','',
'The archive contains frozen source, prompts, outputs, logs, analysis, plots, audit code and the controlled router/optimizer checkpoints. It excludes base-model weights, Python environment, installed plotting dependencies and temporary compiler caches. Checkpoint paths in the scripts reflect the original local layout. Re-running inference requires that model/environment/layout or an explicitly documented port; re-running it is not needed to inspect these results.','',
'Raw-result SHA256: `'+a['results_sha256']+'`. Frozen-summary SHA256: `'+a['summary_sha256']+'`.']
(R/'FINAL_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
for name in ('final_experiment_verify.py','final_experiment_deliver.py','final_experiment_supervise.py'):shutil.copyfile(W/name,R/name)
archive=R/'final_experiment_evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in R.iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl','.md','.log','.err','.png','.pdf','.py'):z.write(p,p.name)
    for folder in ('cases','source'):
        for p in (R/folder).rglob('*'):
            if p.is_file():z.write(p,p.relative_to(R))
    for name in ('router.pt','checkpoint.pt'):z.write(Path('W:/Codex-ZHA-L6-20261104/R16')/name,'checkpoints/R16/'+name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'report':str(R/'FINAL_EXPERIMENT_REPORT.md'),'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
