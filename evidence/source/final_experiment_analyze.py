"""Prompt-stratified paired bootstrap and quality-vs-decode-key-share plot."""
import collections,json,os,random,statistics,sys
from pathlib import Path
root=Path(os.environ['FINAL_W_ROOT']);split=sys.argv[1];assert split in ('dev','final')
protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'));modes=protocol['configurations']
rows=[json.loads(s) for s in (root/f'{split}_results.jsonl').read_text(encoding='utf8').splitlines()]
groups=collections.defaultdict(dict)
for r in rows:
 key=(r['task'],r['index']);assert r['mode'] not in groups[key];groups[key][r['mode']]=r
assert all(set(v)==set(modes) for v in groups.values()),'incomplete case; wait or inspect raw results'
expected=1 if split=='dev' else 24;tasks=('vt','cwe','niah_distractor','niah_multikey')
assert all(sum(k[0]==task for k in groups)==expected for task in tasks)
for key,d in groups.items():
 assert len({x['token_sha256'] for x in d.values()})==1
 for x in d.values():assert 0<x['prefill_selected']<=x['prefill_available'] and 0<x['decode_selected']<=x['decode_available'] and x['fallback_count']==0
def point(cases,mode):
 z=[groups[k][mode] for k in cases]
 return {'score':statistics.mean(r['score'] for r in z),'all_recovered':statistics.mean(r['all_expected_recovered'] for r in z),
  'prefill_key_share':sum(r['prefill_selected'] for r in z)/sum(r['prefill_available'] for r in z),
  'decode_key_share':sum(r['decode_selected'] for r in z)/sum(r['decode_available'] for r in z)}
keys=sorted(groups);result={'split':split,'n':len(keys),'configurations':{},'tasks':{},'paired':{},
 'bootstrap':'10000 prompt-level paired resamples stratified by task; 95% percentile; seed 4721; not heads or expected items',
 'key_share':'sum selected / sum causally available over head-query evaluations, separately by phase',
 'qualification':'estimated threshold p is not true dense attention mass; distinct model trajectories may select different keys'}
for mode in modes:result['configurations'][mode]=point(keys,mode)
for task in tasks:
 sub=[k for k in keys if k[0]==task];result['tasks'][task]={'n':len(sub),'modes':{m:point(sub,m) for m in modes}}
rng=random.Random(4721);by_task={task:[k for k in keys if k[0]==task] for task in tasks}
samples={m:{k:[] for k in ('score','all_recovered','prefill_key_share','decode_key_share')} for m in modes}
diffs={threshold:{'score':[],'decode_key_share':[],'prefill_key_share':[]} for threshold in ('090','095')}
for _ in range(10000):
 pick=[rng.choice(k) for task in tasks for k in [by_task[task]] for _ in range(len(k))]
 p={m:point(pick,m) for m in modes}
 for m in modes:
  for k in samples[m]:samples[m][k].append(p[m][k])
 for threshold in diffs:
  for k in diffs[threshold]:diffs[threshold][k].append(p['trained_'+threshold][k]-p['init_'+threshold][k])
def ci(values):
 v=sorted(values);return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]
for m in modes:result['configurations'][m]['intervals']={k:ci(v) for k,v in samples[m].items()}
for threshold in ('090','095'):
 d=[groups[k]['trained_'+threshold]['score']-groups[k]['init_'+threshold]['score'] for k in keys]
 result['paired'][threshold]={'better_tied_worse':[sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)],
  'mean_score_difference':statistics.mean(d),'intervals':{k:ci(v) for k,v in diffs[threshold].items()},
  'decode_key_share_difference':result['configurations']['trained_'+threshold]['decode_key_share']-result['configurations']['init_'+threshold]['decode_key_share'],
  'prefill_key_share_difference':result['configurations']['trained_'+threshold]['prefill_key_share']-result['configurations']['init_'+threshold]['prefill_key_share']}
(root/f'{split}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if split=='final':
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,5.4),layout='constrained')
 for m,color,marker in [('init_090','#e07a27','o'),('init_095','#e07a27','s'),('trained_090','#1965b0','o'),('trained_095','#1965b0','s')]:
  d=result['configurations'][m];x=d['decode_key_share'];y=d['score'];xi=d['intervals']['decode_key_share'];yi=d['intervals']['score']
  ax.errorbar(x,y,xerr=[[max(0,x-xi[0])],[max(0,xi[1]-x)]],yerr=[[max(0,y-yi[0])],[max(0,yi[1]-y)]],fmt=marker,color=color,capsize=3,label=m.replace('_',' ')+' (95% CI)')
  ax.annotate(m.replace('_',' '),(x,y),xytext=(5,7),textcoords='offset points',fontsize=8)
 dense=result['configurations']['dense'];ax.axhline(dense['score'],color='#555555',linestyle='--',label='dense score')
 ax.plot([1],[dense['score']],marker='D',color='#555555',markersize=5)
 ax.set(xlabel='Decode selected / causally visible keys',ylabel='Mean partial-credit answer score',title='16K quality versus decode key share (96 prompts)',xlim=(0,1.03),ylim=(-.02,1.05))
 ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right');fig.savefig(root/'quality_vs_decode_key_share.png',dpi=180);fig.savefig(root/'quality_vs_decode_key_share.pdf');plt.close(fig)
print(json.dumps(result,indent=2))
