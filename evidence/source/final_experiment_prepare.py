"""Generate all 100 prompts and lock a five-configuration comparison."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from transformers import AutoTokenizer

root=Path(os.environ['FINAL_W_ROOT']);root.mkdir(parents=True,exist_ok=True)
base=Path('results/plan_scaled/RULER_pinned/scripts/data/synthetic').resolve();sys.path.insert(0,str(base))
from constants import TASKS
specs={
 'vt':('variable_tracking',['--type_haystack','noise','--num_chains','1','--num_hops','4'],30),
 'cwe':('common_words_extraction',['--freq_cw','30','--freq_ucw','3','--num_cw','10'],120),
 'niah_distractor':('niah',['--type_haystack','needle','--num_needle_k','4','--num_needle_v','3'],128),
 'niah_multikey':('niah',['--type_haystack','noise','--num_needle_k','4','--num_needle_q','4','--num_needle_v','1'],128),
}
seeds={'dev':{'vt':291001,'cwe':291101,'niah_distractor':291201,'niah_multikey':291301},
       'final':{'vt':292001,'cwe':292101,'niah_distractor':292201,'niah_multikey':292301}}
protocol={'model':'Qwen/Qwen3.5-4B-Base','model_revision':'1001bb4d826a52d1f399e183466143f4da7b741b',
 'ruler_commit':'db43f92ff634bbce1c33d1fb15858aec4bba509a','target_length':16384,
 'dev_cases_per_task':1,'final_cases_per_task':24,'task_specs':specs,'seeds':seeds,
 'configurations':['dense','init_090','trained_090','init_095','trained_095'],
 'router':'saved original R16 initialization or pre-32K controlled R16, C4/L6, sink+current chunk, adaptive top-p, grouped, both prefill/decode',
 'batch':1,'prefill_chunk':512,'generation':'greedy, stop at EOS or task cap',
 'metrics':'case-insensitive expected substrings: fraction recovered and all recovered; selected / causally visible head-query keys, prefill and decode separately',
 'bootstrap':'10000 paired prompt-level resamples stratified by task; seed 4721; 95% percentile; no independent head/item resampling',
 'measure_mass':False,'disk_guards':'C: >=26 GiB, W: >=10 GiB; checked each block/group/tile/step',
 'splits':'4 dev prompts separate from 96 final prompts; do not tune after inspecting answers',
 'limitation':'generated synthetic tasks share templates; no semantic deduplication; p does not equal key budget'}
path=root/'protocol.json'
if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
else:path.write_text(json.dumps(protocol,indent=2),encoding='utf8')
tok=AutoTokenizer.from_pretrained('model',local_files_only=True);manifest=[]
for split in ('dev','final'):
 for label,(task,extra,cap) in specs.items():
  name=f'{split}_{label}';target=root/'cases'/name/'validation.jsonl';count=1 if split=='dev' else 24
  if not target.exists():
   cfg=TASKS[task];args=[sys.executable,str(base/(task+'.py')),'--save_dir',str(root/'cases'),'--save_name',name,
     '--tokenizer_path',str(Path('model').resolve()),'--tokenizer_type','hf','--max_seq_length','16384',
     '--tokens_to_generate',str(cap),'--num_samples',str(count),'--random_seed',str(seeds[split][label]),
     '--template',cfg['template']+cfg['answer_prefix']]+extra
   print('GENERATE',name,'seed',seeds[split][label],flush=True);subprocess.run(args,check=True)
  rows=[json.loads(s) for s in target.read_text(encoding='utf8').splitlines()];assert len(rows)==count
  for i,r in enumerate(rows):
   ids=tok(r['input']+r['answer_prefix'],add_special_tokens=False).input_ids
   manifest.append({'split':split,'task':label,'index':i,'seed':seeds[split][label],
       'input_sha256':hashlib.sha256(r['input'].encode()).hexdigest(),
       'token_sha256':hashlib.sha256(str(ids).encode()).hexdigest(),'input_tokens':len(ids),
       'expected':r['outputs'],'generation_cap':cap,'case_file_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
assert len(manifest)==100 and len({r['input_sha256'] for r in manifest})==100
prior=set();scanned=0
for oldroot in (Path('results'),Path('W:/Codex-ZHA-L6-20261104/gate'),Path('W:/Codex-ZHA-L6-20261104/long_gate'),Path('W:/Codex-ZHA-L6-20261104/quality_32k_update/cases')):
 for f in oldroot.rglob('validation.jsonl'):
  try:
   with f.open(encoding='utf8') as h:
    for line in h:
     z=json.loads(line)
     if 'input' in z:prior.add(hashlib.sha256(z['input'].encode()).hexdigest());scanned+=1
  except (OSError,ValueError):pass
overlap=[r for r in manifest if r['input_sha256'] in prior]
if overlap:raise RuntimeError(f'exact previous input overlap: {len(overlap)}')
(root/'manifest.json').write_text(json.dumps({'cases':manifest,'prior_rows_scanned':scanned,'exact_overlap':0},indent=2),encoding='utf8')
print('LOCKED',len(manifest),'prior_rows',scanned,'overlap',len(overlap),flush=True)
