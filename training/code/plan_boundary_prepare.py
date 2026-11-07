"""Prepare distinct boundary-safe PG-19 books and WikiText articles.

Downloaded book prefixes are capped at 1 MiB each. Splits are fixed by upstream
PG-19 lists and WikiText files; selection uses a stable hash order and lengths,
never evaluation scores. Book and article identities are recorded and checked.
"""
import concurrent.futures,hashlib,json,re,shutil,time,urllib.request
from pathlib import Path
from transformers import AutoTokenizer

base=Path('results/plan_boundary');books=base/'books';books.mkdir(parents=True,exist_ok=True)
t=AutoTokenizer.from_pretrained('model',local_files_only=True)
seed='qwen-boundary-v1:';cap=1024*1024
lists={s:Path(base/f'pg19_{s}_files.txt').read_text().splitlines() for s in ('train','validation','test')}
ids={s:{Path(n).name for n in names} for s,names in lists.items()}
assert all(ids[a].isdisjoint(ids[b]) for a,b in (('train','validation'),('train','test'),('validation','test')))
def get(name):
 p=books/name.replace('/','_')
 if not p.exists():
  if shutil.disk_usage('.').free<20*2**30:raise RuntimeError('disk reserve below 20 GiB')
  for attempt in range(3):
   try:
    with urllib.request.urlopen('https://storage.googleapis.com/deepmind-gutenberg/'+name,timeout=45) as r:data=r.read(cap)
    temp=p.with_suffix('.partial');temp.write_bytes(data);temp.replace(p);break
   except Exception:
    if attempt==2:raise
    time.sleep(attempt+1)
 data=p.read_bytes();text=data.decode('utf8',errors='ignore')
 tokens=t(text,add_special_tokens=False).input_ids
 return {'source':'pg19','split':name.split('/')[0], 'document_id':name,'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'token_count':len(tokens),'window_tokens':16384}
allrows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for split,want in [('train',64),('validation',4),('test',4)]:
  candidates=sorted(lists[split],key=lambda s:hashlib.sha256((seed+s).encode()).hexdigest())
  selected=[];offset=0
  while len(selected)<want:
   batch=candidates[offset:offset+8];offset+=len(batch)
   if not batch:raise RuntimeError('not enough long documents')
   for row in pool.map(get,batch):
    if row['token_count']>=16384 and row['sha256'] not in {x['sha256'] for x in selected} and len(selected)<want:selected.append(row)
   print('pg19',split,'selected',len(selected),'candidates',offset,flush=True)
  allrows.extend(selected)

for split,want in [('train',64),('valid',4),('test',4)]:
 p=Path(f'data/wikitext2_{split}.txt');raw=p.read_text(encoding='utf8')
 matches=list(re.finditer(r'(?m)^[ \t]*=[ \t]+[^\s=].*?[ \t]+=[ \t]*$',raw));rows=[]
 for i,hit in enumerate(matches):
  end=matches[i+1].start() if i+1<len(matches) else len(raw)
  text=raw[hit.start():end];tokens=t(text,add_special_tokens=False).input_ids
  if len(tokens)>=4096:
   rows.append({'source':'wikitext2','split':split,'document_id':f'{split}:{i}','start_char':hit.start(),'end_char':end,'sha256':hashlib.sha256(text.encode()).hexdigest(),'token_count':len(tokens),'window_tokens':4096})
 rows.sort(key=lambda x:hashlib.sha256((seed+x['document_id']).encode()).hexdigest())
 assert len(rows)>=want
 allrows.extend(rows[:want])
by_split={s:{x['sha256'] for x in allrows if x['split']==s} for s in ('train','valid','validation','test')}
train=by_split['train'];val=by_split['valid']|by_split['validation'];test=by_split['test']
assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
manifest={'selection_seed':seed,'pg19_manifest_revision':'4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b','pg19_note':'individual official-split book prefix capped 1 MiB; each window stays within one book','wiki_note':'single top-level article per window, official splits','checks':'upstream PG book IDs and exact document hashes disjoint across train/validation/test; no near-duplicate guarantee','documents':allrows}
(base/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('complete documents',len(allrows),'free bytes',shutil.disk_usage('.').free,flush=True)
