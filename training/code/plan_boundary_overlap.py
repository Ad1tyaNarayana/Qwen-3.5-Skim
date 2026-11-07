"""Quantify cross-split exact 13-word shingle overlap without printing text."""
import collections,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer
b=Path('results/plan_boundary');rows=json.loads((b/'dataset_manifest.json').read_text())['documents'];t=AutoTokenizer.from_pretrained('model',local_files_only=True);raw={}
def window(row):
 if row['source']=='pg19':text=Path(row['path']).read_bytes().decode('utf8',errors='ignore')
 else:
  split=row['split']
  if split not in raw:raw[split]=Path(f'data/wikitext2_{split}.txt').read_text(encoding='utf8')
  text=raw[split][row['start_char']:row['end_char']]
 ids=t(text,add_special_tokens=False).input_ids[:row['window_tokens']]
 return t.decode(ids,skip_special_tokens=False)
def shingles(text):
 words=re.findall(r'\w+',text.lower())
 return {hashlib.blake2b(' '.join(words[i:i+13]).encode(),digest_size=8).digest() for i in range(max(0,len(words)-12))}
train=set();hold=[]
for row in rows:
 s=shingles(window(row))
 if row['split']=='train':train.update(s)
 else:hold.append((row,s))
result=[]
for row,s in hold:
 overlap=len(s&train)
 result.append({'source':row['source'],'split':row['split'],'document_id':row['document_id'],'unique_shingles':len(s),'overlap_train':overlap,'overlap_fraction':overlap/max(1,len(s))})
out={'method':'exact lowercased 13-word shingles over selected tokenizer-decoded windows; overlap against union of train shingles; no semantic or paraphrase deduplication','train_unique_shingles':len(train),'documents':result}
(b/'split_overlap.json').write_text(json.dumps(out,indent=2))
print('train shingles',len(train),'holdout docs',len(hold),'max overlap fraction',max(x['overlap_fraction'] for x in result))
