"""Check matched training provenance and summarize held-out teacher KL."""
import json, os
from pathlib import Path
root=Path(os.environ['ZHA_W_ROOT'])
a,b=(json.loads((root/f'R{r}'/'training.json').read_text(encoding='utf8')) for r in (4,16))
assert a['dataset_sha256']==b['dataset_sha256']
assert len(a['progress'])==len(b['progress'])==256
for x,y in zip(a['progress'],b['progress']):
    for k in ('document_id','input_sha256','sampled_positions','epoch','index','lr'):
        assert x[k]==y[k],(k,x['document_id'])
assert a['unique_train_tokens']==b['unique_train_tokens'] and a['sampled_layer_query_rows']==b['sampled_layer_query_rows']
def means(rows):
    result={}
    for src in ('pg19','wikitext2'):
        vals=[sum(v['sum_kl'] for v in row['losses'].values())/len(row['losses']) for row in rows if row['source']==src]
        result[src]=sum(vals)/len(vals)
    return result
result={'matched_document_passes':256,'unique_input_tokens':a['unique_train_tokens'],
        'token_exposures_per_run':a['teacher_token_exposures'],
        'sampled_layer_query_rows_per_run':a['sampled_layer_query_rows'],
        'R4_before':means(a['validation_before']),'R4_after':means(a['validation_after']),
        'R16_before':means(b['validation_before']),'R16_after':means(b['validation_after']),
        'R4_document_seconds':sum(r['seconds'] for r in a['progress']),
        'R16_document_seconds':sum(r['seconds'] for r in b['progress']),
        'limitation':'teacher KL, not model task accuracy; fixed sample positions repeat across epochs'}
(root/'training_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
