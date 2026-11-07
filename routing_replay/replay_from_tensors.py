import json, hashlib, torch
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from zha_l6_reference import make_tree, selected_indices

torch.set_num_threads(4)
out=(ROOT/'recomputed');out.mkdir(exist_ok=True)
teacher=(ROOT/'teacher/doc0_layer3.pt')
checkpoint=(ROOT/'checkpoints/router.pt')
d=torch.load(teacher,map_location='cpu',weights_only=True)
c=torch.load(checkpoint,map_location='cpu',weights_only=True)
layer=3;head=0;kv=0;p=.9;scale=d['scale'];keys=d['key'];states={}
for label,refs in [('untrained',c['initial'][layer]),('trained',c['refs'][layer])]:
 levels=make_tree(keys,refs,scale,64,1e7,4,6)
 records=[]
 for j in range(56,64):
  end=int(d['positions'][j])+1;q=d['queries'][head,j];local=1+((end-2)//4)*4;chosen=None;trace=[]
  for lev in range(len(levels)-1,-1,-1):
   means,bias=levels[lev];width=4**(lev+1);eligible=min(means.shape[1],(local-1)//width)
   if chosen is None:candidates=list(range(eligible))
   else:
    candidates=[x*4+t for x in chosen for t in range(4) if x*4+t<eligible]
    candidates+=list(range(((local-1)//(width*4))*4,eligible))
   if candidates:
    idx=torch.tensor(candidates);scores=(torch.einsum('nrd,d->nr',means[kv,idx],q.float())*scale+bias[kv,idx]).amax(-1)
    order=torch.argsort(scores,descending=True,stable=True);probs=scores[order].softmax(0)
    n=int((probs.cumsum(0)-probs<p).sum());chosen=idx[order[:n]].tolist()
   else:chosen=[]
   trace.append({'level':lev+1,'span':width,'eligible':eligible,'candidates':candidates,'selected':chosen})
  ids=sorted([0]+list(range(local,end))+[1+x*4+t for x in chosen for t in range(4)])
  ref=selected_indices(q,levels,end,kv,scale,p,4).tolist();assert ids==ref
  mass=(keys[kv,:end].float()@q.float()*scale).softmax(0)[ids].sum().item()
  records.append({'query_index':j,'end':end,'selected_keys':ids,'dense_mass_retained':mass,'levels':trace})
 states[label]=records
result={'provenance':'Offline replay on saved dense-teacher Q/K; not the final 96-prompt benchmark, not a generation rollout or timing demo.',
 'model':'Qwen3.5-4B-Base','layer_zero_based':layer,'query_head_zero_based':head,'p':p,'geometry':'C4/L6/R16',
 'selection_rule':'Layer 3, head 0, final eight saved query positions chosen by fixed index, not ranked by outcomes.',
 'teacher_file':str(teacher),'teacher_sha256':hashlib.sha256(teacher.read_bytes()).hexdigest(),
 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 'validation':'All 16 selections exactly match the existing CPU selected_indices reference.',
 'states':states}
(out/'trace.json').write_text(json.dumps(result),encoding='utf8')
for label,rows in states.items():print(label,[(r['end'],len(r['selected_keys']),round(r['dense_mass_retained'],3)) for r in rows])
