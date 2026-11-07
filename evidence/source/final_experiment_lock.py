"""Freeze source, data and reference provenance before any evaluation."""
import hashlib,json,os,shutil,torch
from pathlib import Path
from zha_l6_controlled import LAYERS,initial_references
root=Path(os.environ['FINAL_W_ROOT']);files=['final_experiment_lock.py','final_experiment_prepare.py','final_experiment_eval.py','final_experiment_analyze.py','final_router_counter.py','test_final_router_counter.py','zha_l6_controlled.py','zha_l6_sparse.py','zha_l6_reference.py','vector_router.py','maskless_decode_router.py','sparse_kernel.py','incremental_tree.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
trained_path=Path('W:/Codex-ZHA-L6-20261104/R16/router.pt');saved_path=Path('W:/Codex-ZHA-L6-20261104/R16/checkpoint.pt')
a=torch.load(trained_path,map_location='cpu',weights_only=True);b=torch.load(saved_path,map_location='cpu',weights_only=True);re=initial_references(16)
for l in LAYERS:
 assert torch.equal(a['initial'][l],re[l]) and torch.equal(b['initial'][l],re[l])
 assert torch.equal(a['refs'][l],b['refs'][str(l)])
initial_hash=hashlib.sha256(b''.join(a['initial'][l].numpy().tobytes() for l in LAYERS)).hexdigest()
trained_hash=hashlib.sha256(b''.join(a['refs'][l].numpy().tobytes() for l in LAYERS)).hexdigest()
lock={'source_sha256':{p:sha(p) for p in files},'trained_router_sha256':sha(trained_path),'training_checkpoint_sha256':sha(saved_path),
      'initial_tensor_sha256':initial_hash,'trained_reference_tensor_sha256':trained_hash,
      'protocol_sha256':sha(root/'protocol.json'),'manifest_sha256':sha(root/'manifest.json'),
      'initialization_check':'saved router and optimizer checkpoint initial tensors exactly match seed 94121 in all eight layers',
      'verification':'saved final references match the controlled pre-32K training checkpoint in all eight layers'}
path=root/'source_lock.json'
if path.exists():assert json.loads(path.read_text())==lock
else:path.write_text(json.dumps(lock,indent=2),encoding='utf8')
dest=root/'source';dest.mkdir(exist_ok=True)
for p in files:
 q=dest/p
 if q.exists():assert sha(q)==sha(p)
 else:shutil.copyfile(p,q);assert sha(q)==sha(p)
print(json.dumps(lock,indent=2),flush=True)
