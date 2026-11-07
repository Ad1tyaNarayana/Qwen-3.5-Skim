"""Operational supervision only; does not alter frozen experiment inputs/code."""
import ctypes,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('W:/Codex-Qwen-Final-20261104')
WORK=Path(__file__).resolve().parent
PYTHON=r'C:\Users\Admin\qwenputer-games-work\.venv\Scripts\python.exe'
env=os.environ.copy()
env.update(FINAL_W_ROOT=str(ROOT),PYTHONPATH=str(WORK/'.deps'),TEMP=str(ROOT/'temp'),TMP=str(ROOT/'temp'),TRITON_CACHE_DIR=str(ROOT/'temp'/'triton'))
def status(stage,**extra):
    d=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),c_free=shutil.disk_usage('C:/').free,w_free=shutil.disk_usage(ROOT).free,**extra)
    for split in ('dev','final'):
        p=ROOT/f'{split}_results.jsonl'
        d[split+'_saved_rows']=len(p.read_text(encoding='utf8').splitlines()) if p.exists() else 0
    tmp=ROOT/'supervisor_status.tmp';tmp.write_text(json.dumps(d,indent=2),encoding='utf8');tmp.replace(ROOT/'supervisor_status.json')
    print(json.dumps(d),flush=True)
def guard():
    assert shutil.disk_usage('C:/').free>=26*2**30,'C disk guard'
    assert shutil.disk_usage(ROOT).free>=10*2**30,'W disk guard'
def run(script,split,log,plot=False):
    guard();e=env.copy()
    if plot:e['PYTHONPATH']=str(ROOT/'plot_deps')+';'+e['PYTHONPATH']
    with (ROOT/log).open('a',encoding='utf8') as f:
        p=subprocess.Popen([PYTHON,str(WORK/script),split],cwd=WORK,env=e,stdout=f,stderr=subprocess.STDOUT)
        while True:
            try:rc=p.wait(timeout=30);break
            except subprocess.TimeoutExpired:status(script+':'+split,launcher_pid=p.pid)
        assert rc==0,(script,split,rc)
def main():
    # Attach to the verified existing development worker, waiting without touching it.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    if sys.argv[1]=='resume':
        run('final_experiment_eval.py','dev','dev_eval.log')
    else:
        pid=int(sys.argv[1]);h=kernel.OpenProcess(0x100000|0x1000,False,pid)
        assert h,('cannot attach to verified dev worker',pid,ctypes.get_last_error())
        try:
            while kernel.WaitForSingleObject(h,30000)==258:status('waiting_for_development',worker_pid=pid)
            rc=ctypes.c_ulong();assert kernel.GetExitCodeProcess(h,ctypes.byref(rc));assert rc.value==0,('development failed',rc.value)
        finally:kernel.CloseHandle(h)
    run('final_experiment_analyze.py','dev','dev_summary.log')
    rows=[json.loads(s) for s in (ROOT/'dev_results.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==20 and len({(r['task'],r['index'],r['mode']) for r in rows})==20
    assert all(r['fallback_count']==0 and r['peak_allocated_gib']<14 for r in rows)
    assert all(r['routed_calls']==8*((r['input_tokens']-2)//512+1+r['generated_tokens']) for r in rows if r['mode']!='dense')
    estimate={'development_branches':20,'final_branches':480,'extrapolated_final_hours':24*sum(r['prefill_s']+r['decode_s_total'] for r in rows)/3600,'max_peak_allocated_gib':max(r['peak_allocated_gib'] for r in rows),'qualification':'Development timing extrapolation only; output lengths, selections and system load may vary. Safety gate does not use answer scores.'}
    (ROOT/'development_feasibility.json').write_text(json.dumps(estimate,indent=2),encoding='utf8')
    guard();status('development_passed',**estimate)
    run('final_experiment_eval.py','final','final_eval.log')
    run('final_experiment_analyze.py','final','final_summary.log',plot=True)
    status('completed',final_summary=str(ROOT/'final_summary.json'),figure=str(ROOT/'quality_vs_decode_key_share.png'))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('blocked',error=repr(exc));raise
