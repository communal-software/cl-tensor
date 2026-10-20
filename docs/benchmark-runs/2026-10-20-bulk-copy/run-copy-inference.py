import os, subprocess, csv, json, hashlib, time
from pathlib import Path
out=Path('/tmp/copy-inference-2026-10-20');out.mkdir(exist_ok=True)
def snapshot():
 return {c:subprocess.run(c.split(),capture_output=True,text=True).stdout for c in ('vm_stat','sysctl vm.swapusage','ps -axo pid,pcpu,comm')}
with (out/'results.csv').open('w') as stream:
 writer=csv.writer(stream);writer.writerow(['pair','variant','model','phase','milliseconds'])
 for pair in range(1,4):
  for variant in (('before','after') if pair%2 else ('after','before')):
   print(f'pair {pair} {variant}',flush=True)
   before=snapshot();env=dict(os.environ,COPY_VARIANT=variant,VECLIB_MAXIMUM_THREADS='1',TRIVIAL_SIMD_BACKEND='native')
   path=out/f'{pair}-{variant}.log'
   with path.open('w') as log:
    p=subprocess.run(['sbcl','--dynamic-space-size','3072','--script','/tmp/copy-inference-bench.lisp'],env=env,stdout=log,stderr=subprocess.STDOUT)
   lines=path.read_text().splitlines();rows=[s.split(',')[1:] for s in lines if s.startswith('RESULT,')]
   verify=[s for s in lines if s.startswith('VERIFY,')]
   assert p.returncode==0 and len(rows)==6 and len(verify)==2
   assert len({(r[0],r[1]) for r in rows})==6
   writer.writerows([[pair,variant,*r] for r in rows]);stream.flush()
   (out/f'{pair}-{variant}-environment.json').write_text(json.dumps({'before':before,'after':snapshot()},indent=2))
   print(f'pair {pair} {variant} verified',flush=True)
