"""Summarize the original 476 TCMP per-method JSONs without prediction caches."""
from pathlib import Path
import argparse,json,math
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/reference/formal/results'));p.add_argument('--output',type=Path,default=Path('outputs/formal_summary.json'));a=p.parse_args()
rs=[json.loads(f.read_text(encoding='utf-8')) for f in a.input.glob('*.json')];assert len(rs)==476
out={'closed':{},'open':{}}
for shot in [4,16]:
 out['closed'][str(shot)]={}
 for method in ['lr','fixed','soft','lp']:
  rows=[r for r in rs if r['key'].startswith('closed_') and r['shot']==shot and r['method']==method];assert len(rows)==7
  v=[r['classification']['accuracy'] for r in rows];out['closed'][str(shot)][method]=dict(mean_accuracy=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),seeds=7)
for condition in ['clean_full','add10','add30','matched30','clean_thin']:
 out['open'][condition]={}
 for method in ['lr','fixed','soft','lp']:
  rows=[r for r in rs if r['key'].startswith('open_') and r['key'].endswith('_'+condition) and r['method']==method];assert len(rows)==21
  out['open'][condition][method]=dict(mean_accuracy=float(np.mean([r['classification']['accuracy'] for r in rows])),contexts=21)
reference=a.input.parent/'artifacts/summary.json'
if reference.exists():
 expected=json.loads(reference.read_text(encoding='utf-8'))
 for shot in ['4','16']:
  for method in ['lr','fixed','soft','lp']:
   for actual_key,ref_key in [('mean_accuracy','mean_accuracy'),('sample_sd','sd')]:assert math.isclose(out['closed'][shot][method][actual_key],expected['closed'][shot][method][ref_key],abs_tol=1e-10)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print('FORMAL_SUMMARY_COMPLETE',a.output)
