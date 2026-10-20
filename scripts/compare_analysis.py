"""Compare all numerical saved-result reanalysis values with release references."""
from pathlib import Path
import json,math,argparse
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--actual',type=Path,default=R/'outputs/analysis/derived_statistics.json');p.add_argument('--reference',type=Path,default=R/'results/reference/derived_statistics.json');a=p.parse_args()
n=0;largest=0.
def compare(x,y,path=''):
 global n,largest
 assert type(x)==type(y) or isinstance(x,(float,int)) and isinstance(y,(float,int)),path
 if isinstance(x,dict):
  assert set(x)==set(y),path
  for k in x:compare(x[k],y[k],path+'/'+k)
 elif isinstance(x,list):
  assert len(x)==len(y),path
  for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
 elif isinstance(x,(float,int)) and not isinstance(x,bool):
  n+=1;largest=max(largest,abs(x-y));assert math.isclose(x,y,abs_tol=1e-10,rel_tol=1e-10),(path,x,y)
 else:assert x==y,(path,x,y)
compare(json.loads(a.actual.read_text(encoding='utf-8')),json.loads(a.reference.read_text(encoding='utf-8')))
report=dict(passed=True,numeric_entries=n,max_abs_difference=largest,tolerance=1e-10);out=R/'outputs/verification';out.mkdir(parents=True,exist_ok=True);(out/'analysis_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
