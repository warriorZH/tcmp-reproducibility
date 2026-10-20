"""Synthetic offline integration check; does not read real query labels or run an encoder."""
from pathlib import Path
import sys,json,tempfile,gzip,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from tcmp_repro import common,learner,evaluation
from tcmp_repro.io import read
def main():
 rng=np.random.default_rng(20261020);centers=rng.normal(size=(4,768));centers/=np.linalg.norm(centers,axis=1,keepdims=True)
 tx=np.repeat(centers,12,axis=0)+rng.normal(scale=.015,size=(48,768));tx/=np.linalg.norm(tx,axis=1,keepdims=True)
 q=np.repeat(centers,3,axis=0)+rng.normal(scale=.015,size=(12,768));q/=np.linalg.norm(q,axis=1,keepdims=True)
 l=np.array([c*12+i for c in range(3) for i in range(4)]);y=np.repeat(np.arange(3),4);truth=np.repeat(np.arange(4),12);truth[truth==3]=-1;t=np.repeat(np.arange(4),3);t[t==3]=-1
 with tempfile.TemporaryDirectory() as temp:
  o=Path(temp)/'synthetic';(o/'cache').mkdir(parents=True);(o/'results').mkdir();common.R=learner.R=o;evaluation.P=read(R/'configs/experiment.json')
  b=learner.base(tx,q,l,y,1000,'smoke',evaluation.P['temperature_grid']);evaluation.evaluate_context(o,'smoke',tx,q,l,y,t,truth,{'pool_seen':t<0,'query_only':np.zeros(len(t),bool)},b,4,dict(split='open',regime='fixed_known',rho=.25,role=1,seed=1))
  r=read(o/'results/smoke.json');assert len(r['metrics'])==7 and all(len(x)==3 for x in r['metrics'].values());assert r['learner_hidden_labels_read'] is False
  # Confirm the reusable learner functions accept only arrays of features/support labels.
  import inspect
  assert all(x not in inspect.signature(learner.selftrain).parameters for x in ['pool_truth','query_truth','paths','classes'])
  # Exercise feature-row lookup and the public experiment CLI using a tiny closed fixture.
  from tcmp_repro import run
  fixture=Path(temp)/'fixture';(fixture/'configs').mkdir(parents=True);(fixture/'data/manifests').mkdir(parents=True)
  shutil.copy2(R/'configs/experiment.json',fixture/'configs/experiment.json')
  records=[dict(id=f't{i}',class_name=str(i//12)) for i in range(48)];queries=[dict(id=f'q{i}',class_name=str(i//3)) for i in range(12)]
  fixture_ids=[r['id'] for r in records+queries];support=[f't{c*12+i}' for c in range(4) for i in range(4)]
  d=dict(classes=list(map(str,range(4))),train=records,query=queries,ids=fixture_ids,plans=[dict(seed=1,shot=4,support=support)],roles=[])
  with gzip.open(fixture/'data/manifests/tcmp.json.gz','wt',encoding='utf-8') as f:json.dump(d,f)
  np.save(fixture/'features.npy',np.r_[tx,q]);(fixture/'ids.json').write_text(json.dumps(fixture_ids),encoding='utf-8')
  old_repo=run.REPO;old_argv=sys.argv[:]
  try:
   run.REPO=fixture;sys.argv=['run','--dataset','tcmp','--features',str(fixture/'features.npy'),'--ids',str(fixture/'ids.json'),'--output',str(fixture/'run'),'--suite','closed','--max-contexts','1'];run.main()
  finally:run.REPO=old_repo;sys.argv=old_argv
  assert (fixture/'run/results/closed_s1_n4.json').is_file()
 report=dict(passed=True,synthetic_only=True,classifiers=7,gates=3,feature_dimension=768)
 out=R/'outputs/verification';out.mkdir(parents=True,exist_ok=True);(out/'smoke_test.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
