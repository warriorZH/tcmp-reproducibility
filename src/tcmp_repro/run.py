"""Portable experiment orchestration; existing core learners remain unchanged."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from . import common,learner,evaluation
from .io import read,write,sha

REPO=Path(__file__).resolve().parents[2]

def contexts(d,tx,qx,dataset,suite,seed_filter=None,role_filter=None,rho_filter=None):
 classes=d['classes'];ci={c:i for i,c in enumerate(classes)}
 cname=lambda r:r.get('class',r.get('class_name'))
 ty=np.array([ci[cname(r)] for r in d['train']]);qt=np.array([ci[cname(r)] for r in d['query']]);ti={r['id']:i for i,r in enumerate(d['train'])}
 if suite in ['closed','all']:
  for plan in d['plans']:
   if seed_filter is not None and plan['seed']!=seed_filter:continue
   l=np.array([ti[v] for v in plan['support']]);key=f"closed_s{plan['seed']}_n{plan['shot']}"
   b=learner.base(tx,qx,l,ty[l],evaluation.P['C'],key,evaluation.P['temperature_grid'])
   yield key,tx,l,ty[l],qt,ty,{},b,plan['shot'],dict(split='closed',regime='full_known',rho=0.,role=-1,seed=plan['seed'])
 if suite=='closed':return
 for role in d['roles']:
  if role_filter is not None and role['seed']!=role_filter:continue
  mapping={v:j for j,v in enumerate(role['known'])};kt=np.flatnonzero(np.isin(ty,role['known']));ut=np.flatnonzero(np.isin(ty,role['pool_unknown']))
  target=np.array([mapping.get(v,-1) for v in qt]);groups={'pool_seen':np.isin(qt,role['pool_unknown']),'query_only':np.isin(qt,role['query_unknown'])}
  for plan in [p for p in d['plans'] if p['shot']==4]:
   seed=plan['seed']
   if seed_filter is not None and seed!=seed_filter:continue
   support=np.array([ti[v] for v in plan['support'] if ty[ti[v]] in mapping]);ku=np.setdiff1d(kt,support)
   baseindex=np.sort(np.r_[support,ku]);look={v:i for i,v in enumerate(baseindex)};bl=np.array([look[v] for v in support]);by=np.array([mapping[ty[v]] for v in support])
   b=learner.base(tx[baseindex],qx,bl,by,evaluation.P['C'],f"open_r{role['seed']}_s{seed}_clean_full",evaluation.P['temperature_grid'])
   rng=np.random.default_rng(role['seed']+seed*10000);unknown=rng.permutation(ut);known=rng.permutation(ku);cap=min(len(known),len(unknown));retained=known[:cap]
   for regime,rhos in [('fixed_known',evaluation.P['contamination']),('fixed_total',evaluation.P['fixed_total_sensitivity'])]:
    for rho in rhos:
     if rho_filter is not None and not np.isclose(rho,rho_filter):continue
     if regime=='fixed_known':pool=np.r_[retained,unknown[:round(cap*rho/(1-rho))]]
     else:
      nk=round(cap*(1-rho));pool=np.r_[retained[:nk],unknown[:cap-nk]]
     index=np.sort(np.r_[support,pool]);look={v:i for i,v in enumerate(index)};l=np.array([look[v] for v in support]);truth=np.array([mapping.get(v,-1) for v in ty[index]])
     key=f"r{role['seed']}_s{seed}_{regime}_p{round(100*rho):02d}"
     with zipfile.ZipFile(REPO/'data/context_indices.zip') as z:saved=json.loads(z.read(f'{dataset}/indices/{key}_indices.json'))
     assert np.array_equal(index,saved['original_train_indices']), 'Locked context indices differ'
     assert not set(support)&set(pool)
     yield key,tx[index],l,truth[l],target,truth,groups,b,4,dict(split='open',regime=regime,rho=float(rho),role=role['seed'],seed=seed,known_cap=cap)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['tcmp','indian'],required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--ids',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--suite',choices=['closed','gradients','all'],default='all');p.add_argument('--seed',type=int);p.add_argument('--role',type=int);p.add_argument('--rho',type=float);p.add_argument('--max-contexts',type=int);a=p.parse_args()
 d=read(REPO/f'data/manifests/{a.dataset}.json.gz');ids=read(a.ids);assert len(ids)==len(set(ids)) and set(ids)==set(d['ids'])
 f=np.load(a.features,mmap_mode='r');assert f.shape==(len(ids),768) and np.isfinite(f).all();assert np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5)
 ix={v:i for i,v in enumerate(ids)};tx=np.asarray(f[[ix[r['id']] for r in d['train']]]);qx=np.asarray(f[[ix[r['id']] for r in d['query']]])
 o=a.output;o.mkdir(parents=True,exist_ok=True)
 for name in ['cache','results','artifacts']:(o/name).mkdir(exist_ok=True)
 cfg=read(REPO/'configs/experiment.json');evaluation.P=cfg;common.R=learner.R=o
 stamp=dict(dataset=a.dataset,features_sha256=sha(a.features),ids_sha256=sha(a.ids),config_sha256=sha(REPO/'configs/experiment.json'))
 if (o/'run_inputs.json').exists():assert read(o/'run_inputs.json')==stamp,'Use a new output directory for changed inputs'
 write(o/'run_inputs.json',stamp);n=0
 for key,x,l,y,t,truth,groups,b,shot,meta in contexts(d,tx,qx,a.dataset,a.suite,a.seed,a.role,a.rho):
  evaluation.evaluate_context(o,key,x,qx,l,y,t,truth,groups,b,shot,meta);n+=1
  if a.max_contexts and n>=a.max_contexts:break
 if n:evaluation.aggregate(o)
 print('Completed selected contexts:',n)
if __name__=='__main__':main()
