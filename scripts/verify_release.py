"""Offline verification of splits, locked indices and released result records."""
from pathlib import Path
import gzip,json,zipfile,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):
 if str(p).endswith('.gz'):
  with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
 return json.loads(Path(p).read_text(encoding='utf-8'))
checks={}
with zipfile.ZipFile(R/'data/context_indices.zip') as z:
 for dataset,nclasses,ntrain,nquery in [('tcmp',240,29050,11993),('indian',60,3955,988)]:
  d=read(R/f'data/manifests/{dataset}.json.gz');assert (len(d['classes']),len(d['train']),len(d['query']))==(nclasses,ntrain,nquery)
  cname=lambda r:r.get('class',r.get('class_name'));ci={c:i for i,c in enumerate(d['classes'])};ty=np.array([ci[cname(r)] for r in d['train']]);ix={r['id']:i for i,r in enumerate(d['train'])}
  assert not {r['id'] for r in d['train']}&{r['id'] for r in d['query']}
  tg={r['group'] for r in d['train']};qg={r['group'] for r in d['query']};assert not tg&qg
  for seed in sorted({p['seed'] for p in d['plans']}):
   ps={p['shot']:p['support'] for p in d['plans'] if p['seed']==seed};assert set(ps[4])<=set(ps[16])
   for shot,ids in ps.items():
    assert len(ids)==len(set(ids))==shot*nclasses
    assert set(ids)<=set(ix)
    assert np.all(np.bincount(ty[[ix[v] for v in ids]],minlength=nclasses)==shot)
  verified=0
  for role in d['roles']:
   assert sorted(role['known']+role['pool_unknown']+role['query_unknown'])==list(range(nclasses))
   kt=np.flatnonzero(np.isin(ty,role['known']));ut=np.flatnonzero(np.isin(ty,role['pool_unknown']))
   for plan in [p for p in d['plans'] if p['shot']==4]:
    seed=plan['seed'];support=np.array([ix[v] for v in plan['support'] if ty[ix[v]] in role['known']]);ku=np.setdiff1d(kt,support)
    rng=np.random.default_rng(role['seed']+seed*10000);unknown=rng.permutation(ut);known=rng.permutation(ku);cap=min(len(known),len(unknown));retained=known[:cap]
    for regime,rhos in [('fixed_known',[0,.1,.2,.3,.4,.5]),('fixed_total',[.3,.5])]:
     for rho in rhos:
      if regime=='fixed_known':pool=np.r_[retained,unknown[:round(cap*rho/(1-rho))]]
      else:
       nk=round(cap*(1-rho));pool=np.r_[retained[:nk],unknown[:cap-nk]]
      index=np.sort(np.r_[support,pool]);key=f"{dataset}/indices/r{role['seed']}_s{seed}_{regime}_p{round(100*rho):02d}_indices.json";stored=json.loads(z.read(key))
      assert np.array_equal(index,stored['original_train_indices']),key
      assert len(set(pool))==len(pool) and not set(pool)&set(support);verified+=1
  results=list((R/f'results/reference/{dataset}/results').glob('*.json'));assert len(results)==(168 if dataset=='tcmp' else 182)
  for path in results:
   r=read(path);assert len(r['metrics'])==7
   assert len({m['lr_msp']['known_coverage'] for m in r['metrics'].values()})==1
   assert len({m['lr_msp']['unknown_acceptance'] for m in r['metrics'].values()})==1
  checks[dataset]=dict(classes=nclasses,train=ntrain,query=nquery,index_contexts_verified=verified,result_contexts=len(results),support_nested=True,train_query_groups_disjoint=True)
formal=list((R/'results/reference/formal/results').glob('*.json'));assert len(formal)==476
checks['formal_result_records']=len(formal)
if (R/'SHA256SUMS').exists():
 for line in (R/'SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
checks['passed']=True
out=R/'outputs/verification';out.mkdir(parents=True,exist_ok=True);(out/'manifest_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))
