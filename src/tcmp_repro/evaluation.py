"""Original evaluation kernels; portable imports and configuration injection."""
import os,time,pathlib
import numpy as np
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from . import learner
from .io import read,write
P={}
ROOT=pathlib.Path('.')
def status(*args,**kwargs):pass
def pool_group_score(score,pred,t,cut,groups):
 k=t>=0;a=np.ones(len(t),bool) if score is None else score>=cut;acc=k&a;unknown=(~k)&a;known_classes=np.unique(t[k]);per=[float(np.mean(pred[t==c]==c)) for c in known_classes];risks=[float(np.mean(pred[a&(t==c)]!=c)) for c in known_classes if (a&(t==c)).any()]
 v=dict(accuracy=100*float(np.mean(pred[k]==t[k])),macro_recall=100*float(np.mean(per)),worst_class_recall=100*min(per),bottom_decile=100*float(np.mean(sorted(per)[:max(1,len(per)//10)])),known_coverage=float(a[k].mean()),accepted_known_risk=float(np.mean(pred[acc]!=t[acc])) if acc.any() else None,macro_accepted_known_risk=float(np.mean(risks)) if risks else None,classes_with_no_accepted_known=int(sum(not(a&(t==c)).any() for c in known_classes)),overall_accepted_error=float(((pred[acc]!=t[acc]).sum()+unknown.sum())/a.sum()) if a.any() else None)
 v['unknown_acceptance']=float(a[~k].mean()) if (~k).any() else None
 if score is not None and (~k).any():
  order=np.argsort(-score,kind='stable');ends=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(t)-1];kk=k[order];cc=kk&(pred[order]==t[order]);x=np.r_[0,np.cumsum(~kk)[ends]/(~kk).sum()];y=np.r_[0,np.cumsum(cc)[ends]/kk.sum()];v['oscr']=float(np.sum(np.diff(x)*(y[1:]+y[:-1])/2))
 else:v['oscr']=None
 for name,g in groups.items():v[name+'_unknown_acceptance']=float(a[g].mean()) if g.any() else None
 return v

def one_step(g,l,y,nc,alpha):
 a,ii,ww=g;seed=np.zeros((a.shape[0],nc));seed[l,y]=1;f=alpha*(a@seed)+(1-alpha)*seed;f[l]=seed[l];m=f.sum(1);u=f/np.maximum(m[:,None],1e-30);u[m<=1e-30]=1/nc;q=(u[ii]*ww[:,:,None]).sum(1);mass=q.sum(1);q/=np.maximum(mass[:,None],1e-30);q[mass<=1e-30]=1/nc;return q

def support_controls(x,q,l,y,nc):
 xs=np.asarray(x[l],np.float64);qq=np.asarray(q,np.float64);nn=NearestNeighbors(n_neighbors=min(10,len(l)),metric='cosine').fit(xs);dist,ind=nn.kneighbors(qq);w=np.maximum(1-dist,0)**3;w/=np.maximum(w.sum(1,keepdims=True),1e-30);p=(np.eye(nc)[y[ind]]*w[:,:,None]).sum(1);zero=p.sum(1)<=1e-30;p[zero]=1/nc
 centers=np.array([xs[y==c].mean(0) for c in range(nc)]);centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-30);ncm=np.eye(nc)[(qq@centers.T).argmax(1)];return p,ncm

def evaluate_context(O,key,x,q,l,y,t,pool_truth,groups,b,shot,meta):
 target=O/'results'/f'{key}.json'
 if target.exists():assert (O/'results'/f'{key}.npz').exists();return
 begun=time.time();nc=int(y.max()+1);assert set(y)==set(range(nc));assert len(set(l))==len(l);assert x.shape[1]==q.shape[1]==768
 state=dict(dataset=O.name,current=key,completed=len(list((O/'results').glob('*.json'))));status(O.name,'classifying',**state)
 # Learners receive only features and support labels; hidden truths used below.
 b=dict(b);b['train_logits']=b['head'].decision_function(x);b['query_logits']=b['head'].decision_function(q)
 lr=softmax(b['query_logits'],axis=1);score_lr=lr.max(1);cut_lr=float(np.quantile(softmax(b['oof_logits'],axis=1).max(1),.1))
 timers={};at=time.time();g=learner.graph(x,q,P['graph_k'],key);timers['graph_build_seconds']=time.time()-at
 at=time.time();lp=learner.propagate(g,l,y,nc,P['alpha'][str(shot)]);timers['lp_seconds']=time.time()-at
 at=time.time();one=one_step(g,l,y,nc,P['alpha'][str(shot)]);timers['one_step_seconds']=time.time()-at
 knn,ncm=support_controls(x,q,l,y,nc);probs={'lr':lr,'lp':lp['query'],'one_step_lp':one,'support_knn':knn,'ncm':ncm};pseudo={};arrays={};u=np.setdiff1d(np.arange(len(x)),l)
 for method in ['fixed','soft']:
  at=time.time();out=learner.selftrain(x,q,l,y,P['C'],b,method,P['fixed_threshold'][str(shot)],P['soft_mass']);timers[method+'_seconds']=time.time()-at;probs[method]=out['query'];rows=[]
  for label,w in zip(out['pseudo'],out['weight']):
   truth=pool_truth[out['u']];sel=w>0;rows.append(dict(selected=int(sel.sum()),precision=float(np.mean(label[sel]==truth[sel])) if sel.any() else None,weighted_precision=float(np.sum(w*(label==truth))/w.sum()) if w.sum() else None,unknown_weight_fraction=float(w[truth<0].sum()/w.sum()) if w.sum() else None,known_selected_fraction=float(sel[truth>=0].mean()) if (truth>=0).any() else None))
  pseudo[method]=rows;arrays['pseudo_'+method]=out['pseudo'];arrays['weight_'+method]=out['weight'];arrays['thresholds_'+method]=out['thresholds']
 score_lp=lp['query'].max(1);oo_lp=np.zeros(len(l))
 for a,bb in b['folds']:oo_lp[bb]=learner.propagate(g,l[a],y[a],nc,P['alpha'][str(shot)])['train'][l[bb]].max(1)
 cuts={'lr_msp':cut_lr,'lp_msp':float(np.quantile(oo_lp,.1))};scores={'lr_msp':score_lr,'lp_msp':score_lp};metrics={};curves={}
 for method,prob in probs.items():
  assert prob.shape==(len(t),nc) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
  pred=prob.argmax(1);arrays['pred_'+method]=pred
  metrics[method]={'none':pool_group_score(None,pred,t,None,groups)}
  for gate,score in scores.items():
   metrics[method][gate]=pool_group_score(score,pred,t,cuts[gate],groups);points=[]
   for cov in np.arange(.1,1.01,.1):
    cut=float(np.quantile(score[t>=0],max(0,1-cov)));m=pool_group_score(score,pred,t,cut,groups);points.append({k:m[k] for k in ['known_coverage','accepted_known_risk','unknown_acceptance']})
   curves[method+'__'+gate]=points
 # Fixed supervised gate must be identical across classifiers at every draw.
 assert len({metrics[m]['lr_msp']['known_coverage'] for m in probs})==1
 arrays.update(query_truth=t,pool_truth=pool_truth,support=l,support_labels=y,u=u,score_lr_msp=score_lr,score_lp_msp=score_lp,oof_lr_msp=softmax(b['oof_logits'],axis=1).max(1),oof_lp_msp=oo_lp)
 (O/'results').mkdir(exist_ok=True);tmp=O/'results'/f'{key}.tmp.npz';np.savez_compressed(tmp,**arrays);os.replace(tmp,O/'results'/f'{key}.npz')
 record=dict(key=key,**meta,shot=shot,n_train=len(x),n_support=len(l),n_query=len(q),n_pool=len(u),unknown_pool_count=int((pool_truth[u]<0).sum()),actual_unknown_fraction=float(np.mean(pool_truth[u]<0)),metrics=metrics,pseudo=pseudo,cuts=cuts,curves=curves,timing=timers,seconds=time.time()-begun,learner_hidden_labels_read=False)
 write(target,record);print('SUPPLEMENT_RESULT',O.name,key,'LP',round(metrics['lp']['none']['accuracy'],4),'fixed',round(metrics['fixed']['none']['accuracy'],4),'seconds',round(time.time()-begun,1),flush=True)

def aggregate(O):
 rs=[read(f) for f in sorted((O/'results').glob('*.json'))];summary=dict(contexts=len(rs),groups={},paired={},limitations=P['interpretation'])
 for split,regime,rho,shot in sorted({(r['split'],r['regime'],r['rho'],r['shot']) for r in rs}):
  group=[r for r in rs if (r['split'],r['regime'],r['rho'],r['shot'])==(split,regime,rho,shot)];key=f'{split}__{regime}__{rho:.2f}__{shot}shot';summary['groups'][key]={'configurations':len(group),'methods':{}}
  for m in P['classifiers']:
   summary['groups'][key]['methods'][m]={}
   for gate in P['gates']:
    vals={}
    for metric in group[0]['metrics'][m][gate]:
     v=[r['metrics'][m][gate][metric] for r in group if r['metrics'][m][gate][metric] is not None]
     vals[metric]=dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else None) if v else None
    summary['groups'][key]['methods'][m][gate]=vals
  if split=='open':
   roles=sorted({r['role'] for r in group});seeds=sorted({r['seed'] for r in group});rng=np.random.default_rng(P['bootstrap']['seed']);bi=rng.integers(0,len(seeds),size=(P['bootstrap']['draws'],len(seeds)))
   for baseline in ['fixed','soft','lr','one_step_lp','support_knn','ncm']:
    a=np.array([[next(r for r in group if r['role']==role and r['seed']==seed)['metrics']['lp']['none']['accuracy']-next(r for r in group if r['role']==role and r['seed']==seed)['metrics'][baseline]['none']['accuracy'] for seed in seeds] for role in roles]);means=a[:,bi].mean(axis=(0,2));summary['paired'][key+'__lp_minus_'+baseline]=dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),positive=int((a>0).sum()),configurations=a.size,seed_blocks=len(seeds),conditional_on_fixed_roles_and_images=True)
 # Area under accuracy-loss curve, per draw: baseline = same retained clean pool.
 areas={}
 for m in ['lr','fixed','soft','lp']:
  vals=[]
  for role,seed in sorted({(r['role'],r['seed']) for r in rs if r['split']=='open'}):
   rows=sorted([r for r in rs if r['split']=='open' and r['regime']=='fixed_known' and (r['role'],r['seed'])==(role,seed)],key=lambda r:r['rho'])
   if len(rows)==6:
    acc=np.array([r['metrics'][m]['none']['accuracy'] for r in rows]);x=np.array([r['rho'] for r in rows]);loss=acc[0]-acc;vals.append(float(np.sum(np.diff(x)*(loss[1:]+loss[:-1])/2)/.5))
  if vals:areas[m]=dict(mean_integrated_loss_points=float(np.mean(vals)),per_configuration=vals)
 summary['contamination_integrated_loss']=areas;write(O/'artifacts/summary.json',summary);return summary

