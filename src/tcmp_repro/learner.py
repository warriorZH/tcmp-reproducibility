"""No pool truth, query truth, class names or file paths enter this module."""
import numpy as np,time,warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.special import softmax
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from .common import R,write
import joblib
def fit(x,y,c,w=None):
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');h=LogisticRegression(C=c,max_iter=1500,tol=1e-5,solver='lbfgs').fit(x,y,sample_weight=w)
 if caught:print('FIT_WARNING',[(type(v.message).__name__,str(v.message)[:120]) for v in caught],flush=True)
 return h
def base(x,q,l,y,c,key,grid):
 path=R/'cache'/f'base_{key}.joblib';path.parent.mkdir(exist_ok=True)
 if path.exists():return joblib.load(path)
 h=fit(x[l],y,c);folds=list(StratifiedKFold(n_splits=4,shuffle=True,random_state=5521).split(x[l],y));oo=np.zeros((len(l),int(y.max()+1)))
 for a,b in folds:oo[b]=fit(x[l[a]],y[a],c).decision_function(x[l[b]])
 loss=[-np.log(np.maximum(softmax(oo/t,axis=1)[np.arange(len(y)),y],1e-30)).mean() for t in grid];temp=grid[int(np.argmin(loss))];hard=np.array([np.mean(oo[y==c].argmax(1)!=c) for c in range(y.max()+1)]);hard=np.lexsort((np.arange(len(hard)),-hard))[:max(1,len(hard)//5)]
 out=dict(head=h,train_logits=h.decision_function(x),query_logits=h.decision_function(q),oof_logits=oo,temperature=float(temp),folds=folds,hard=hard,loss=loss);joblib.dump(out,path);return out
def graph(x,q,k,key):
 a_path=R/'cache'/f'graph_{key}_k{k}.npz';n_path=R/'cache'/f'neighbors_{key}_k{k}.npz'
 if a_path.exists() and n_path.exists():
  z=np.load(n_path);return sparse.load_npz(a_path),z['ii'],z['ww']
 nn=NearestNeighbors(n_neighbors=k+1,metric='cosine',n_jobs=2).fit(x);dist,ind=nn.kneighbors(x);ind=ind[:,1:];v=np.maximum(1-dist[:,1:],0)**3;a=sparse.csr_matrix((v.ravel(),(np.repeat(np.arange(len(x)),k),ind.ravel())),shape=(len(x),len(x)));a=a.maximum(a.T);a=sparse.diags(1/np.maximum(np.asarray(a.sum(1)).ravel(),1e-12))@a;dd,ii=nn.kneighbors(q,n_neighbors=k);ww=np.maximum(1-dd,0)**3;ww/=np.maximum(ww.sum(1,keepdims=True),1e-12);sparse.save_npz(a_path,a);np.savez_compressed(n_path,ii=ii,ww=ww);return a,ii,ww
def propagate(g,l,y,nc,alpha):
 a,ii,ww=g;seed=np.zeros((a.shape[0],nc));seed[l,y]=1;f=seed.copy()
 for _ in range(100):
  z=alpha*(a@f)+(1-alpha)*seed;z[l]=seed[l]
  if np.max(np.abs(z-f))<1e-6:f=z;break
  f=z
 mass=f.sum(1);pu=f/np.maximum(mass[:,None],1e-30);pu[mass<=1e-30]=1/nc;p=(pu[ii]*ww[:,:,None]).sum(1);rawq=(f[ii]*ww[:,:,None]).sum(1);pm=p.sum(1);p/=np.maximum(pm[:,None],1e-30);p[pm<=1e-30]=1/nc;return dict(train=pu,query=p,mass=mass,raw_query=rawq,raw_train=f)
def selftrain(x,q,l,y,c,b,kind,quota,multiplier):
 h=b['head'];u=np.setdiff1d(np.arange(len(x)),l);hist=[];pseudo=[];weight=[];thresholds=[];temp=b['temperature']
 for iteration in range(3):
  p=softmax(h.decision_function(x[u])/temp,axis=1);conf=p.max(1);label=p.argmax(1)
  if kind=='quota':
   n=max(1,int(round(quota*len(u))));order=np.argsort(-conf,kind='stable');w=np.zeros(len(u));w[order[:n]]=1;threshold=float(conf[order[n-1]])
  elif kind=='fixed':
   threshold=float(quota);w=(conf>=threshold).astype(float)
  elif kind=='soft':
   mu=conf.mean();sd=max(conf.std(),.05);w=np.exp(-np.maximum(mu-conf,0)**2/(2*sd*sd));threshold=float(mu)
  else:raise ValueError(kind)
  pseudo.append(label.copy());weight.append(w.copy());thresholds.append(threshold);hist.append(dict(iteration=iteration,selected=int((w>0).sum()),weight_sum=float(w.sum()),effective_count=float(w.sum()**2/np.maximum((w*w).sum(),1e-30))))
  if w.sum()==0:break
  scale=multiplier*len(l)/max(w.sum(),1e-30);h=fit(np.r_[x[l],x[u]],np.r_[y,label],c,np.r_[np.ones(len(l)),scale*w])
 return dict(query=softmax(h.decision_function(q)/temp,axis=1),pool=softmax(h.decision_function(x[u])/temp,axis=1),pseudo=np.array(pseudo),weight=np.array(weight),thresholds=np.array(thresholds),history=hist,u=u)
def support_scores(x,z,l,y):
 dd,_=NearestNeighbors(n_neighbors=1,metric='cosine').fit(x[l]).kneighbors(z);means=np.array([x[l[y==j]].mean(0) for j in range(y.max()+1)]);basis=np.linalg.qr(means.T,mode='reduced')[0];return dict(support_distance=-dd[:,0],support_subspace=np.linalg.norm(z@basis,axis=1))
def scores(x,q,l,y,b,lp,g,alpha,perturb_seed):
 nc=int(y.max()+1);qscore=dict(lr_msp=softmax(b['query_logits'],axis=1).max(1),calibrated_lr_msp=softmax(b['query_logits']/b['temperature'],axis=1).max(1),lp_msp=lp['query'].max(1),lp_raw_max=lp['raw_query'].max(1),lp_mass=lp['raw_query'].sum(1),**support_scores(x,q,l,y));oof=dict(lr_msp=softmax(b['oof_logits'],axis=1).max(1),calibrated_lr_msp=softmax(b['oof_logits']/b['temperature'],axis=1).max(1));oo={name:np.zeros(len(l)) for name in ['lp_msp','lp_raw_max','lp_mass','support_distance','support_subspace']}
 for a,bb in b['folds']:
  pp=propagate(g,l[a],y[a],nc,alpha);held=l[bb];oo['lp_msp'][bb]=pp['train'][held].max(1);oo['lp_raw_max'][bb]=pp['raw_train'][held].max(1);oo['lp_mass'][bb]=pp['mass'][held]
  sc=support_scores(x,x[held],l[a],y[a])
  for key in ['support_distance','support_subspace']:oo[key][bb]=sc[key]
 oof.update(oo);cuts={name:float(np.quantile(z,.1)) for name,z in oof.items()};u=np.setdiff1d(np.arange(len(x)),l);votes=[]
 for j in range(3):
  rng=np.random.default_rng(perturb_seed+j);delete=[rng.choice(np.flatnonzero(y==c)) for c in range(nc)];take=np.setdiff1d(np.arange(len(l)),delete);pp=propagate(g,l[take],y[take],nc,alpha);votes.append(pp['train'][u].argmax(1)==lp['train'][u].argmax(1))
 stability=np.mean(votes,axis=0);poolscore=dict(confidence=softmax(b['train_logits'][u]/b['temperature'],axis=1).max(1),distance=support_scores(x,x[u],l,y)['support_distance'],mass=lp['mass'][u],stability=stability,lp_prediction=lp['train'][u].argmax(1),lr_prediction=b['train_logits'][u].argmax(1))
 return qscore,oof,cuts,u,poolscore
