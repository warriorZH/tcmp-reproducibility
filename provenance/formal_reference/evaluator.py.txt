"""Hidden truths enter this evaluator only after learner outputs exist."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
def oscr(score,t,pred):
 o=np.argsort(-score,kind='stable');ends=np.r_[np.flatnonzero(np.diff(score[o])!=0),len(o)-1];k=t[o]>=0;correct=k&(pred[o]==t[o]);return float(np.trapz(np.r_[0,np.cumsum(correct)[ends]/k.sum()],np.r_[0,np.cumsum(~k)[ends]/(~k).sum()]))
def classification(p,t,hard):
 pred=p.argmax(1);known=t>=0;cs=np.unique(t[known]);recall=[float(np.mean(pred[t==c]==c)) for c in cs];hm=known&np.isin(t,hard);return dict(accuracy=100*float(np.mean(pred[known]==t[known])),macro_recall=100*float(np.mean(recall)),per_class_recall=recall,hard_accuracy=100*float(np.mean(pred[hm]==t[hm])) if hm.any() else None)
def pseudo_metrics(r,t):
 yu=t[r['u']];out=[]
 for label,w,threshold in zip(r['pseudo'],r['weight'],r['thresholds']):
  sel=w>0;known=yu>=0;out.append(dict(selected=int(sel.sum()),actual_training_precision=float(np.mean(label[sel]==yu[sel])) if sel.any() else None,weighted_training_precision=float(np.sum(w*(label==yu))/w.sum()) if w.sum()>0 else None,unknown_nonzero_fraction=float(np.mean(sel[~known])) if (~known).any() else 0,unknown_weight_fraction=float(w[~known].sum()/w.sum()) if w.sum()>0 else 0,known_selected_fraction=float(np.mean(sel[known])),effective_count=float(w.sum()**2/max((w*w).sum(),1e-30)),threshold=float(threshold)))
 return out
def reliability(score,pred,t,cut,groups):
 k=t>=0;accept=score>=cut;threshold=float(np.quantile(score[k],.1));diagnostic=score>=threshold;out=dict(auroc=float(roc_auc_score(k,score)),aupr_unknown=float(average_precision_score(~k,-score)),oscr=oscr(score,t,pred),deployment_threshold=float(cut),known_coverage=float(np.mean(accept[k])),unknown_acceptance=float(np.mean(accept[~k])),known_selective_risk=float(np.mean(pred[k&accept]!=t[k&accept])) if (k&accept).any() else None,diagnostic_unknown_at_known90=float(np.mean(diagnostic[~k])),diagnostic_known_coverage=float(np.mean(diagnostic[k])))
 for name,mask in groups.items():out[name+'_unknown_acceptance']=float(np.mean(accept[mask]));out[name+'_diagnostic_unknown_at_known90']=float(np.mean(diagnostic[mask]))
 return out
def conditional_evidence(poolscore,truth):
 conf=poolscore['confidence'];distance=poolscore['distance'];known=truth>=0;correct=known&(poolscore['lp_prediction']==truth);bins=[]
 for z in [conf,distance]:bins.append(np.minimum(np.searchsorted(np.quantile(z,[.2,.4,.6,.8]),z,side='right'),4))
 group=bins[0]*5+bins[1];result={}
 for targetname,target in [('known_vs_unknown',known),('lp_correct_vs_other',correct)]:
  result[targetname]={}
  for evidence in ['mass','stability']:
   pairs=[]
   for i in range(25):
    m=group==i
    if m.sum()>=20 and len(np.unique(target[m]))==2:pairs.append((int(m.sum()),float(roc_auc_score(target[m],poolscore[evidence][m]))))
   result[targetname][evidence]=dict(conditional_auroc=float(np.average([a for n,a in pairs],weights=[n for n,a in pairs])) if pairs else None,bins_used=len(pairs),samples=sum(n for n,a in pairs),interpretation='diagnostic association only, no trained selector or causal claim')
 return result
