"""Recompute tables, paired intervals and figures from released per-context JSONs."""
from pathlib import Path
import json,csv,hashlib,zipfile,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--input',default='results/reference')
parser.add_argument('--output',default='outputs/analysis')
args=parser.parse_args();P=Path(args.input)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(args.output);O.mkdir(parents=True,exist_ok=True);(O/'figures').mkdir(exist_ok=True)
D={j:[json.loads(p.read_text(encoding='utf-8')) for p in sorted((P/j/'results').glob('*.json'))] for j in ['tcmp','indian']}
S={j:json.loads((P/j/'summary.json' if (P/j/'summary.json').exists() else P/j/'artifacts/summary.json').read_text(encoding='utf-8')) for j in D}
assert len(D['tcmp'])==168 and len(D['indian'])==182
SEED=202610164;N=20000;RHOS=[0,.1,.2,.3,.4,.5];methods=['lr','fixed','soft','lp','one_step_lp','support_knn','ncm']
names={'lr':'Supervised LR','fixed':'Fixed self-training','soft':'Soft self-training','lp':'Iterative LP','one_step_lp':'One-step LP','support_knn':'Support 10NN','ncm':'NCM'}
colors={'lr':'#687783','fixed':'#d57b2a','soft':'#a96d9f','lp':'#2475b4','one_step_lp':'#8c7c67','support_knn':'#899b72','ncm':'#74a8a3'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def select(j,rho=0,regime='fixed_known',split='open',shot=4):return sorted([r for r in D[j] if r['split']==split and r['regime']==regime and r['rho']==rho and r['shot']==shot],key=lambda r:(r['role'],r['seed']))
def val(r,m='lp',gate='none',metric='accuracy'):return r['metrics'][m][gate][metric]
def boot(a):
 a=np.asarray(a,float)
 if a.ndim==1:a=a[None,:]
 rng=np.random.default_rng(SEED);idx=rng.integers(0,a.shape[1],size=(N,a.shape[1]));v=a[:,idx].mean(axis=(0,2));return dict(mean=float(a.mean()),ci95=np.quantile(v,[.025,.975]).tolist(),positive=int((a>0).sum()),n=int(a.size),seed_blocks=a.shape[1])
def paired(rows,fun):
 roles=sorted({r['role'] for r in rows});seeds=sorted({r['seed'] for r in rows});a=np.array([[fun(next(r for r in rows if r['role']==role and r['seed']==seed)) for seed in seeds] for role in roles]);return boot(a)
def avg(j,rho,m,gate='none',metric='accuracy',regime='fixed_known'):return float(np.mean([val(r,m,gate,metric) for r in select(j,rho,regime)]))
def save(fig,name):
 fig.savefig(O/'figures'/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(O/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def pct(x):return f'{x:.2f}'
def interval(v):return f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
stats={'protocol':'post-formal supplement; conditional intervals; no multiple-comparison significance claim','datasets':{}}
long=[]
for j,rs in D.items():
 for r in rs:
  assert len(r['metrics'])==7
  for m in methods:
   for gate in ['none','lr_msp','lp_msp']:
    long.append(dict(dataset=j,key=r['key'],split=r['split'],regime=r['regime'],rho=r['rho'],role=r['role'],seed=r['seed'],shot=r['shot'],method=m,gate=gate,**r['metrics'][m][gate]))
 with (O/(j+'_context_metrics.csv')).open('w',newline='',encoding='utf-8-sig') as f:
  keys=list(dict.fromkeys(k for row in long if row['dataset']==j for k in row));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(row for row in long if row['dataset']==j)
 st=stats['datasets'][j]={'gradient':{},'fixed_total':{},'gate_comparisons':{},'tail_50':{},'pseudo_50':{}}
 clean={(r['role'],r['seed']):r for r in select(j)}
 for rho in RHOS:
  rs=select(j,rho);st['gradient'][str(rho)]={}
  for m in methods:
   st['gradient'][str(rho)][m]=dict(accuracy=avg(j,rho,m),loss_from_clean=paired(rs,lambda r:val(clean[(r['role'],r['seed'])],m)-val(r,m)))
  st['gradient'][str(rho)]['lp_minus_fixed']=paired(rs,lambda r:val(r)-val(r,'fixed'))
 for rho in [.3,.5]:st['fixed_total'][str(rho)]=paired(select(j,rho,'fixed_total'),lambda r:val(r)-val(r,'fixed'))
 # Paired reduction in integrated loss (positive favors LP).
 st['integrated_loss_reduction_vs_fixed']=boot((np.array(S[j]['contamination_integrated_loss']['fixed']['per_configuration'])-np.array(S[j]['contamination_integrated_loss']['lp']['per_configuration'])).reshape(3,7))
 st['lp_minus_one_step_50']=paired(select(j,.5),lambda r:val(r)-val(r,'one_step_lp'))
 st['lp_minus_fixed_50_shared_gate_risk_points']=paired(select(j,.5),lambda r:100*(val(r,'lp','lr_msp','accepted_known_risk')-val(r,'fixed','lr_msp','accepted_known_risk')))
 for rho in [0,.3,.5]:
  st['gate_comparisons'][str(rho)]={g:{k:avg(j,rho,'lp',g,k) for k in ['known_coverage','unknown_acceptance','accepted_known_risk','macro_accepted_known_risk','overall_accepted_error','oscr','pool_seen_unknown_acceptance','query_only_unknown_acceptance']} for g in ['lr_msp','lp_msp']}
 for m in ['lr','fixed','soft','lp']:st['tail_50'][m]={k:avg(j,.5,m,'none',k) for k in ['macro_recall','worst_class_recall','bottom_decile']}
 for m in ['fixed','soft']:
  rows=[r['pseudo'][m][-1] for r in select(j,.5)];st['pseudo_50'][m]={k:float(np.mean([r[k] for r in rows])) for k in ['precision','weighted_precision','unknown_weight_fraction','known_selected_fraction']}
 if j=='indian':
  st['closed']={}
  for shot in [4,16]:
   rows=select(j,0,'full_known','closed',shot);st['closed'][str(shot)]={m:float(np.mean([val(r,m) for r in rows])) for m in methods};st['closed'][str(shot)]['lp_minus_fixed']=paired(rows,lambda r:val(r)-val(r,'fixed'));st['closed'][str(shot)]['lp_minus_lr']=paired(rows,lambda r:val(r)-val(r,'lr'))
(O/'derived_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')

fig,axes=plt.subplots(2,2,figsize=(12.8,8),sharex=True)
for col,j in enumerate(D):
 clean={(r['role'],r['seed']):r for r in select(j)}
 for m in ['lr','fixed','soft','lp']:
  points=[paired(select(j,r),lambda v:val(v,m)) for r in RHOS];loss=[paired(select(j,r),lambda v:val(clean[(v['role'],v['seed'])],m)-val(v,m)) for r in RHOS]
  for row,vs in [(0,points),(1,loss)]:
   ax=axes[row,col];means=[v['mean'] for v in vs];lo=[v['ci95'][0] for v in vs];hi=[v['ci95'][1] for v in vs];ax.plot(np.array(RHOS)*100,means,'o-',label=names[m],color=colors[m]);ax.fill_between(np.array(RHOS)*100,lo,hi,color=colors[m],alpha=.12);ax.grid(alpha=.15)
 axes[0,col].set_title('TCMP-300 (120 known)' if j=='tcmp' else 'Indian leaves (30 known)');axes[0,col].set_ylabel('Known accuracy (%)');axes[1,col].set_ylabel('Loss from same clean pool (points)');axes[1,col].set_xlabel('Unknown fraction in unlabeled pool (%)');axes[0,col].legend(fontsize=9);axes[1,col].axhline(0,color='#999',lw=.8)
fig.suptitle('Fixed known subset; nested unknown additions\nBands: conditional 95% shared-seed block intervals',fontsize=13);fig.tight_layout();save(fig,'01_contamination_gradients')

fig,axes=plt.subplots(1,2,figsize=(12.8,5))
ab=['lr','support_knn','ncm','one_step_lp','lp']
for ax,j in zip(axes,D):
 vs=[paired(select(j,.5),lambda r:val(r,m)) for m in ab];means=[v['mean'] for v in vs];err=np.array([[v['mean']-v['ci95'][0],v['ci95'][1]-v['mean']] for v in vs]).T;ax.bar(range(5),means,color=[colors[m] for m in ab],yerr=err,capsize=3);ax.set_xticks(range(5),['LR','Support\n10NN','NCM','One-step\nLP','Iterative\nLP']);ax.set_ylim(80,100);ax.set_ylabel('Known accuracy (%)');ax.set_title(j.upper()+' / 4-shot / 50% unknown');ax.grid(axis='y',alpha=.15)
 for i,v in enumerate(means):ax.text(i,v+1.25,f'{v:.2f}',ha='center',fontsize=10)
fig.tight_layout();save(fig,'02_component_ablation')

fig,axes=plt.subplots(2,3,figsize=(14,7))
for row,j in enumerate(D):
 for col,(metric,title) in enumerate([('known_coverage','Known coverage'),('unknown_acceptance','Unknown acceptance'),('accepted_known_risk','Accepted-known error')]):
  ax=axes[row,col]
  for gate,color,label in [('lr_msp','#d57b2a','Supervised MSP gate'),('lp_msp','#2475b4','LP MSP gate')]:ax.plot(np.array(RHOS)*100,[100*avg(j,r,'lp',gate,metric) for r in RHOS],'o-',color=color,label=label)
  ax.set_title(j.upper()+' / '+title);ax.set_ylabel('%');ax.set_xlabel('Unknown pool fraction (%)');ax.grid(alpha=.15);ax.legend(fontsize=9)
fig.suptitle('Same iterative-LP classifier; support-derived thresholds for each score',fontsize=13);fig.tight_layout();save(fig,'03_rejection_tradeoffs')

fig,axes=plt.subplots(1,2,figsize=(12.8,5))
for ax,shot in zip(axes,[4,16]):
 ms=['lr','fixed','soft','lp'];rows=select('indian',0,'full_known','closed',shot);means=[np.mean([val(r,m) for r in rows]) for m in ms];sd=[np.std([val(r,m) for r in rows],ddof=1) for m in ms];ax.bar(range(4),means,color=[colors[m] for m in ms],yerr=sd,capsize=3);ax.set_xticks(range(4),['LR','Fixed ST','Soft ST','LP']);ax.set_ylim(88,100);ax.set_ylabel('Closed-set accuracy (%)');ax.set_title(f'Indian leaves: 60 classes, {shot}-shot');ax.grid(axis='y',alpha=.15)
 for i,v in enumerate(means):ax.text(i,v+sd[i]+.25,f'{v:.2f}',ha='center',fontsize=10)
fig.suptitle('Seven support draws; error bars are sample standard deviations',fontsize=13);fig.tight_layout();save(fig,'04_external_closed_set')

fig,ax=plt.subplots(figsize=(9,4.4));labels=[]
for i,(j,rho) in enumerate([('tcmp',.3),('tcmp',.5),('indian',.3),('indian',.5)]):
 v=stats['datasets'][j]['fixed_total'][str(rho)];ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],fmt='o',color='#2475b4',capsize=4);labels.append(f'{j.upper()} / {int(rho*100)}% unknown')
ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='#999',ls='--');ax.set_xlabel('LP minus fixed self-training accuracy (points)');ax.set_title('Fixed-total-pool sensitivity: paired conditional 95% intervals');ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'05_fixed_total_control')

fig,axes=plt.subplots(1,2,figsize=(12.8,4.5))
for ax,j in zip(axes,D):
 for m in ['fixed','soft']:
  ax.plot(np.array(RHOS)*100,[100*np.mean([r['pseudo'][m][-1]['weighted_precision'] for r in select(j,rho)]) for rho in RHOS],'o-',color=colors[m],label=names[m])
 ax.set_title(j.upper()+' / final self-training round');ax.set_xlabel('Unknown pool fraction (%)');ax.set_ylabel('Weighted pseudo-label precision (%)');ax.set_ylim(40,100);ax.grid(alpha=.15);ax.legend(fontsize=9)
fig.tight_layout();save(fig,'06_pseudo_label_contamination')


lines=['# Saved-result analysis', '', 'All released dual-source contexts are included. Intervals use shared support-seed blocks conditional on fixed images and class roles.', '', '| Dataset | Unknown fraction | LR | Fixed | Soft | Iterative LP | LP minus fixed (95% interval) |', '|---|---:|---:|---:|---:|---:|---|']
for j in D:
 for rho in RHOS:
  est=stats['datasets'][j]['gradient'][str(rho)]['lp_minus_fixed']
  lines.append('|'+j+'|'+str(rho)+'|'+'|'.join(f'{avg(j,rho,m):.3f}' for m in ['lr','fixed','soft','lp'])+'|'+interval(est)+'|')
lines+=['', 'Accuracy is in percent; differences are percentage points.', '', '![Contamination gradients](figures/01_contamination_gradients.png)', '', '![Component controls](figures/02_component_ablation.png)', '', '![Acceptance tradeoffs](figures/03_rejection_tradeoffs.png)', '', 'The JSON and CSV files contain full fixed-total, tail-class, sixteen-shot and pseudo-label results. The scientific scope and implementation are documented in docs/PROTOCOL.md.']
(O/'analysis_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('ANALYSIS_COMPLETE',O)
