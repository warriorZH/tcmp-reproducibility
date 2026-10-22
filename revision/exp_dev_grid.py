"""W4 / Q2 (development classes): repeat the k x alpha (LP) and threshold (fixed
self-training) grid on the 60 TCMP development classes, which are disjoint from
the 240 evaluation classes. This shows which values a user would select from a
small development split, and whether the transferred defaults lie on the plateau.

Protocol mirrors the evaluation protocol at a smaller scale: 30 known / 20
pool-unknown / 10 query-only classes per role (3 roles), 7 nested 4-/16-shot
support draws, fixed-known pools at 0% and 50% contamination, and closed
4-/16-shot pools. Development images were not re-audited for near duplicates.
"""
import argparse, itertools
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import rcommon as rc
from tcmp_repro import learner
from scipy.special import softmax

K = [5, 10, 15, 20, 30]; A = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]; TH = [0.5, 0.7, 0.8, 0.9, 0.95]
ROLES = [70001, 70002, 70003]; SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007]


def contexts(feat_dir):
    f = np.load(Path(feat_dir) / 'features.npy'); recs = rc.read(Path(feat_dir) / 'records.json')
    classes = sorted({r['cls'] for r in recs}); ci = {c: i for i, c in enumerate(classes)}
    tr = [i for i, r in enumerate(recs) if r['split'] == 'train']; qu = [i for i, r in enumerate(recs) if r['split'] == 'query']
    tx, qx = f[tr], f[qu]; ty = np.array([ci[recs[i]['cls']] for i in tr]); qt = np.array([ci[recs[i]['cls']] for i in qu])
    sup16 = {}
    for s in SEEDS:
        rng = np.random.default_rng(s); sup16[s] = {c: rng.permutation(np.flatnonzero(ty == c))[:16] for c in range(len(classes))}
    for s in SEEDS:
        for shot in [4, 16]:
            l = np.sort(np.concatenate([sup16[s][c][:shot] for c in range(len(classes))]))
            yield dict(key=f'dev_closed_s{s}_n{shot}', x=tx, q=qx, l=l, y=ty[l], t=qt, truth=ty, groups={}, shot=shot,
                       meta=dict(split='closed', rho=0.0, role=-1, seed=s))
    for role in ROLES:
        perm = np.random.default_rng(role).permutation(len(classes)); known, pu, qo = perm[:30], perm[30:50], perm[50:]
        mp = {v: j for j, v in enumerate(known)}; t = np.array([mp.get(v, -1) for v in qt]); groups = {'pool_seen': np.isin(qt, pu), 'query_only': np.isin(qt, qo)}
        for s in SEEDS:
            support = np.concatenate([sup16[s][c][:4] for c in known]); kt = np.flatnonzero(np.isin(ty, known)); ut = np.flatnonzero(np.isin(ty, pu))
            ku = np.setdiff1d(kt, support); rng = np.random.default_rng(role + s * 10000); unknown = rng.permutation(ut); kn = rng.permutation(ku); cap = min(len(kn), len(unknown))
            for rho in [0.0, 0.5]:
                pool = np.r_[kn[:cap], unknown[:round(cap * rho / (1 - rho))]]; index = np.sort(np.r_[support, pool]); look = {v: i for i, v in enumerate(index)}
                l = np.array([look[v] for v in support]); truth = np.array([mp.get(v, -1) for v in ty[index]])
                yield dict(key=f'dev_r{role}_s{s}_p{round(100*rho):02d}', x=tx[index], q=qx, l=l, y=truth[l], t=t, truth=truth, groups=groups, shot=4,
                           meta=dict(split='open', rho=rho, role=role, seed=s, known_cap=int(cap)))


def run_context(ctx, res):
    target = res / f"{ctx['key']}.json"
    if target.exists():
        return rc.read(target)
    tm = rc.timer(); x, q, l, y, t, groups = ctx['x'], ctx['q'], ctx['l'], ctx['y'], ctx['t'], ctx['groups']; nc = int(y.max() + 1)
    b = learner.base(x, q, l, y, rc.CFG['C'], ctx['key'], rc.CFG['temperature_grid'])
    nb = rc.build_neighbors(x, q, max(K)); acc = {}
    acc['lr'] = rc.metrics(None, b['query_logits'].argmax(1), t, None, groups)['accuracy']
    for k in K:
        g = rc.graph_from(nb, k, len(x))
        for al in A:
            acc[f'lp_k{k}_a{al}'] = rc.metrics(None, learner.propagate(g, l, y, nc, al)['query'].argmax(1), t, None, groups)['accuracy']
    for th in TH:
        acc[f'fixed_t{th}'] = rc.metrics(None, learner.selftrain(x, q, l, y, rc.CFG['C'], b, 'fixed', th, 1)['query'].argmax(1), t, None, groups)['accuracy']
    acc['soft'] = rc.metrics(None, learner.selftrain(x, q, l, y, rc.CFG['C'], b, 'soft', 0, 1)['query'].argmax(1), t, None, groups)['accuracy']
    rec = dict(key=ctx['key'], shot=ctx['shot'], **ctx['meta'], acc=acc, n_train=len(x), seconds=tm()); rc.write(target, rec)
    rc.log('DEV', ctx['key'], 'LR %.2f LP(k10,a0.9) %.2f fixed0.9 %.2f (%.0fs)' % (acc['lr'], acc['lp_k10_a0.9'], acc['fixed_t0.9'], tm()))
    return rec


def summarise(rows):
    out = {}
    for split, shot, rho in sorted({(r['split'], r['shot'], r['rho']) for r in rows}):
        rr = [r for r in rows if (r['split'], r['shot'], r['rho']) == (split, shot, rho)]
        tab = {m: float(np.mean([r['acc'][m] for r in rr])) for m in rr[0]['acc']}
        lp = {m: v for m, v in tab.items() if m.startswith('lp_')}; fx = {m: v for m, v in tab.items() if m.startswith('fixed_')}
        out[f'{split}_{shot}shot_rho{rho:.1f}'] = dict(configurations=len(rr), accuracy=tab, best_lp=max(lp, key=lp.get), best_fixed=max(fx, key=fx.get))
    # selection criterion: mean over the three 4-shot conditions (closed, open 0%, open 50%)
    keys = [k for k in out if '4shot' in k]
    if keys:
        ms = out[keys[0]]['accuracy'].keys(); agg = {m: float(np.mean([out[k]['accuracy'][m] for k in keys])) for m in ms}
        lp = {m: v for m, v in agg.items() if m.startswith('lp_')}; fx = {m: v for m, v in agg.items() if m.startswith('fixed_')}
        out['selection_4shot'] = dict(criterion='mean accuracy over closed, open 0% and open 50% (4-shot)', accuracy=agg, best_lp=max(lp, key=lp.get), best_fixed=max(fx, key=fx.get))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--features-dir', required=True); p.add_argument('--output', required=True); p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--max-contexts', type=int); a = p.parse_args()
    rc.set_out(Path(a.output) / 'work' / 'tcmp_dev'); res = Path(a.output) / 'dev' / 'tcmp_dev'; res.mkdir(parents=True, exist_ok=True)
    rows = Parallel(n_jobs=a.jobs, backend='loky', pre_dispatch='n_jobs')(delayed(run_context)(c, res) for c in itertools.islice(contexts(a.features_dir), a.max_contexts))
    rc.write(Path(a.output) / 'dev' / 'tcmp_dev_summary.json', summarise(rows)); rc.log('DEV done', len(rows))


if __name__ == '__main__':
    main()
