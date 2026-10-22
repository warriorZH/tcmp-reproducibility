"""W4 / Q2: sensitivity of iterative LP to the graph size k and the propagation weight alpha.

Grid: k in {5,10,15,20,30} x alpha in {0.5,0.7,0.8,0.9,0.95,0.99}; the inductive
query extension uses the same k (as in the released method). Evaluated on the
locked 4-shot fixed-known contexts at 0% and 50% contamination and on the closed
4-/16-shot contexts. This grid is post hoc: it is reported for users and was NOT
used to choose the defaults (k=10, alpha=0.9/0.8 were fixed on the 60 TCMP
development classes). The acceptance-quantile sensitivity (5/10/20%) of both
gates is computed at the default graph.
"""
import argparse, itertools
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import rcommon as rc
from tcmp_repro import learner

K = [5, 10, 15, 20, 30]
A = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
Q = [0.05, 0.1, 0.2]


def run_context(ctx, dataset, res):
    target = res / f"{ctx['key']}.json"
    if target.exists():
        return rc.read(target)
    tm = rc.timer(); x, q, l, y, t, groups = ctx['x'], ctx['q'], ctx['l'], ctx['y'], ctx['t'], ctx['groups']
    nc = int(y.max() + 1); shot = ctx['shot']; a0 = rc.CFG['alpha'][str(shot)]; k0 = rc.CFG['graph_k']
    nb = rc.build_neighbors(x, q, max(K)); grid = {}; per_prop = []
    for k in K:
        g = rc.graph_from(nb, k, len(x))
        for al in A:
            t0 = rc.timer(); lp = learner.propagate(g, l, y, nc, al); per_prop.append(t0())
            pred = lp['query'].argmax(1); m = rc.metrics(None, pred, t, None, groups)
            grid[f'k{k}_a{al}'] = dict(accuracy=m['accuracy'], macro_recall=m['macro_recall'], worst_class_recall=m['worst_class_recall'])
            if k == k0 and al == a0:
                default = (g, lp, pred)
    b = rc.prepare_base(ctx); g, lp, pred = default
    s_lr = rc.lr_gate(b)[0]; from scipy.special import softmax
    oof_lr = softmax(b['oof_logits'], axis=1).max(1); s_lp, _, oof_lp = rc.lp_gate(g, l, y, nc, a0, b, lp)
    gates = {}
    if ctx['meta']['split'] == 'open':
        for qq in Q:
            for name, s, oo in [('lr_msp', s_lr, oof_lr), ('lp_msp', s_lp, oof_lp)]:
                m = rc.metrics(s, pred, t, float(np.quantile(oo, qq)), groups)
                gates[f'{name}_q{qq}'] = {kk: m[kk] for kk in ['known_coverage', 'unknown_acceptance', 'accepted_known_risk', 'overall_accepted_error']}
            # operating points at target unknown acceptance (diagnostic: uses query truth)
            for name, s in [('lr_msp', s_lr), ('lp_msp', s_lp)]:
                for target_ua in [0.05, 0.10]:
                    cut = float(np.quantile(s[t < 0], 1 - target_ua)); m = rc.metrics(s, pred, t, cut, groups)
                    gates[f'{name}_ua{int(target_ua*100)}'] = {kk: m[kk] for kk in ['known_coverage', 'unknown_acceptance', 'accepted_known_risk']}
    rec = dict(key=ctx['key'], dataset=dataset, **ctx['meta'], shot=shot, grid=grid, gates=gates, n_train=len(x),
               lp_seconds_mean=float(np.mean(per_prop)),
               reference_delta=rc.reference_check(dataset, ctx['key'], dict(lp=grid[f'k{k0}_a{a0}']['accuracy'])), seconds=tm())
    rc.write(target, rec); rc.log('W4', dataset, ctx['key'], 'default %.2f best %.2f (%.0fs)' % (grid[f'k{k0}_a{a0}']['accuracy'], max(v['accuracy'] for v in grid.values()), tm()))
    return rec


def summarise(rows):
    out = {}
    for split, shot, rho in sorted({(r['split'], r['shot'], r['rho']) for r in rows}):
        rr = [r for r in rows if (r['split'], r['shot'], r['rho']) == (split, shot, rho)]
        tab = {g: float(np.mean([r['grid'][g]['accuracy'] for r in rr])) for g in rr[0]['grid']}
        best = max(tab, key=tab.get); a0 = rc.CFG['alpha'][str(shot)]
        o = dict(configurations=len(rr), accuracy=tab, macro_recall={g: float(np.mean([r['grid'][g]['macro_recall'] for r in rr])) for g in tab},
                 default=f"k{rc.CFG['graph_k']}_a{a0}", best=best, default_minus_best=tab[f"k{rc.CFG['graph_k']}_a{a0}"] - tab[best],
                 within_0_5_of_best=[g for g in tab if tab[best] - tab[g] <= .5], lp_seconds_mean=float(np.mean([r['lp_seconds_mean'] for r in rr])))
        if rr[0]['gates']:
            o['gates'] = {g: {kk: float(np.mean([r['gates'][g][kk] for r in rr if r['gates'][g][kk] is not None])) for kk in rr[0]['gates'][g]} for g in rr[0]['gates']}
        deltas = [abs(v) for r in rr if r['reference_delta'] for v in r['reference_delta'].values()]
        o['max_abs_reference_delta'] = float(max(deltas)) if deltas else None
        out[f'{split}_{shot}shot_rho{rho:.1f}'] = o
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--features', required=True); p.add_argument('--ids', required=True)
    p.add_argument('--output', required=True); p.add_argument('--jobs', type=int, default=4); p.add_argument('--max-contexts', type=int); a = p.parse_args()
    rc.set_out(Path(a.output) / 'work' / a.dataset); res = Path(a.output) / 'w4' / a.dataset; res.mkdir(parents=True, exist_ok=True)
    d, tx, qx = rc.load(a.dataset, a.features, a.ids)
    gen = itertools.chain(rc.closed_contexts(d, tx, qx, a.dataset), rc.open_contexts(d, tx, qx, a.dataset))
    rows = Parallel(n_jobs=a.jobs, backend='loky', pre_dispatch='n_jobs')(delayed(run_context)(c, a.dataset, res) for c in itertools.islice(gen, a.max_contexts))
    rc.write(Path(a.output) / 'w4' / f'{a.dataset}_summary.json', summarise(rows)); rc.log('W4 done', a.dataset, len(rows))


if __name__ == '__main__':
    main()
