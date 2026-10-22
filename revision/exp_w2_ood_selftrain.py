"""W2 / Q1: out-of-distribution-filtered self-training on the same frozen features.

For every locked 4-shot fixed-known context at 0% and 50% contamination we run
fixed-threshold (0.9) and soft self-training after excluding pool images flagged
as unknown by a support-calibrated filter. All filters are label-free except
the clearly marked 'oracle' upper bound.

  msp    : untempered LR MSP >= tau_LR (support OOF 10% quantile), re-evaluated with
           the current head before each refit (dynamic, as in Q1 of the review)
  lpmsp  : graph MSP of the pool row >= tau_LP (held-out-support propagation, 10%)
  dist   : max cosine similarity to the support set >= 10% quantile of held-out
           support-to-support similarity (distance-based class-mismatch filter)
  oracle : remove all true unknown pool images (diagnostic upper bound, uses truth)

A threshold sensitivity for unfiltered fixed self-training (0.7, 0.95) is added.
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
from scipy.special import softmax
from joblib import Parallel, delayed
import rcommon as rc
from tcmp_repro import learner

FILTERS = ['msp', 'lpmsp', 'dist', 'oracle']


def selftrain_filtered(x, q, l, y, c, b, kind, thr, keep_static=None, dyn_cut=None):
    h = b['head']; u = np.setdiff1d(np.arange(len(x)), l); temp = b['temperature']; hist = []
    w = np.zeros(len(u)); label = np.zeros(len(u), int); keep = np.ones(len(u), bool)
    for it in range(3):
        logits = h.decision_function(x[u]); p = softmax(logits / temp, axis=1); conf = p.max(1); label = p.argmax(1)
        keep = np.ones(len(u), bool) if keep_static is None else keep_static.copy()
        if dyn_cut is not None:
            keep &= softmax(logits, axis=1).max(1) >= dyn_cut
        if kind == 'fixed':
            w = ((conf >= thr) & keep).astype(float)
        else:
            if keep.any():
                mu = conf[keep].mean(); sd = max(conf[keep].std(), .05)
                w = np.exp(-np.maximum(mu - conf, 0) ** 2 / (2 * sd * sd)) * keep
            else:
                w = np.zeros(len(u))
        hist.append(dict(iteration=it, kept=int(keep.sum()), selected=int((w > 0).sum()), weight_sum=float(w.sum())))
        if w.sum() == 0:
            break
        scale = len(l) / max(w.sum(), 1e-30)
        h = learner.fit(np.r_[x[l], x[u]], np.r_[y, label], c, np.r_[np.ones(len(l)), scale * w])
    return dict(query=softmax(h.decision_function(q) / temp, axis=1), u=u, w=w, label=label, keep=keep, history=hist)


def pseudo_stats(out, truth):
    t = truth[out['u']]; w = out['w']; k = out['keep']
    return dict(kept_known_fraction=float(k[t >= 0].mean()), kept_unknown_fraction=float(k[t < 0].mean()) if (t < 0).any() else None,
                selected=int((w > 0).sum()), weight_sum=float(w.sum()),
                unknown_weight_fraction=float(w[t < 0].sum() / w.sum()) if w.sum() else None,
                weighted_precision=float(np.sum(w * (out['label'] == t)) / w.sum()) if w.sum() else None, history=out['history'])


def run_context(ctx, dataset, out_dir):
    target = out_dir / f"{ctx['key']}.json"
    if target.exists():
        return rc.read(target)
    tm = rc.timer(); x, q, l, y, t, truth, groups = ctx['x'], ctx['q'], ctx['l'], ctx['y'], ctx['t'], ctx['truth'], ctx['groups']
    shot = ctx['shot']; alpha = rc.CFG['alpha'][str(shot)]; C = rc.CFG['C']; nc = int(y.max() + 1)
    b = rc.prepare_base(ctx); nb = rc.build_neighbors(x, q, rc.CFG['graph_k']); g = rc.graph_from(nb, rc.CFG['graph_k'], len(x))
    lp = learner.propagate(g, l, y, nc, alpha)
    s_lr, cut_lr = rc.lr_gate(b); s_lp, cut_lp, _ = rc.lp_gate(g, l, y, nc, alpha, b, lp)
    u = np.setdiff1d(np.arange(len(x)), l)
    # support-distance filter, calibrated on held-out support folds
    sims = x[u] @ x[l].T; pool_sim = sims.max(1); held = np.zeros(len(l))
    for a, bb in b['folds']:
        held[bb] = (x[l[bb]] @ x[l[a]].T).max(1)
    cut_dist = float(np.quantile(held, .1))
    keeps = dict(lpmsp=lp['train'][u].max(1) >= cut_lp, dist=pool_sim >= cut_dist, oracle=truth[u] >= 0)
    probs = dict(lr=softmax(b['query_logits'], axis=1), lp=lp['query']); stats = {}
    for kind in ['fixed', 'soft']:
        thr = rc.CFG['fixed_threshold'][str(shot)]
        o = selftrain_filtered(x, q, l, y, C, b, kind, thr); probs[kind] = o['query']; stats[kind] = pseudo_stats(o, truth)
        for f in FILTERS:
            o = selftrain_filtered(x, q, l, y, C, b, kind, thr, keep_static=keeps.get(f), dyn_cut=cut_lr if f == 'msp' else None)
            probs[f'{kind}_{f}'] = o['query']; stats[f'{kind}_{f}'] = pseudo_stats(o, truth)
    for thr in [0.7, 0.95]:
        o = selftrain_filtered(x, q, l, y, C, b, 'fixed', thr); name = f'fixed_t{int(thr*100)}'
        probs[name] = o['query']; stats[name] = pseudo_stats(o, truth)
    acc, gated = {}, {}
    for m, p in probs.items():
        pred = p.argmax(1)
        acc[m] = rc.metrics(None, pred, t, None, groups)['accuracy']
        gated[m] = dict(lr_msp=rc.metrics(s_lr, pred, t, cut_lr, groups), lp_msp=rc.metrics(s_lp, pred, t, cut_lp, groups))
    rec = dict(key=ctx['key'], dataset=dataset, **ctx['meta'], acc=acc, gated=gated, stats=stats,
               cuts=dict(lr=cut_lr, lp=cut_lp, dist=cut_dist), reference_delta=rc.reference_check(dataset, ctx['key'], dict(lr=acc['lr'], lp=acc['lp'], fixed=acc['fixed'], soft=acc['soft'])),
               seconds=tm())
    rc.write(target, rec); rc.log('W2', dataset, ctx['key'], 'LP %.2f fixed %.2f fixed_lpmsp %.2f fixed_oracle %.2f (%.0fs)' % (acc['lp'], acc['fixed'], acc['fixed_lpmsp'], acc['fixed_oracle'], tm()))
    return rec


def summarise(rows):
    out = {}
    methods = list(rows[0]['acc'])
    for rho in sorted({r['rho'] for r in rows}):
        rr = [r for r in rows if r['rho'] == rho]
        out[f'rho{rho:.1f}'] = dict(
            configurations=len(rr),
            accuracy={m: float(np.mean([r['acc'][m] for r in rr])) for m in methods},
            lr_gate_known_risk={m: float(np.mean([r['gated'][m]['lr_msp']['accepted_known_risk'] for r in rr])) for m in methods},
            unknown_weight_fraction={m: (float(np.mean([r['stats'][m]['unknown_weight_fraction'] for r in rr if r['stats'][m]['unknown_weight_fraction'] is not None])) if any(r['stats'][m]['unknown_weight_fraction'] is not None for r in rr) else None) for m in methods if m in rr[0]['stats']},
            weighted_precision={m: (float(np.mean([r['stats'][m]['weighted_precision'] for r in rr if r['stats'][m]['weighted_precision'] is not None])) if any(r['stats'][m]['weighted_precision'] is not None for r in rr) else None) for m in methods if m in rr[0]['stats']},
            kept_known={m: float(np.mean([r['stats'][m]['kept_known_fraction'] for r in rr])) for m in methods if m in rr[0]['stats']},
            kept_unknown={m: (float(np.mean([r['stats'][m]['kept_unknown_fraction'] for r in rr if r['stats'][m]['kept_unknown_fraction'] is not None])) if rho > 0 else None) for m in methods if m in rr[0]['stats']},
            lp_minus={m: rc.paired(rr, 'lp', m) for m in methods if m != 'lp'},
            max_abs_reference_delta=float(max(abs(v) for r in rr if r['reference_delta'] for v in r['reference_delta'].values())) if any(r['reference_delta'] for r in rr) else None,
            mean_seconds=float(np.mean([r['seconds'] for r in rr])))
    if 'rho0.0' in out and 'rho0.5' in out:
        out['loss_0_to_50'] = {m: out['rho0.0']['accuracy'][m] - out['rho0.5']['accuracy'][m] for m in methods}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--features', required=True); p.add_argument('--ids', required=True)
    p.add_argument('--output', required=True); p.add_argument('--jobs', type=int, default=4); p.add_argument('--max-contexts', type=int); a = p.parse_args()
    out = rc.set_out(Path(a.output) / 'work' / a.dataset); res = Path(a.output) / 'w2' / a.dataset; res.mkdir(parents=True, exist_ok=True)
    d, tx, qx = rc.load(a.dataset, a.features, a.ids)
    ctxs = itertools.islice(rc.open_contexts(d, tx, qx, a.dataset), a.max_contexts)
    rows = Parallel(n_jobs=a.jobs, backend='loky', pre_dispatch='n_jobs')(delayed(run_context)(c, a.dataset, res) for c in ctxs)
    rc.write(Path(a.output) / 'w2' / f'{a.dataset}_summary.json', summarise(rows)); rc.log('W2 done', a.dataset, len(rows))


if __name__ == '__main__':
    main()
