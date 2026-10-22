"""W5: is the pipeline encoder-agnostic? Same protocol, same (untuned) settings,
features from another frozen encoder (DINOv2 ViT-L/14 or CLIP ViT-L/14).

Runs the closed 4-shot contexts and the 4-shot fixed-known contexts at 0% and
50% contamination with LR, fixed/soft self-training, iterative and one-step LP,
support 10NN and NCM, plus both support-calibrated gates. With --encoder
bioclip2 the same code reproduces the released numbers (sanity check).
"""
import argparse, itertools
from pathlib import Path
import numpy as np
from scipy.special import softmax
from joblib import Parallel, delayed
import rcommon as rc
from tcmp_repro import learner, evaluation


def evaluate_default(ctx):
    x, q, l, y, t, groups = ctx['x'], ctx['q'], ctx['l'], ctx['y'], ctx['t'], ctx['groups']
    shot = ctx['shot']; alpha = rc.CFG['alpha'][str(shot)]; nc = int(y.max() + 1); C = rc.CFG['C']; tim = {}
    b = rc.prepare_base(ctx)
    tm = rc.timer(); nb = rc.build_neighbors(x, q, rc.CFG['graph_k']); g = rc.graph_from(nb, rc.CFG['graph_k'], len(x)); tim['graph_build_seconds'] = tm()
    tm = rc.timer(); lp = learner.propagate(g, l, y, nc, alpha); tim['lp_seconds'] = tm()
    probs = dict(lr=softmax(b['query_logits'], axis=1), lp=lp['query'], one_step_lp=evaluation.one_step(g, l, y, nc, alpha))
    probs['support_knn'], probs['ncm'] = evaluation.support_controls(x, q, l, y, nc)
    for kind in ['fixed', 'soft']:
        tm = rc.timer(); probs[kind] = learner.selftrain(x, q, l, y, C, b, kind, rc.CFG['fixed_threshold'][str(shot)], rc.CFG['soft_mass'])['query']; tim[kind + '_seconds'] = tm()
    s_lr, cut_lr = rc.lr_gate(b); s_lp, cut_lp, _ = rc.lp_gate(g, l, y, nc, alpha, b, lp)
    acc, gated = {}, {}
    for m, p in probs.items():
        pred = p.argmax(1); mm = rc.metrics(None, pred, t, None, groups); acc[m] = mm['accuracy']
        gated[m] = dict(none={k: mm[k] for k in ['accuracy', 'macro_recall', 'worst_class_recall']})
        if ctx['meta']['split'] == 'open':
            for gate, s, cut in [('lr_msp', s_lr, cut_lr), ('lp_msp', s_lp, cut_lp)]:
                gm = rc.metrics(s, pred, t, cut, groups)
                gated[m][gate] = {k: gm[k] for k in ['known_coverage', 'unknown_acceptance', 'accepted_known_risk', 'overall_accepted_error', 'oscr']}
    return acc, gated, tim


def run_context(ctx, dataset, encoder, res):
    target = res / f"{ctx['key']}.json"
    if target.exists():
        return rc.read(target)
    tm = rc.timer(); acc, gated, tim = evaluate_default(ctx)
    rec = dict(key=ctx['key'], dataset=dataset, encoder=encoder, **ctx['meta'], shot=ctx['shot'], acc=acc, gated=gated, timing=tim,
               dim=int(ctx['x'].shape[1]), n_train=len(ctx['x']),
               reference_delta=rc.reference_check(dataset, ctx['key'], acc) if encoder == 'bioclip2' else None, seconds=tm())
    rc.write(target, rec); rc.log('W5', encoder, dataset, ctx['key'], 'LR %.2f fixed %.2f LP %.2f (%.0fs)' % (acc['lr'], acc['fixed'], acc['lp'], tm()))
    return rec


def summarise(rows):
    out = {}
    for split, rho in sorted({(r['split'], r['rho']) for r in rows}):
        rr = [r for r in rows if (r['split'], r['rho']) == (split, rho)]; ms = list(rr[0]['acc'])
        o = dict(configurations=len(rr), accuracy={m: float(np.mean([r['acc'][m] for r in rr])) for m in ms},
                 accuracy_sd={m: (float(np.std([r['acc'][m] for r in rr], ddof=1)) if len(rr) > 1 else None) for m in ms},
                 timing={k: float(np.mean([r['timing'][k] for r in rr])) for k in rr[0]['timing']})
        if split == 'open':
            o['lp_minus'] = {m: rc.paired(rr, 'lp', m) for m in ms if m != 'lp'}
            o['gates'] = {m: {g: {k: float(np.mean([r['gated'][m][g][k] for r in rr if r['gated'][m][g][k] is not None])) for k in rr[0]['gated'][m][g]} for g in ['lr_msp', 'lp_msp']} for m in ['lp', 'fixed', 'lr']}
        out[f'{split}_rho{rho:.1f}'] = o
    if 'open_rho0.0' in out and 'open_rho0.5' in out:
        rr0 = {(r['role'], r['seed']): r for r in rows if r['split'] == 'open' and r['rho'] == 0}
        loss_rows = [dict(role=r['role'], seed=r['seed'], acc={m: rr0[(r['role'], r['seed'])]['acc'][m] - r['acc'][m] for m in r['acc']}) for r in rows if r['split'] == 'open' and r['rho'] == .5]
        out['loss_0_to_50'] = {m: out['open_rho0.0']['accuracy'][m] - out['open_rho0.5']['accuracy'][m] for m in out['open_rho0.0']['accuracy']}
        out['fixed_minus_lp_loss'] = rc.paired(loss_rows, 'fixed', 'lp'); out['soft_minus_lp_loss'] = rc.paired(loss_rows, 'soft', 'lp')
    d = [abs(v) for r in rows if r.get('reference_delta') for v in r['reference_delta'].values()]
    out['max_abs_reference_delta'] = float(max(d)) if d else None
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--encoder', required=True)
    p.add_argument('--features', required=True); p.add_argument('--ids', required=True); p.add_argument('--output', required=True)
    p.add_argument('--jobs', type=int, default=4); p.add_argument('--max-contexts', type=int); a = p.parse_args()
    rc.set_out(Path(a.output) / 'work' / f'{a.encoder}_{a.dataset}'); res = Path(a.output) / 'w5' / a.encoder / a.dataset; res.mkdir(parents=True, exist_ok=True)
    d, tx, qx = rc.load(a.dataset, a.features, a.ids)
    gen = itertools.chain(rc.closed_contexts(d, tx, qx, a.dataset, shots=(4,)), rc.open_contexts(d, tx, qx, a.dataset))
    rows = Parallel(n_jobs=a.jobs, backend='loky', pre_dispatch='n_jobs')(delayed(run_context)(c, a.dataset, a.encoder, res) for c in itertools.islice(gen, a.max_contexts))
    rc.write(Path(a.output) / 'w5' / f'{a.encoder}_{a.dataset}_summary.json', summarise(rows)); rc.log('W5 done', a.encoder, a.dataset, len(rows))


if __name__ == '__main__':
    main()
