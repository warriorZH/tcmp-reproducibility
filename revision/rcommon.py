"""Shared helpers for the Plant Methods revision experiments.

Everything here reuses the released learners (src/tcmp_repro/learner.py) and the
locked manifests / context indices. Hidden pool and query labels are used only
for evaluation, never by a learner (the 'oracle' OOD filter in W2 is a labelled
diagnostic upper bound and is reported as such).
"""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '4')
import sys, json, time, zipfile
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from tcmp_repro import learner, evaluation, common, run as reprorun  # noqa: E402
from tcmp_repro.io import read, write, sha  # noqa: E402

CFG = read(REPO / 'configs/experiment.json')
evaluation.P = CFG
SEEDS = {'tcmp': CFG['tcmp_seeds'], 'indian': CFG['indian_seeds']}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load(dataset, features, ids):
    d = read(REPO / f'data/manifests/{dataset}.json.gz')
    ids = read(ids)
    assert len(ids) == len(set(ids)) and set(ids) == set(d['ids']), 'feature ids do not match manifest'
    f = np.load(features, mmap_mode='r')
    assert f.shape[0] == len(ids) and np.isfinite(f).all()
    assert np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-4), 'features must be L2-normalised'
    ix = {v: i for i, v in enumerate(ids)}
    tx = np.asarray(f[[ix[r['id']] for r in d['train']]], dtype=np.float32)
    qx = np.asarray(f[[ix[r['id']] for r in d['query']]], dtype=np.float32)
    return d, tx, qx


def set_out(out):
    out = Path(out)
    (out / 'cache').mkdir(parents=True, exist_ok=True)
    common.R = learner.R = out
    return out


def open_contexts(d, tx, qx, dataset, rhos=(0.0, 0.5)):
    """Yield the locked 4-shot fixed-known contexts (21 per rho)."""
    for rho in rhos:
        for c in reprorun.contexts(d, tx, qx, dataset, 'gradients', rho_filter=rho):
            key, x, l, y, t, truth, groups, b, shot, meta = c
            if meta['regime'] != 'fixed_known':
                continue
            yield dict(key=key, x=x, q=qx, l=l, y=y, t=t, truth=truth, groups=groups, b=b, shot=shot, meta=meta)


def closed_contexts(d, tx, qx, dataset, shots=(4, 16)):
    for c in reprorun.contexts(d, tx, qx, dataset, 'closed'):
        key, x, l, y, t, truth, groups, b, shot, meta = c
        if shot in shots:
            yield dict(key=key, x=x, q=qx, l=l, y=y, t=t, truth=truth, groups=groups, b=b, shot=shot, meta=meta)


def prepare_base(ctx):
    b = dict(ctx['b'])
    b['train_logits'] = b['head'].decision_function(ctx['x'])
    b['query_logits'] = b['head'].decision_function(ctx['q'])
    return b


def build_neighbors(x, q, kmax):
    """Exact cosine neighbours with the released convention (k+1, drop first)."""
    nn = NearestNeighbors(n_neighbors=kmax + 1, metric='cosine', n_jobs=2).fit(x)
    dist, ind = nn.kneighbors(x)
    qd, qi = nn.kneighbors(q, n_neighbors=kmax)
    return dict(dist=dist[:, 1:], ind=ind[:, 1:], qd=qd, qi=qi)


def graph_from(nb, k, n):
    from scipy import sparse
    ind = nb['ind'][:, :k]; v = np.maximum(1 - nb['dist'][:, :k], 0) ** 3
    a = sparse.csr_matrix((v.ravel(), (np.repeat(np.arange(n), k), ind.ravel())), shape=(n, n))
    a = a.maximum(a.T)
    a = sparse.diags(1 / np.maximum(np.asarray(a.sum(1)).ravel(), 1e-12)) @ a
    ww = np.maximum(1 - nb['qd'][:, :k], 0) ** 3
    ww /= np.maximum(ww.sum(1, keepdims=True), 1e-12)
    return a, nb['qi'][:, :k], ww


def metrics(score, pred, t, cut, groups):
    return evaluation.pool_group_score(score, pred, t, cut, groups)


def lr_gate(b):
    s = softmax(b['query_logits'], axis=1).max(1)
    cut = float(np.quantile(softmax(b['oof_logits'], axis=1).max(1), .1))
    return s, cut


def lp_gate(g, l, y, nc, alpha, b, lp):
    oo = np.zeros(len(l))
    for a, bb in b['folds']:
        oo[bb] = learner.propagate(g, l[a], y[a], nc, alpha)['train'][l[bb]].max(1)
    return lp['query'].max(1), float(np.quantile(oo, .1)), oo


def paired(rows, a_name, b_name, metric='accuracy', draws=20000, seed=None):
    """Seed-block bootstrap of a-minus-b with roles resampled jointly (as released)."""
    seed = CFG['bootstrap']['seed'] if seed is None else seed
    roles = sorted({r['role'] for r in rows}); seeds = sorted({r['seed'] for r in rows})
    look = {(r['role'], r['seed']): r for r in rows}
    a = np.array([[look[(ro, s)]['acc'][a_name] - look[(ro, s)]['acc'][b_name] for s in seeds] for ro in roles])
    rng = np.random.default_rng(seed); bi = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    means = a[:, bi].mean(axis=(0, 2))
    return dict(mean=float(a.mean()), ci95=[float(v) for v in np.quantile(means, [.025, .975])],
                positive=int((a > 0).sum()), configurations=int(a.size))


def reference_check(dataset, key, acc):
    """Compare default-setting accuracies with the released per-context results."""
    p = REPO / f'results/reference/{dataset}/results/{key}.json'
    if not p.exists():
        return None
    ref = read(p)['metrics']
    out = {}
    for m, v in acc.items():
        if m in ref:
            out[m] = float(v - ref[m]['none']['accuracy'])
    return out


def timer():
    t0 = time.time()
    return lambda: time.time() - t0
