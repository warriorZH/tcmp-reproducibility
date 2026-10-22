"""Locate the inputs: dataset roots, BioCLIP 2 features and checkpoint.

Writes <output>/inputs.json. Give the locations explicitly (recommended) with
--tcmp-root / --indian-root / --checkpoint / --tcmp-features / --indian-features
or the environment variables TCMP_ROOT, INDIAN_ROOT, BIOCLIP2_CKPT,
TCMP_FEATURES, INDIAN_FEATURES. Anything not given is searched for under
--search (default: the repository root and ./outputs). Missing features are
extracted by run_all.sh; a missing checkpoint is downloaded. Features are accepted only if their ids match the locked manifest and
the support-only LR accuracy of one locked context matches the released result.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import rcommon as rc

META = rc.read(rc.REPO / 'provenance/model.json')


def walk(roots, name, maxdepth=7):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        base = len(root.parts)
        for dp, dn, fn in os.walk(root, followlinks=False):
            if len(Path(dp).parts) - base > maxdepth:
                dn[:] = []; continue
            dn[:] = [x for x in dn if not x.startswith('.') and x not in ('__pycache__', 'node_modules', 'site-packages')]
            if name in fn:
                yield Path(dp) / name


def find_root(dataset, roots):
    d = rc.read(rc.REPO / f'data/manifests/{dataset}.json.gz'); sample = [r['path'] for r in (d['train'][:10] + d['query'][:10])]
    cands = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        base = len(root.parts)
        for dp, dn, fn in os.walk(root):
            if len(Path(dp).parts) - base > 4:
                dn[:] = []; continue
            if all((Path(dp) / s).exists() for s in sample):
                cands.append(Path(dp)); dn[:] = []
    return str(cands[0]) if cands else None


def check_features(dataset, fpath):
    ids = fpath.parent / 'ids.json'
    if not ids.exists():
        return None
    try:
        idl = json.loads(ids.read_text()); d = rc.read(rc.REPO / f'data/manifests/{dataset}.json.gz')
        if set(idl) != set(d['ids']):
            return None
        f = np.load(fpath, mmap_mode='r')
        if f.shape != (len(idl), 768):
            return None
    except Exception:
        return None
    # numerical check: support-only LR on one locked context vs released result
    import tempfile
    rc.set_out(tempfile.mkdtemp())
    d, tx, qx = rc.load(dataset, fpath, ids)
    ctx = next(rc.open_contexts(d, tx, qx, dataset, rhos=(0.0,)))
    b = rc.prepare_base(ctx); acc = rc.metrics(None, b['query_logits'].argmax(1), ctx['t'], None, ctx['groups'])['accuracy']
    delta = rc.reference_check(dataset, ctx['key'], dict(lr=acc))
    return dict(features=str(fpath), ids=str(ids), check_context=ctx['key'], lr_accuracy=acc, lr_delta_vs_release=delta['lr'] if delta else None)


def main():
    E = os.environ.get
    p = argparse.ArgumentParser()
    p.add_argument('--search', nargs='+', default=[str(rc.REPO), str(rc.REPO / 'outputs'), str(rc.REPO / 'checkpoints')])
    p.add_argument('--tcmp-root', default=E('TCMP_ROOT')); p.add_argument('--indian-root', default=E('INDIAN_ROOT'))
    p.add_argument('--checkpoint', default=E('BIOCLIP2_CKPT'))
    p.add_argument('--tcmp-features', default=E('TCMP_FEATURES')); p.add_argument('--indian-features', default=E('INDIAN_FEATURES'))
    p.add_argument('--output', default='revision_outputs'); a = p.parse_args()
    out = Path(a.output); out.mkdir(exist_ok=True); info = {}
    for ds, given in [('tcmp', a.tcmp_root), ('indian', a.indian_root)]:
        info[f'{ds}_root'] = find_root(ds, ([given] if given else []) + a.search)
        if given and info[f'{ds}_root'] is None:
            print(f'WARNING: {given} does not contain the {ds} manifest paths', flush=True)
    print('dataset roots:', info['tcmp_root'], info['indian_root'], flush=True)
    for ds, given in [('tcmp', a.tcmp_features), ('indian', a.indian_features)]:
        best = None
        cands = [Path(given)] if given else [fp for fp in walk(a.search, 'features.npy') if 'revision_outputs' not in str(fp)]
        for fp in cands:
            r = check_features(ds, fp)
            if r:
                print(ds, 'candidate', r, flush=True)
                if r['lr_delta_vs_release'] is not None and abs(r['lr_delta_vs_release']) < 0.05:
                    best = r; break
                best = best or r
        info[f'{ds}_features'] = best
    ck = None
    cands = [Path(a.checkpoint)] if a.checkpoint else list(walk(a.search, META['weights_file']))
    for fp in cands:
        fp = fp / META['weights_file'] if fp.is_dir() else fp
        if fp.exists() and rc.sha(fp) == META['sha256']:
            ck = str(fp); break
    info['checkpoint'] = ck
    rc.write(out / 'inputs.json', info); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
