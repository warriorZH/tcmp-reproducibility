"""Recompute every released summary of the additional experiments from its per-configuration files.

No GPU, images or features are needed. For each experiment the per-configuration
JSON records under results/revision/ are passed to the same summarise() function
that produced the released *_summary.json, and the two are compared value by value.

    python scripts/verify_revision.py            # all experiments
    python scripts/verify_revision.py --only w4  # one experiment

Exit status is 0 only if every recomputed value matches (|diff| <= 1e-9);
wall-clock timing fields are listed but not compared.
"""
import argparse, importlib, json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'revision'), str(REPO / 'src')]
RES = REPO / 'results' / 'revision'

# (name, module, per-configuration directory, released summary, expected number of files)
JOBS = [
    ('w2 OOD-filtered self-training, TCMP', 'exp_w2_ood_selftrain', 'w2/tcmp', 'w2/tcmp_summary.json', 42),
    ('w2 OOD-filtered self-training, Indian', 'exp_w2_ood_selftrain', 'w2/indian', 'w2/indian_summary.json', 42),
    ('w4 k x alpha sensitivity, TCMP', 'exp_w4_sensitivity', 'w4/tcmp', 'w4/tcmp_summary.json', 56),
    ('w4 k x alpha sensitivity, Indian', 'exp_w4_sensitivity', 'w4/indian', 'w4/indian_summary.json', 56),
    ('w5 DINOv2, TCMP', 'exp_w5_encoders', 'w5/dinov2/tcmp', 'w5/dinov2_tcmp_summary.json', 49),
    ('w5 DINOv2, Indian', 'exp_w5_encoders', 'w5/dinov2/indian', 'w5/dinov2_indian_summary.json', 49),
    ('w5 CLIP (QuickGELU), TCMP', 'exp_w5_encoders', 'w5/clip-quickgelu/tcmp', 'w5/clip-quickgelu_tcmp_summary.json', 49),
    ('w5 CLIP (QuickGELU), Indian', 'exp_w5_encoders', 'w5/clip-quickgelu/indian', 'w5/clip-quickgelu_indian_summary.json', 49),
    ('w5 superseded CLIP (GELU), TCMP', 'exp_w5_encoders', 'superseded/clip_gelu/w5/clip/tcmp', 'superseded/clip_gelu/w5/clip_tcmp_summary.json', 49),
    ('w5 superseded CLIP (GELU), Indian', 'exp_w5_encoders', 'superseded/clip_gelu/w5/clip/indian', 'superseded/clip_gelu/w5/clip_indian_summary.json', 49),
    ('w5 BioCLIP 2 (recomputed), Indian', 'exp_w5_encoders', 'w5/bioclip2/indian', 'w5/bioclip2_indian_summary.json', 49),
    ('dev development-class grid, TCMP', 'exp_dev_grid', 'dev/tcmp_dev', 'dev/tcmp_dev_summary.json', 56),
]


# Wall-clock timings depend on machine load and were refreshed when a stage was re-run,
# so they are reported but not compared. Every accuracy/risk/interval value is compared.
TIMING = lambda path: 'seconds' in path or '/timing/' in path


def compare(a, b, path=''):
    """Yield (path, released, recomputed) for every mismatch."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield (f'{path}/{k}', a.get(k, '<missing>'), b.get(k, '<missing>'))
            else:
                yield from compare(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if all(isinstance(x, str) for x in a + b):
            if sorted(a) != sorted(b):
                yield (path, a, b)
        elif len(a) != len(b):
            yield (path, a, b)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from compare(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=0, abs_tol=1e-9) or (math.isnan(a) and math.isnan(b))):
            yield (path, a, b)
    elif a != b:
        yield (path, a, b)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--only', help='prefix of the job name, e.g. w2, w4, w5, dev'); a = p.parse_args()
    bad = 0; checked = 0
    for name, mod, d, summ, n in JOBS:
        if a.only and not name.startswith(a.only):
            continue
        files = sorted((RES / d).glob('*.json'))
        rows = [json.loads(f.read_text()) for f in files]
        released = json.loads((RES / summ).read_text())
        recomputed = json.loads(json.dumps(importlib.import_module(mod).summarise(rows)))
        alld = list(compare(released, recomputed))
        timing = [x for x in alld if TIMING(x[0])]; diffs = [x for x in alld if not TIMING(x[0])]
        ok = not diffs and len(files) == n
        bad += not ok; checked += 1
        note = f' ({len(timing)} wall-clock timing fields differ: not compared)' if timing else ''
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {len(files)}/{n} configuration files -> {summ}{note}")
        for d_ in diffs[:5]:
            print('      mismatch', d_)
    for f in ['w3/zeroshot_tcmp.json', 'w3/zeroshot_indian.json'] + [f'w3/finetune_{b}_{ds}_s{s}.json' for b in ['bioclip2_vitl14', 'resnet50_imagenet']
                                                                  for ds, ss in [('tcmp', [4001, 4002, 4003]), ('indian', [5001, 5002, 5003])] for s in ss]:
        present = (RES / f).exists(); bad += not present; checked += 1
        if not present:
            print('FAIL missing', f)
    print(f"w3 zero-shot / fine-tuning result files: {'all 14 present' if bad == 0 else 'see above'} (single runs, no summary to recompute)")
    print('\nALL CHECKS PASSED' if bad == 0 else f'\n{bad} CHECK(S) FAILED')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
