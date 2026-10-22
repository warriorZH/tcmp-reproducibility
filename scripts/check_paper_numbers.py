"""Recompute the numbers printed in the manuscript's additional-experiment tables and text.

Each check pairs a value exactly as printed in the manuscript (Sections 3.3-3.7,
Tables 3-7 of the main text) with the value recomputed from results/revision/
(and, for the frozen-feature rows of Table 4, from results/reference/).
A check passes when the recomputed value, rounded to the printed number of
decimals, equals the printed value. No GPU, images or features are needed.

    python scripts/check_paper_numbers.py            # summary
    python scripts/check_paper_numbers.py --verbose  # every check

Numbers of the original contamination study (Tables 1-2, Figures 4-6) are
checked separately by scripts/compare_analysis.py; the regenerated Additional
file 1 tables (S16-S26) are written by scripts/make_revision_tables.py.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'results' / 'revision'
J = lambda p: json.loads((R / p).read_text())
CHECKS = []


def chk(item, printed, value):
    CHECKS.append((item, printed, value))


def ci(item, printed, d):
    """printed like '1.009 [0.787, 1.263]' (minus signs as '-')."""
    m, lo, hi = [s.strip() for s in printed.replace('[', ',').replace(']', '').split(',')]
    chk(item + ' mean', m, d['mean']); chk(item + ' CI low', lo, d['ci95'][0]); chk(item + ' CI high', hi, d['ci95'][1])


# ---- Table 3 (main text): out-of-distribution-filtered self-training --------------------
OOD = {  # method: (acc0, acc50, loss, LP-minus, unknown removed %, unknown weight %)
    'tcmp': {'lp': ('93.28', '93.10', '0.177', None, None, None),
             'fixed': ('93.40', '92.09', '1.306', '1.009 [0.787, 1.263]', '0', '43.9'),
             'fixed_msp': ('93.39', '92.10', '1.290', '1.000 [0.769, 1.261]', '8.6', '43.8'),
             'fixed_dist': ('93.28', '92.43', '0.852', '0.668 [0.460, 0.919]', '40.9', '33.8'),
             'fixed_lpmsp': ('93.42', '93.07', '0.351', '0.024 [-0.194, 0.280]', '80.0', '16.0'),
             'fixed_oracle': ('93.40', '93.40', '0.000', '-0.297 [-0.531, -0.036]', '100', '0'),
             'fixed_t95': ('93.42', '92.34', '1.086', '0.762 [0.589, 0.958]', '0', '39.8'),
             'soft': ('93.04', '91.29', '1.755', '1.808 [1.455, 2.143]', '0', '46.5'),
             'soft_lpmsp': ('93.33', '92.83', '0.495', '0.265 [0.033, 0.525]', '80.0', '16.6')},
    'indian': {'lp': ('96.20', '96.10', '0.106', None, None, None),
               'fixed': ('96.81', '95.32', '1.484', '0.776 [0.362, 1.129]', '0', '44.8'),
               'fixed_msp': ('96.82', '95.39', '1.428', '0.710 [0.230, 1.130]', '10.0', '44.7'),
               'fixed_dist': ('96.59', '95.81', '0.780', '0.287 [-0.155, 0.689]', '72.4', '19.2'),
               'fixed_lpmsp': ('96.79', '96.46', '0.326', '-0.364 [-0.746, 0.048]', '79.3', '17.1'),
               'fixed_oracle': ('96.81', '96.81', '0.000', '-0.709 [-1.034, -0.334]', '100', '0'),
               'fixed_t95': ('96.80', '95.52', '1.282', '0.579 [0.198, 0.946]', '0', '41.1'),
               'soft': ('96.40', '94.67', '1.731', '1.430 [0.889, 1.996]', '0', '47.0'),
               'soft_lpmsp': ('96.75', '96.14', '0.606', '-0.044 [-0.421, 0.352]', '79.3', '17.0')}}
for ds, rows in OOD.items():
    s = J(f'w2/{ds}_summary.json'); s0, s5 = s['rho0.0'], s['rho0.5']
    for m, (a0, a5, loss, lpm, rem, wt) in rows.items():
        it = f'Table 3 {ds} {m}'
        chk(it + ' acc 0%', a0, s0['accuracy'][m]); chk(it + ' acc 50%', a5, s5['accuracy'][m]); chk(it + ' loss', loss, s['loss_0_to_50'][m])
        if lpm: ci(it + ' LP-minus', lpm, s5['lp_minus'][m])
        if rem: chk(it + ' unknown removed %', rem, 100 * (1 - s5['kept_unknown'][m]))
        if wt: chk(it + ' unknown weight %', wt, 100 * s5['unknown_weight_fraction'][m])
chk('Text 3.3 TCMP unknown pseudo-label weight %', '43.86', 100 * J('w2/tcmp_summary.json')['rho0.5']['unknown_weight_fraction']['fixed'])
chk('Text 3.3 Indian unknown pseudo-label weight %', '44.85', 100 * J('w2/indian_summary.json')['rho0.5']['unknown_weight_fraction']['fixed'])
chk('Text 3.3 TCMP weighted precision % ("about 54")', '54', 100 * J('w2/tcmp_summary.json')['rho0.5']['weighted_precision']['fixed'])

# ---- Table 4: zero-shot and fine-tuning ---------------------------------------------------
zt, zi = J('w3/zeroshot_tcmp.json'), J('w3/zeroshot_indian.json')
chk('Table 4 zero-shot TCMP (Latin binomials)', '87.25', zt['binomial']['closed_accuracy'])
chk('Table 4 zero-shot Indian (common names)', '31.98', zi['common_name']['closed_accuracy'])
chk('Text 3.4 zero-shot TCMP open known accuracy', '89.82', zt['binomial']['open_known_accuracy'])
chk('Text 3.4 zero-shot TCMP gate unknown acceptance %', '36.5', 100 * zt['binomial']['open_gate']['unknown_acceptance'])
REF = REPO / 'results' / 'reference'
for m, (pt, pi) in {'lr': ('87.89', '91.40'), 'fixed': ('91.08', '94.74'), 'lp': ('91.14', '94.37')}.items():  # frozen rows on the same 3 draws
    tc = [json.loads((REF / f'formal/results/closed_s{s}_n4_{m}.json').read_text())['classification']['accuracy'] for s in (4001, 4002, 4003)]
    ic = [json.loads((REF / f'indian/results/closed_s{s}_n4.json').read_text())['metrics'][m]['none']['accuracy'] for s in (5001, 5002, 5003)]
    chk(f'Table 4 frozen BioCLIP 2 {m} TCMP (3 draws)', pt, np.mean(tc)); chk(f'Table 4 frozen BioCLIP 2 {m} Indian (3 draws)', pi, np.mean(ic))
FT = {('bioclip2_vitl14', 'tcmp'): ('87.61', '.27', '62', '6.1'), ('bioclip2_vitl14', 'indian'): ('93.86', '1.52', '16', '6.1'),
      ('resnet50_imagenet', 'tcmp'): ('30.97', '.41', '8', '1.9'), ('resnet50_imagenet', 'indian'): ('65.11', '1.47', '3', '1.9')}
for (bb, ds), (acc, sd, mins, mem) in FT.items():
    fs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(R / f'w3/finetune_{bb}_{ds}_s*.json')))]
    it = f'Table 4 fine-tuned {bb} {ds}'
    chk(it + f' mean ({len(fs)} draws)', acc, np.mean([x['accuracy'] for x in fs]))
    chk(it + ' SD', sd, np.std([x['accuracy'] for x in fs], ddof=1))
    chk(it + ' training GPU-min', mins, np.mean([x['train_seconds'] for x in fs]) / 60)
    if ds == 'tcmp':  # the table prints one memory figure per backbone (the TCMP run)
        chk(it + ' peak GPU memory GB', mem, np.mean([x['peak_gpu_memory_gb'] for x in fs]))

# ---- Table 5: other frozen encoders ----------------------------------------------------------
ENC = {('bioclip2', 'tcmp'): ('90.79', '93.40', '92.09', '93.28', '93.10', '0.177', '1.009 [0.787, 1.263]'),
       ('bioclip2', 'indian'): ('94.26', '96.81', '95.32', '96.20', '96.10', '0.106', '0.776 [0.362, 1.129]'),
       ('dinov2', 'tcmp'): ('85.70', '87.72', '86.70', '88.69', '88.29', '0.398', '1.594 [1.185, 1.994]'),
       ('dinov2', 'indian'): ('93.04', '96.14', '94.74', '95.91', '95.83', '0.079', '1.096 [0.806, 1.360]'),
       ('clip-quickgelu', 'tcmp'): ('64.34', '65.01', '64.25', '65.88', '64.87', '1.014', '0.619 [0.183, 0.986]'),
       ('clip-quickgelu', 'indian'): ('75.86', '80.96', '77.70', '74.37', '74.35', '0.018', '-3.346 [-4.276, -2.213]')}
for (enc, ds), (lr, f0, f5, l0, l5, loss, lpf) in ENC.items():
    it = f'Table 5 {enc} {ds}'
    if (enc, ds) == ('bioclip2', 'tcmp'):  # BioCLIP 2 TCMP row is the main experiment (same values as Table 3)
        s = J('w2/tcmp_summary.json'); o0, o5, lossd = s['rho0.0'], s['rho0.5'], s['loss_0_to_50']
    else:
        s = J(f'w5/{enc}_{ds}_summary.json'); o0, o5, lossd = s['open_rho0.0'], s['open_rho0.5'], s['loss_0_to_50']
    chk(it + ' LR', lr, o0['accuracy']['lr']); chk(it + ' fixed 0%', f0, o0['accuracy']['fixed']); chk(it + ' fixed 50%', f5, o5['accuracy']['fixed'])
    chk(it + ' LP 0%', l0, o0['accuracy']['lp']); chk(it + ' LP 50%', l5, o5['accuracy']['lp']); chk(it + ' LP loss', loss, lossd['lp'])
    ci(it + ' LP-fixed at 50%', lpf, o5['lp_minus']['fixed'])

ct, ci_ = J('w5/clip-quickgelu_tcmp_summary.json'), J('w5/clip-quickgelu_indian_summary.json')
chk('Text 3.5 CLIP TCMP reference-only 10NN %', '52', ct['open_rho0.0']['accuracy']['support_knn'])
chk('Text 3.5 CLIP Indian reference-only 10NN %', '56', ci_['open_rho0.0']['accuracy']['support_knn'])
chk('Text 3.5 CLIP TCMP LP-LR at 50%', '0.53', ct['open_rho0.5']['lp_minus']['lr']['mean'])
ci('Text 3.5 CLIP TCMP LP-fixed at 50%', '0.62 [0.18, 0.99]', ct['open_rho0.5']['lp_minus']['fixed'])
chk('Text 3.5 CLIP TCMP LP loss', '1.01', ct['loss_0_to_50']['lp']); chk('Text 3.5 CLIP TCMP fixed loss', '0.76', ct['loss_0_to_50']['fixed'])
chk('Text 3.5 CLIP TCMP LP-soft at 50% not significant (CI spans 0)', True, ct['open_rho0.5']['lp_minus']['soft']['ci95'][0] < 0 < ct['open_rho0.5']['lp_minus']['soft']['ci95'][1])
chk('Text 3.5 CLIP Indian LP-LR at 50%', '-1.5', ci_['open_rho0.5']['lp_minus']['lr']['mean'])
chk('Text 3.5 CLIP Indian LP-fixed at 50%', '-3.3', ci_['open_rho0.5']['lp_minus']['fixed']['mean'])

# ---- Table 6 and Section 3.6: gate operating points --------------------------------------
GATES = {'lr_msp_q0.05': {'tcmp': ('96.5', '54.2', '5.5'), 'indian': ('96.0', '52.1', '2.7')},
         'lr_msp_q0.1': {'tcmp': ('93.0', '33.3', '4.4'), 'indian': ('92.4', '33.0', '2.0')},
         'lr_msp_q0.2': {'tcmp': ('85.7', '15.2', '2.9'), 'indian': ('84.8', '14.6', '1.1')},
         'lr_msp_ua10': {'tcmp': ('80.7', '10.0', '2.2'), 'indian': ('81.9', '10.1', '0.9')},
         'lr_msp_ua5': {'tcmp': ('70.3', '5.0', '1.3'), 'indian': ('74.3', '5.0', '0.6')}}
for g, per in GATES.items():
    for ds, (cov, ua, risk) in per.items():
        o = J(f'w4/{ds}_summary.json')['open_4shot_rho0.5']['gates'][g]; it = f'Table 6 {ds} {g}'
        chk(it + ' coverage', cov, 100 * o['known_coverage']); chk(it + ' unknown acceptance', ua, 100 * o['unknown_acceptance']); chk(it + ' risk', risk, 100 * o['accepted_known_risk'])
gt = J('w4/tcmp_summary.json')
chk('Text 3.6 TCMP graph-MSP gate UA at 50%', '22.2', 100 * gt['open_4shot_rho0.5']['gates']['lp_msp_q0.1']['unknown_acceptance'])
chk('Text 3.6 TCMP graph-MSP gate coverage at 50%', '89.5', 100 * gt['open_4shot_rho0.5']['gates']['lp_msp_q0.1']['known_coverage'])
chk('Text 3.6 TCMP graph-MSP gate UA, clean archive', '52.65', 100 * gt['open_4shot_rho0.0']['gates']['lp_msp_q0.1']['unknown_acceptance'])
chk('Text 3.6 TCMP supervised gate UA, clean archive', '33.33', 100 * gt['open_4shot_rho0.0']['gates']['lr_msp_q0.1']['unknown_acceptance'])

# ---- Table 7 and Section 3.7: k x alpha sensitivity and development grid --------------------
SENS = {'tcmp': {'closed_4shot_rho0.0': ('91.15', 'k10_a0.95', '91.23', '0.93', 15), 'closed_16shot_rho0.0': ('93.00', 'k5_a0.99', '93.08', '0.06', 21),
                 'open_4shot_rho0.0': ('93.28', 'k10_a0.9', '93.28', '0.45', 19), 'open_4shot_rho0.5': ('93.10', 'k10_a0.95', '93.11', '0.55', 19)},
        'indian': {'closed_4shot_rho0.0': ('94.38', 'k5_a0.95', '95.36', '0.25', 5), 'closed_16shot_rho0.0': ('96.14', 'k5_a0.99', '96.89', '0.09', 6),
                   'open_4shot_rho0.0': ('96.20', 'k5_a0.99', '96.79', '0.13', 6), 'open_4shot_rho0.5': ('96.10', 'k5_a0.99', '97.05', '0.06', 5)}}  # corrected from 6/30 in manuscript version 8 (22 Oct 2026)
for ds, conds in SENS.items():
    s = J(f'w4/{ds}_summary.json')
    for c, (dflt, best, bacc, rng, within) in conds.items():
        o = s[c]; acc = o['accuracy']; it = f'Table 7 {ds} {c}'; k10 = [acc[f'k10_a{a}'] for a in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]]
        chk(it + ' default', dflt, acc[o['default']]); chk(it + ' best setting', best, o['best']); chk(it + ' best', bacc, acc[o['best']])
        k10 = [round(v, 2) for v in k10]  # range as read from the printed grid (Additional file 1, Tables S16-S17)
        chk(it + ' range at k=10', rng, max(k10) - min(k10)); chk(it + ' within 0.5 of best', within, len(o['within_0_5_of_best']))
    fixed50 = J(f'w2/{ds}_summary.json')['rho0.5']['accuracy']['fixed']
    chk(f'Text 3.7 {ds} settings where PropGate > fixed self-training at 50%', {'tcmp': 26, 'indian': 25}[ds],
        sum(v > fixed50 for v in s['open_4shot_rho0.5']['accuracy'].values()))
gap = max(-J('w4/tcmp_summary.json')[c]['default_minus_best'] for c in SENS['tcmp'])
chk('Text 3.7 TCMP max default-to-best gap <= 0.09 ("within 0.09")', True, gap <= 0.09)
dv = J('dev/tcmp_dev_summary.json')['selection_4shot']
chk('Text 3.7 development grid best LP setting', 'lp_k5_a0.99', dv['best_lp']); chk('Text 3.7 development grid best LP', '96.42', dv['accuracy'][dv['best_lp']])
chk('Text 3.7 development grid default LP', '96.29', dv['accuracy']['lp_k10_a0.9']); chk('Text 3.7 development grid best fixed threshold', 'fixed_t0.95', dv['best_fixed'])
chk('Text 3.7 development best fixed minus default fixed', '0.03', dv['accuracy']['fixed_t0.95'] - dv['accuracy']['fixed_t0.9'])


def ok(printed, value):
    if isinstance(printed, (int, bool)) or isinstance(value, (str, bool)):
        return printed == value
    dec = len(printed.split('.')[1]) if '.' in printed else 0
    return abs(round(float(value), dec) - float(printed)) < 1e-9 or abs(float(value) - float(printed)) <= 0.5 * 10 ** -dec + 1e-12


def main():
    p = argparse.ArgumentParser(); p.add_argument('--verbose', action='store_true'); a = p.parse_args()
    bad = [(i, pr, v) for i, pr, v in CHECKS if not ok(pr, v)]
    for i, pr, v in CHECKS:
        if a.verbose or (i, pr, v) in bad:
            print(f"{'OK  ' if (i, pr, v) not in bad else 'FAIL'} {i}: printed {pr}, recomputed {v if isinstance(v, (str, int)) else round(float(v), 6)}")
    print(f'{len(CHECKS) - len(bad)}/{len(CHECKS)} manuscript values reproduced from the released results')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
