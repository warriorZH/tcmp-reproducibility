"""Regenerate the Additional file 1 tables of the additional experiments (Sections S7-S11, Tables S16-S26).

    python scripts/make_revision_tables.py                       # -> outputs/revision_tables/additional_file_1_S7-S11.tex
    python scripts/make_revision_tables.py --check               # also compare with docs/validation/revision_tables_reference.tex

The LaTeX is the source of the published tables (wording was edited afterwards in the manuscript;
all numbers are unchanged).
"""
import json, glob, os, sys
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
R = REPO / 'results' / 'revision'
OUT = REPO / 'outputs' / 'revision_tables' / 'additional_file_1_S7-S11.tex'; OUT.parent.mkdir(parents=True, exist_ok=True)
CHECK = '--check' in sys.argv
J = lambda p: json.load(open(p))
f2 = lambda v: '--' if v is None else f'{v:.2f}'
pct = lambda v: '--' if v is None else f'{100*v:.1f}'
neg = lambda s: s.replace('-', '$-$') if s.startswith('-') or ' -' in s or '[-' in s else s
def ci(d): return neg(f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]").replace('[-', '[$-$').replace(', -', ', $-$')
def box(tab): return '\\sbox{\\tablebox}{' + tab + '}\n\\ifdim\\wd\\tablebox>\\linewidth\\resizebox{\\linewidth}{!}{\\usebox{\\tablebox}}\\else\\usebox{\\tablebox}\\fi\n'
L = []
KS = [5, 10, 15, 20, 30]; AS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
NAMES = {'closed_4shot_rho0.0': 'closed, 4-shot', 'closed_16shot_rho0.0': 'closed, 16-shot', 'open_4shot_rho0.0': 'open 4-shot, 0\\%', 'open_4shot_rho0.5': 'open 4-shot, 50\\%'}
DS = {'tcmp': 'TCMP', 'indian': 'Indian'}

# ---------------- S7 sensitivity
L.append('\\section{Parameter and gate sensitivity}\\label{sec:s-sens}')
L.append('This section supports Section~3.7 of the main text. The $k\\times\\alpha$ grid uses the locked contexts and the same inductive extension with the same $k$. It was computed after the main experiments for reporting only; no default was changed. Entries are mean known accuracy (\\%) over 7 (closed) or 21 (open) configurations; the default is in bold.\n')
for ds in ['tcmp', 'indian']:
    s = J(R / 'w4' / f'{ds}_summary.json')
    rows = []
    for cond in ['closed_4shot_rho0.0', 'closed_16shot_rho0.0', 'open_4shot_rho0.0', 'open_4shot_rho0.5']:
        o = s[cond]; d = o['default']
        for i, k in enumerate(KS):
            cells = []
            for a in AS:
                g = f'k{k}_a{a}'; v = f2(o['accuracy'][g]); cells.append(f'\\textbf{{{v}}}' if g == d else v)
            rows.append(((NAMES[cond] if i == 0 else '') + f' & {k} & ' + ' & '.join(cells) + '\\\\'))
        rows.append('\\midrule')
    rows = rows[:-1]
    tab = '\\begin{tabular}{llrrrrrr}\\toprule\nCondition & $k$ & ' + ' & '.join(f'$\\alpha={a}$' for a in AS) + '\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
    L.append(f'\\begin{{table}}[!htbp]\\centering\\small\n\\caption{{{DS[ds]}: PropGate known accuracy (\\%) over the $k\\times\\alpha$ grid.}}\\label{{tab:s-grid-{ds}}}\n' + box(tab) + '\\end{table}\n')
# gate table
rows = []
for ds in ['tcmp', 'indian']:
    s = J(R / 'w4' / f'{ds}_summary.json')
    for cond in ['open_4shot_rho0.0', 'open_4shot_rho0.5']:
        g = s[cond]['gates']
        for gate, gname in [('lr_msp', 'LR MSP'), ('lp_msp', 'graph MSP')]:
            for key, lab in [('q0.05', 'quantile 5\\%'), ('q0.1', 'quantile 10\\% (default)'), ('q0.2', 'quantile 20\\%'), ('ua10', 'target UA 10\\%'), ('ua5', 'target UA 5\\%')]:
                v = g[f'{gate}_{key}']
                rows.append(f"{DS[ds]} & {NAMES[cond].split(', ')[1]} & {gname} & {lab} & {pct(v['known_coverage'])} & {pct(v['unknown_acceptance'])} & {pct(v['accepted_known_risk'])} & {pct(v.get('overall_accepted_error'))}\\\\")
        rows.append('\\midrule')
rows = rows[:-1]
tab = '\\begin{tabular}{llllrrrr}\\toprule\nData & Pool & Gate & Threshold & Coverage & Unknown acc. & Known risk & Overall error\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{Gate operating points with the PropGate class predictor (four-shot, 21 configurations; all values \\%). Quantile thresholds use only reference-set scores (supervised: out-of-fold MSP; graph: held-out-support propagation). Target-UA thresholds are set from unknown query scores and show what a labeled validation set containing unknown species would allow; overall error is not reported for them.}\\label{tab:s-gates}\n' + box(tab) + '\\end{table}\n')
# ---------------- S8 OOD-filtered self-training
L.append('\\FloatBarrier\n\\section{Out-of-distribution-filtered self-training: complete results}\\label{sec:s-ood}')
L.append('This section supports Section~3.3 of the main text. Filters are applied to the unlabeled pool before every refit; the supervised-MSP filter is recomputed with the current head, the graph-MSP and support-distance filters are computed once. All thresholds are 10th percentiles of held-out support scores. The oracle filter uses concealed labels. Kept known/unknown: share of known/unknown pool images passing the filter; Precision: weighted pseudo-label precision in the final round; Unk.\\ weight: share of final-round fitting weight on unknown images; Risk: accepted-known error under the shared supervised gate (all \\%).\n')
NM = {'lr': 'LR', 'lp': 'Iterative LP', 'fixed': 'Fixed (0.9)', 'fixed_msp': 'Fixed + supervised MSP', 'fixed_lpmsp': 'Fixed + graph MSP', 'fixed_dist': 'Fixed + support distance', 'fixed_oracle': 'Fixed + oracle',
      'soft': 'Soft', 'soft_msp': 'Soft + supervised MSP', 'soft_lpmsp': 'Soft + graph MSP', 'soft_dist': 'Soft + support distance', 'soft_oracle': 'Soft + oracle', 'fixed_t70': 'Fixed (0.7)', 'fixed_t95': 'Fixed (0.95)'}
for ds in ['tcmp', 'indian']:
    s = J(R / 'w2' / f'{ds}_summary.json'); a0 = s['rho0.0']['accuracy']; st = s['rho0.5']; rows = []
    for m in NM:
        rows.append(f"{NM[m]} & {f2(a0[m])} & {f2(st['accuracy'][m])} & {s['loss_0_to_50'][m]:.3f} & {ci(st['lp_minus'][m]) if m != 'lp' else '--'} & {pct(st['kept_known'].get(m))} & {pct(st['kept_unknown'].get(m))} & {pct(st['weighted_precision'].get(m))} & {pct(st['unknown_weight_fraction'].get(m))} & {pct(st['lr_gate_known_risk'][m])}\\\\")
    tab = '\\begin{tabular}{lrrrlrrrrr}\\toprule\nMethod & Acc.\\ 0\\% & Acc.\\ 50\\% & Loss & LP$-$method at 50\\% & Kept known & Kept unknown & Precision & Unk.\\ weight & Risk\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
    L.append(f'\\begin{{table}}[!htbp]\\centering\\small\n\\caption{{{DS[ds]}: out-of-distribution-filtered and threshold-varied self-training (four-shot fixed-known, 21 configurations at each level). The default LR, LP, fixed and soft rows reproduce the released results exactly.}}\\label{{tab:s-ood-{ds}}}\n' + box(tab) + '\\end{table}\n')

# ---------------- S9 zero-shot and fine-tuning
L.append('\\FloatBarrier\n\\section{Zero-shot and fine-tuning details}\\label{sec:s-alt}')
L.append('This section supports Section~3.4 of the main text. Zero-shot class embeddings average the L2-normalized text embeddings of the listed templates and are scaled by the checkpoint logit scale. TCMP binomials remove the class index and authority from the folder name (e.g., \\texttt{001.Veronica persica Poir.} $\\rightarrow$ ``a photo of Veronica persica.\'\'); Indian prompts replace underscores in folder names. Open-setting entries average 21 role/support configurations; the gate threshold is the 10th percentile of zero-shot MSP on support images.\n')
rows = []
VN = {'binomial': '``a photo of \\{binomial\\}.\'\'', 'binomial_authority': '``a photo of \\{binomial + authority\\}.\'\'', 'ensemble': 'four-template average', 'common_name': '``a photo of \\{name\\}.\'\'', 'common_name_leaf': '``a photo of a \\{name\\} leaf.\'\''}
for ds in ['tcmp', 'indian']:
    z = J(R / 'w3' / f'zeroshot_{ds}.json')
    for v, d in z.items():
        if not isinstance(d, dict): continue
        g = d['open_gate']
        rows.append(f"{DS[ds]} & {VN[v]} & {f2(d['closed_accuracy'])} & {f2(d['closed_macro_recall'])} & {f2(d['open_known_accuracy'])} & {pct(g['known_coverage'])} & {pct(g['unknown_acceptance'])} & {pct(g['accepted_known_risk'])} & {g['oscr']:.3f}\\\\")
tab = '\\begin{tabular}{llrrrrrrr}\\toprule\nData & Prompt & Closed acc. & Closed macro & Open known acc. & Coverage & Unknown acc. & Known risk & OSCR\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{BioCLIP 2 zero-shot classification (\\%). The four-template ensemble adds ``\\{name\\}, a medicinal plant\'\' and the bare name.}\\label{tab:s-zeroshot}\n' + box(tab) + '\\end{table}\n')
rows = []
for f in sorted(glob.glob(str(R / 'w3' / 'finetune_*.json'))):
    d = J(f); bb = 'BioCLIP 2 ViT-L/14' if 'bioclip' in d['backbone'] else 'ImageNet ResNet-50'
    rows.append(f"{bb} & {DS[d['dataset']]} & {d['seed']} & {f2(d['accuracy'])} & {f2(d['macro_recall'])} & {f2(d['worst_class_recall'])} & {d['train_seconds']/60:.1f} & {d['eval_seconds']/60:.1f} & {d['peak_gpu_memory_gb']:.1f} & {d['trainable_parameters']/1e6:.1f}\\\\")
tab = '\\begin{tabular}{llrrrrrrrr}\\toprule\nBackbone & Data & Draw & Acc. & Macro & Worst class & Train (min) & Query embed (min) & Peak GPU (GB) & Params (M)\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{End-to-end fine-tuning on clean closed four-shot supports (one Tesla P40, float32). Recipe fixed in advance (main text, Section~2.4); final epoch evaluated on all queries.}\\label{tab:s-finetune}\n' + box(tab) + '\\end{table}\n')

# ---------------- S10 encoders
L.append('\\FloatBarrier\n\\section{Other frozen encoders: complete results}\\label{sec:s-enc}')
L.append('This section supports Section~3.5 of the main text. DINOv2 uses the 1024-dimensional class token after the final layer norm (Hugging Face \\texttt{facebook/dinov2-large}); CLIP uses the OpenAI ViT-L/14 image embedding (768 dimensions, \\texttt{open\\_clip} model \\texttt{ViT-L-14-quickgelu}, i.e.\\ with the QuickGELU activation the OpenAI weights were trained with). Images use the same EXIF, resize, crop and float32 settings with each encoder\'s normalization; features are L2-normalized. All PropGate and control settings are unchanged. Entries are known accuracy (\\%); the BioCLIP 2 Indian rows were recomputed with the same code and match the released results exactly.\n')
MS = ['lr', 'fixed', 'soft', 'lp', 'one_step_lp', 'support_knn', 'ncm']; MN = ['LR', 'Fixed', 'Soft', 'Iter.\\ LP', 'One-step LP', 'Support 10NN', 'NCM']
rows = []
for enc, en in [('bioclip2', 'BioCLIP 2'), ('dinov2', 'DINOv2 ViT-L/14'), ('clip-quickgelu', 'CLIP ViT-L/14')]:
    for ds in ['tcmp', 'indian']:
        p = R / 'w5' / f'{enc}_{ds}_summary.json'
        if not p.exists(): continue
        s = J(p)
        for cond, cn in [('closed_rho0.0', 'closed'), ('open_rho0.0', 'open 0\\%'), ('open_rho0.5', 'open 50\\%')]:
            a = s[cond]['accuracy']; rows.append(f"{en} & {DS[ds]} & {cn} & " + ' & '.join(f2(a[m]) for m in MS) + '\\\\')
        rows.append('\\midrule')
rows = rows[:-1]
tab = '\\begin{tabular}{lll' + 'r' * 7 + '}\\toprule\nEncoder & Data & Pool & ' + ' & '.join(MN) + '\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{All seven predictors with three frozen encoders (four-shot; closed: 7 draws, open: 21 configurations). BioCLIP 2 TCMP values are those of the main text and Tables S1--S6.}\\label{tab:s-enc}\n' + box(tab) + '\\end{table}\n')
rows = []; prow = []
for enc, en in [('dinov2', 'DINOv2'), ('clip-quickgelu', 'CLIP')]:
    for ds in ['tcmp', 'indian']:
        s = J(R / 'w5' / f'{enc}_{ds}_summary.json'); o = s['open_rho0.5']
        for g, gn in [('lr_msp', 'LR MSP'), ('lp_msp', 'graph MSP')]:
            v = o['gates']['lp'][g]
            rows.append(f"{en} & {DS[ds]} & {gn} & {pct(v['known_coverage'])} & {pct(v['unknown_acceptance'])} & {pct(v['accepted_known_risk'])} & {v['oscr']:.3f}\\\\")
        prow.append(f"{en} & {DS[ds]} & {ci(o['lp_minus']['fixed'])} & {ci(o['lp_minus']['soft'])} & {ci(o['lp_minus']['lr'])} & {ci(s['fixed_minus_lp_loss'])}\\\\")
tab = '\\begin{tabular}{lllrrrr}\\toprule\nEncoder & Data & Gate & Coverage & Unknown acc. & Known risk & OSCR\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{Other encoders at 50\\% contamination: PropGate class predictor with both gates (\\%).}\\label{tab:s-enc-gates}\n' + box(tab) + '\\end{table}\n')
tab = '\\begin{tabular}{llllll}\\toprule\nEncoder & Data & LP$-$fixed at 50\\% & LP$-$soft at 50\\% & LP$-$LR at 50\\% & Fixed loss $-$ LP loss\\\\\\midrule\n' + '\n'.join(prow) + '\n\\bottomrule\\end{tabular}'
L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{Other encoders: paired differences in known accuracy (points) with 95\\% seed-block intervals (21 configurations).}\\label{tab:s-enc-paired}\n' + box(tab) + '\\end{table}\n')
dev = R / 'dev' / 'tcmp_dev_summary.json'
L.append('\\FloatBarrier\n\\section{Development-class sensitivity grid}\\label{sec:s-dev}')
L.append('This post hoc check repeats the $k\\times\\alpha$ grid and the fixed self-training threshold grid ($0.5$, $0.7$, $0.8$, $0.9$, $0.95$) on the 60 TCMP development classes, which are disjoint from the 240 evaluation classes, using a smaller version of the evaluation protocol. The original development runs that fixed the defaults are not replayed. Development images were enumerated from the training and validation folders of the TCMP-300 release; 10,415 readable images were used and one truncated file was excluded. Development images were not audited for near duplicates. Support draws use seeds 7001--7007 and role assignments seeds 70001--70003.\n')
if dev.exists():
    s = J(dev); rows = []
    for cond in ['closed_4shot_rho0.0', 'closed_16shot_rho0.0', 'open_4shot_rho0.0', 'open_4shot_rho0.5']:
        if cond not in s: continue
        o = s[cond]; a = o['accuracy']
        best_lp_label = o['best_lp'].replace('lp_k', '$k=').replace('_a', '$, $\\alpha=')
        rows.append(f"{NAMES[cond]} & {o['configurations']} & {f2(a['lr'])} & {f2(a['lp_k10_a0.9'])} & {f2(a['lp_k10_a0.8'])} & {best_lp_label}$ {f2(a[o['best_lp']])} & {f2(a['fixed_t0.9'])} & {f2(a['fixed_t0.7'])} & {o['best_fixed'].replace('fixed_t','')} {f2(a[o['best_fixed']])}\\\\")
    sel = s['selection_4shot']
    tab = '\\begin{tabular}{lrrrrlrrl}\\toprule\nCondition & $n$ & LR & LP ($k$10, $\\alpha$0.9) & LP ($k$10, $\\alpha$0.8) & Best LP & Fixed 0.9 & Fixed 0.7 & Best fixed threshold\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}'
    L.append('\\begin{table}[!htbp]\\centering\\small\n\\caption{Repeating the grid on the 60 TCMP development classes (30 known / 20 pool-unknown / 10 query-only per role, 3 roles, 7 support draws). Known accuracy (\\%). The selection criterion averaged the three four-shot conditions: best LP ' + sel['best_lp'].replace('lp_k', '$k=').replace('_a', '$, $\\alpha=') + '$, best fixed threshold ' + sel['best_fixed'].replace('fixed_t', '') + '.}\\label{tab:s-dev}\n' + box(tab) + '\\end{table}\n')

else:
    L.append('Pending.')
OUT.write_text('\n'.join(L)); print('wrote', OUT.relative_to(REPO))
if CHECK:
    ref = REPO / 'docs' / 'validation' / 'revision_tables_reference.tex'
    same = ref.read_text() == OUT.read_text()
    print('identical to', ref.relative_to(REPO) if same else 'DIFFERENT from reference', ref.name)
    sys.exit(0 if same else 1)
