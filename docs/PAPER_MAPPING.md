# Paper-to-file map

Numbering follows the manuscript submitted to *Plant Methods* (version of 22 October 2026) and its Additional file 1. "Offline" means the value can be regenerated from the saved results in this folder with `python scripts/run_offline_checks.py`; "re-run" gives the command that produces the saved results from images.

Abbreviations of locations: `ref/` = `results/reference/`, `rev/` = `results/revision/`, `analysis/` = `outputs/analysis/` (written by `scripts/analyze_results.py`).

## Main text

| Item | Content | Saved results | Offline check / regeneration | Re-run |
|---|---|---|---|---|
| Fig. 1 | Setting and evaluation scope (schematic) | — | `docs/figures/` (illustration; TCMP example photos, CC BY 4.0) | — |
| Fig. 2 | PropGate from a user's perspective (schematic) | — | `docs/figures/source/make_fig_workflow.py` | — |
| Fig. 3 | Data partitions and evaluation protocol | `data/manifests/`, `data/context_indices.zip`, `configs/experiment.json` | `verify_release.py` (counts, splits, 336 index sets) | — |
| Table 1 | Clean closed-set known accuracy | `ref/formal/results/closed_*` (TCMP), `ref/indian/results/closed_*` | `analyze_formal.py` → `outputs/formal_summary.json`; `analyze_results.py` → `analysis/derived_statistics.json` | `tcmp_repro.original`, `tcmp_repro.run --suite all` |
| Table 2, Fig. 4 | Fixed-known contamination gradient, losses 0–50% | `ref/{tcmp,indian}/results/*fixed_known*` | `analyze_results.py` → `analysis/figures/01_contamination_gradients`; `compare_analysis.py` | `tcmp_repro.run --suite gradients` |
| Fig. 5 | Component controls at 50% | same | `analysis/figures/02_component_ablation` | same |
| Section 3.3 text | Unknown share of pseudo-label weight, pseudo-label precision at 50% | `rev/w2/`; also `ref/{tcmp,indian}/results/` (`pseudo`) | `check_paper_numbers.py`; `analysis/figures/06_pseudo_label_contamination` | stage `w2` |
| Table 3 | OOD-filtered self-training | `rev/w2/` | `verify_revision.py`, `check_paper_numbers.py` | `revision/run_all.sh` stage `w2` |
| Table 4 | Zero-shot, fine-tuning, frozen-feature rows, compute | `rev/w3/`; frozen rows: `ref/formal/results/closed_s400[1-3]_n4_*`, `ref/indian/results/closed_s500[1-3]_n4.json` | `check_paper_numbers.py` | stages `zeroshot`, `finetune` |
| Section 3.4 cost paragraph | CPU timings of graph, propagation, self-training | `ref/{tcmp,indian}/results/*` (`timing` fields) | indicative wall-clock values; not checked | — |
| Table 5 | Other frozen encoders | `rev/w5/`; BioCLIP 2 TCMP row = `rev/w2/tcmp_summary.json` | `verify_revision.py`, `check_paper_numbers.py` | stage `w5` |
| Fig. 6 | Supervised vs graph-MSP gate along the gradient | `ref/{tcmp,indian}/results/` | `analysis/figures/03_rejection_tradeoffs` | `tcmp_repro.run` |
| Table 6, Section 3.6 text | Gate operating points at 50% | `rev/w4/<dataset>_summary.json` (`gates`) | `check_paper_numbers.py` | stage `w4` |
| Table 7, Section 3.7 text | k × α sensitivity; development grid | `rev/w4/`, `rev/dev/` | `check_paper_numbers.py` | stages `w4`, `dev` |
| Section 3.8 | Class-wise limits | `ref/{tcmp,indian}/results/` (per-class recall) | `analysis/derived_statistics.json` (class-tail summaries) | `tcmp_repro.run` |

## Additional file 1

| Item | Content | Saved results | Offline check / regeneration |
|---|---|---|---|
| Tables S1–S4 | Fixed-known accuracy and LP − fixed differences, all 21 configurations | `ref/{tcmp,indian}/results/` | `analyze_results.py` → `analysis/*_context_metrics.csv`, `derived_statistics.json` |
| Table S5 | Full-LP class predictor: gate trade-offs, unknown groups | same | same |
| Table S6 | Class-wise summaries at 50% | same | same |
| Table S7 | Final-round pseudo-label diagnostics | same | same |
| Fig. S1 | Fixed-total controls | `ref/{tcmp,indian}/results/*fixed_total*` | `analysis/figures/05_fixed_total_control` |
| Fig. S2 | Pseudo-label precision | same | `analysis/figures/06_pseudo_label_contamination` |
| Fig. S3 | External 60-class closed set | `ref/indian/results/closed_*` | `analysis/figures/04_external_closed_set` |
| Figs. S4–S5 | Algorithm schematics | implementation: `src/tcmp_repro/learner.py`, `evaluation.py` | — |
| Tables S8, S10–S13; Figs. S6–S7, S9–S11 | Original full-pool TCMP study (Clean-full, Add10, Add30, Matched30, Clean-thin) | `ref/formal/results/` (476 records), `ref/formal/curve_records.json` | `analyze_formal.py`; re-run `tcmp_repro.original` |
| Table S9, Fig. S6 | Support-only 1NN/3NN/NCM controls | `ref/review/results/`, `ref/review/artifacts/neighbor_records.json` | recorded summaries in `ref/review/artifacts/summary.json` |
| Table S15, Fig. S8 | Shared raw-LR gate, all pool conditions | `ref/review/artifacts/shared_gate_records.json` | recorded summary |
| Fig. S12 | Representative images | image IDs from the manifests | images reconstructed from the public dataset (not redistributed) |
| Table S14 | Frozen experiment settings | `configs/experiment.json`, `configs/original_formal.json` | — |
| Tables S16–S18 | k × α grids, gate operating points | `rev/w4/` | `make_revision_tables.py --check`, `check_paper_numbers.py` |
| Tables S19–S20 | OOD-filtered and threshold-varied self-training | `rev/w2/` | same |
| Table S21 | Zero-shot prompts | `rev/w3/zeroshot_*.json` | same |
| Table S22 | Fine-tuning per draw | `rev/w3/finetune_*.json` | same |
| Tables S23–S25 | Three encoders: all predictors, gates, paired differences | `rev/w5/`, `rev/w2/` | same |
| Table S26 | Development-class grid | `rev/dev/` | same |

## Where the defaults and seeds are fixed

`configs/experiment.json` holds the settings of the contamination study and the additional experiments: support seeds (TCMP 4001–4007, Indian 5001–5007), Indian role seeds (202610160–202610162), k = 10, α = 0.9 (4 labels) / 0.8 (16), fixed self-training thresholds 0.9 / 0.7, three self-training rounds, logistic C = 1000 and the bootstrap (seed 202610164, 20,000 draws). TCMP role seeds (202610130–202610132) are in `configs/original_formal.json` and, with the species of every role, in the manifests (`roles`). The gate quantile (10th percentile of out-of-fold MSP) is set in `src/tcmp_repro/evaluation.py` and `learner.py`. The development grid uses support seeds 7001–7007 and role seeds 70001–70003 (`revision/exp_dev_grid.py`).
