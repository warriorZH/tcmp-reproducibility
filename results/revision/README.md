# Saved results of the additional experiments

Produced by `revision/run_all.sh` on the reference server (Python 3.10.12, NumPy 1.24.3, SciPy 1.11.2, scikit-learn 1.7.2, PyTorch 2.7.1 + CUDA 11.8, one Tesla P40). Server paths in logs and `inputs.json` are replaced by placeholders such as `<TCMP_ROOT>`.

| Folder | Experiment | Files | Paper |
|---|---|---|---|
| `w2/` | OOD-filtered and threshold-varied self-training, four-shot fixed-known gradient at 0% and 50% | 2 × 42 configurations + 2 summaries | Table 3; Tables S19–S20 |
| `w3/` | BioCLIP 2 zero-shot (prompt variants); end-to-end fine-tuning of BioCLIP 2 ViT-L/14 and ImageNet ResNet-50 on clean closed four-shot supports (first three draws) | 2 + 12 | Table 4; Tables S21–S22 |
| `w4/` | k ∈ {5,10,15,20,30} × α ∈ {0.5,…,0.99} grid and gate operating points (closed 4/16-shot, open 0%/50%) | 2 × 56 + 2 summaries | Tables 6–7; Tables S16–S18 |
| `w5/` | DINOv2 and CLIP ViT-L/14 (QuickGELU, folder `clip-quickgelu`) frozen features (TCMP, Indian) and BioCLIP 2 recomputed on Indian; all seven predictors and both gates | 5 × 49 + 5 summaries | Table 5; Tables S23–S25 |
| `superseded/clip_gelu/` | First CLIP run with open_clip `ViT-L-14` (GELU instead of the QuickGELU the weights were trained with); replaced in the manuscript of 22 Oct 2026, kept for transparency | 2 × 49 + 2 summaries | — |
| `dev/` | k × α and self-training threshold grid on the 60 TCMP development classes | 56 + 1 summary | Section 3.7; Table S26 |
| `features/` | Metadata, image IDs and feature SHA-256 of every extracted feature set (the feature arrays themselves are regenerated, not stored) | | |
| `logs/` | Run log of each stage, `status.txt`, environment summary | | |
| `inputs.json` | Inputs located by `find_inputs.py`, with the check that the BioCLIP 2 features reproduce a released configuration exactly (`lr_delta_vs_release` = 0) | | |

Apart from wall-clock timings, every `*_summary.json` is a deterministic function of the per-configuration files next to it; `python scripts/verify_revision.py` recomputes and compares them. In the `w2`, `w4` and BioCLIP 2 `w5` summaries, `max_abs_reference_delta` = 0.0: wherever a configuration also exists in the main study, the default-setting accuracies recomputed here equal the released results exactly (`null` where there is no counterpart, e.g. other encoders).
