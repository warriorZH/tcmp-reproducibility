# PropGate — reproduction materials

[中文说明](README_zh.md) · [Paper-to-file map](docs/PAPER_MAPPING.md) · [Protocol](docs/PROTOCOL.md) · [Data](data/README.md) · [Validation](docs/VALIDATION.md) · [Environment](environment/README.md)

Code, locked data splits, saved results and verification scripts for

> Hu J, Zhang P, Li X, Wang Z. **PropGate: graph label propagation with a reference-calibrated acceptance gate for low-label medicinal plant recognition when unlabeled archives contain unknown species.** Manuscript submitted to *Plant Methods* (Methodology), 2026.

PropGate identifies medicinal plant photographs from a few labeled reference images per species plus an unlabeled image archive that may contain species outside the reference list. Images are embedded once with the frozen BioCLIP 2 encoder; labels are propagated over a k-nearest-neighbor graph of references and archive (k = 10, α = 0.9 at four labels per class, 0.8 at sixteen); a logistic-regression head fitted on the references alone sets an acceptance threshold (10th percentile of out-of-fold maximum softmax probability), so each query is either assigned a known species or referred to an expert.

Everything needed to check the paper's numbers runs on a laptop CPU in a few minutes. Re-running the experiments from images additionally needs the two public datasets and, preferably, a GPU.

## Contents

| Path | What it holds |
|---|---|
| `src/tcmp_repro/` | Core learners (LR, fixed/soft self-training, iterative and one-step LP, support kNN/NCM), gates, metrics, feature extraction and the experiment runners |
| `revision/` | Additional experiments reported in Sections 3.3–3.7: OOD-filtered self-training, zero-shot and fine-tuning baselines, k × α sensitivity, DINOv2/CLIP encoders, development-class grid; `run_all.sh` runs them all |
| `configs/` | Frozen settings (`experiment.json`: seeds, k, α, thresholds, bootstrap); BioCLIP 2 model config |
| `data/` | Locked manifests of both datasets (image IDs, relative paths, SHA-256, labels, similarity groups, support draws, species roles) and the 336 archive index sets of the contamination contexts. **No images.** |
| `results/reference/` | Saved results of the main contamination study: 350 per-configuration records (TCMP + Indian), 476 original TCMP records, review-stage controls |
| `results/revision/` | Saved results of the additional experiments: 497 per-configuration files, 14 zero-shot/fine-tuning runs, 10 summaries, feature metadata and run logs (server paths replaced by placeholders) |
| `scripts/` | Verification and analysis scripts (below) |
| `docs/` | Protocol, paper-to-file map, validation record, figure sources |
| `provenance/` | Original source snapshots and BioCLIP 2 checkpoint revision/hash |

## Level 1 — check every reported number (CPU, no data download)

Python 3.10–3.12.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\Activate.ps1
python -m pip install -e . "numpy==1.24.3" "scipy==1.11.2"
python scripts/run_offline_checks.py
```

The eight steps (report in `outputs/offline_check_report.txt`):

| Step | Script | Checks |
|---|---|---|
| 1 | `verify_release.py` | Manifest counts, disjoint train/query images and similarity groups, nested 4/16-shot supports, recomputation of all 336 locked archive index sets, 350 + 476 result records, file checksums |
| 2 | `smoke_test.py` | All seven classifiers × three gates on synthetic features (functional test, not a benchmark) |
| 3 | `analyze_formal.py` | Original TCMP closed/open summaries from 476 per-run records |
| 4 | `analyze_results.py` | Contamination gradients, component controls, acceptance and pseudo-label analyses; regenerates Figures 4–6 and the statistics behind Tables 1–2 |
| 5 | `compare_analysis.py` | Step 4 against the released reference statistics (894 values, tolerance 1e-10) |
| 6 | `verify_revision.py` | Every released summary of the additional experiments recomputed from its per-configuration files |
| 7 | `check_paper_numbers.py` | 308 values printed in Sections 3.3–3.7 and Tables 3–7 of the manuscript |
| 8 | `make_revision_tables.py --check` | Regenerates Additional file 1, Tables S16–S26 |

Which file and command supports each table and figure is listed in [docs/PAPER_MAPPING.md](docs/PAPER_MAPPING.md).

## Level 2 — re-run from images

### Data

Download the two public datasets (CC BY 4.0) as described in [data/README.md](data/README.md):

- TCMP-300 (Figshare, DOI 10.6084/m9.figshare.29432726): root folder containing `train/` and `val/`.
- Indian Medicinal Leaves Image Datasets, version 3 (Mendeley Data, DOI 10.17632/748f8jkphb.3): use the **Medicinal Leaf dataset** folder.

Confirm that the images are byte-identical to the ones used:

```bash
python scripts/verify_images.py --dataset tcmp --data-root /path/to/tcmp-300
python scripts/verify_images.py --dataset indian --data-root "/path/to/Medicinal Leaf dataset"
```

### Install

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[revision]" "numpy==1.24.3" "scipy==1.11.2"
python scripts/download_checkpoint.py --output checkpoints/bioclip-2   # revision and SHA-256 checked against provenance/model.json
```

### Main experiments (Sections 3.1–3.2, 3.6, 3.8; Tables 1–2; Figures 4–6)

```bash
python -m tcmp_repro.extract --dataset tcmp   --data-root /path/to/tcmp-300 --model-dir checkpoints/bioclip-2 --output outputs/features/tcmp   --device cuda:0 --batch-size 4
python -m tcmp_repro.extract --dataset indian --data-root "/path/to/Medicinal Leaf dataset" --model-dir checkpoints/bioclip-2 --output outputs/features/indian --device cuda:0 --batch-size 2
python -m tcmp_repro.run --dataset tcmp   --features outputs/features/tcmp/features.npy   --ids outputs/features/tcmp/ids.json   --output outputs/rerun/tcmp --suite gradients
python -m tcmp_repro.run --dataset indian --features outputs/features/indian/features.npy --ids outputs/features/indian/ids.json --output outputs/rerun/indian --suite all
python -m tcmp_repro.original --features outputs/features/tcmp/features.npy --ids outputs/features/tcmp/ids.json --output outputs/original_tcmp
python scripts/analyze_results.py --input outputs/rerun --output outputs/analysis_rerun
```

Run a single configuration first (`--seed 4001 --role 202610130 --rho 0.5 --max-contexts 1`) to check the set-up. `--device cpu` works for extraction but is slow.

### Additional experiments (Sections 3.3–3.5, 3.7; Tables 3–7; Additional file 1, Sections S7–S11)

```bash
TCMP_ROOT=/path/to/tcmp-300 INDIAN_ROOT="/path/to/Medicinal Leaf dataset" \
TCMP_FEATURES=outputs/features/tcmp/features.npy INDIAN_FEATURES=outputs/features/indian/features.npy \
OUT=outputs/revision bash revision/run_all.sh
```

`run_all.sh` is resumable (re-run it to continue) and accepts `STAGES="w2 w4 zeroshot w5 dev finetune"` to select stages; missing BioCLIP 2 features or checkpoint are extracted/downloaded. It checks that the features reproduce a released configuration before using them. Stages and outputs:

| Stage | Script | Paper | Output under `$OUT/` |
|---|---|---|---|
| `w2` | `exp_w2_ood_selftrain.py` | Table 3, Tables S19–S20 | `w2/<dataset>/`, `w2/<dataset>_summary.json` |
| `zeroshot` | `exp_w3_zeroshot.py` | Table 4, Table S21 | `w3/zeroshot_<dataset>.json` |
| `finetune` | `exp_w3_finetune.py` | Table 4, Table S22 | `w3/finetune_<backbone>_<dataset>_s<seed>.json` |
| `w4` | `exp_w4_sensitivity.py` | Tables 6–7, Tables S16–S18 | `w4/<dataset>/`, `w4/<dataset>_summary.json` |
| `w5` | `extract_features.py`, `exp_w5_encoders.py` | Table 5, Tables S23–S25 | `features/<encoder>_<dataset>/`, `w5/` |
| `dev` | `extract_features.py`, `exp_dev_grid.py` | Section 3.7, Table S26 | `features/bioclip2_tcmp_dev/`, `dev/` |

To compare a re-run with the paper, point the checks at it: copy `$OUT/{w2,w3,w4,w5,dev}` over `results/revision/` in a separate copy of this folder and run steps 6–8. On the reference machine (one Tesla P40, eight parallel CPU jobs) the complete additional suite took roughly a day, most of it fine-tuning and feature extraction for the other encoders.

## Known limitations

- **Fine-tuning is not bit-reproducible.** Random augmentation and GPU kernels change the last decimals of the fine-tuned accuracies across machines. Expect the means of Table 4 to agree within the reported standard deviations, not exactly.
- **CLIP activation (corrected).** The first CLIP run used open_clip's `ViT-L-14` configuration (GELU), although the OpenAI weights were trained with QuickGELU. The manuscript of 22 October 2026 (version 9) reports the re-run with `ViT-L-14-quickgelu` (`--encoder clip-quickgelu`, `revision/run_clip_quickgelu.sh`), which raised CLIP accuracies by 1.5–6.5 points and changed the TCMP comparison (propagation now ahead of LR and fixed self-training at 50%). The superseded GELU results are kept in `results/revision/superseded/clip_gelu/` for transparency.
- **Development images.** One truncated TCMP development image is excluded (`results/revision/features/bioclip2_tcmp_dev/excluded_images.json`); development images were not audited for near duplicates.
- **Images are not redistributed.** Image identity is guaranteed by SHA-256 in the manifests; if a public archive version changes, `verify_images.py` reports which files differ.
- **Not re-run end-to-end on a fresh machine.** All saved results were reproduced from saved records (Level 1); the image-to-result path (Level 2) was run once on the reference server.

## Citation and licenses

Cite the paper (details above; DOI to be added on publication) and the datasets and BioCLIP 2 as listed in `references.bib`. `CITATION.cff` describes this software release (version 0.2.1). Code and documentation: MIT licence. Saved numerical results: CC BY 4.0 with attribution to the study. Dataset metadata and model weights keep their original licences; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
