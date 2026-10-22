# Validation of this release

## Version 0.2.1 (22 October 2026): corrected CLIP control

The CLIP encoder control was re-run on the reference server with open_clip `ViT-L-14-quickgelu` (the activation the OpenAI weights were trained with; the extraction logs no longer show open_clip's activation warning). Results are in `results/revision/w5/clip-quickgelu/` and replace the GELU run in Table 5, Tables S23–S25 and the text of the manuscript of 22 October 2026 (version 9); the GELU run is kept under `results/revision/superseded/clip_gelu/`. `verify_revision.py` reproduces the new summaries from their 98 per-configuration files, and `check_paper_numbers.py` now checks 308 values, including the corrected CLIP rows and the CLIP sentences of Section 3.5. All eight offline steps pass.

## Version 0.2.0 (22 October 2026): additional experiments of the Plant Methods manuscript

Checked in a fresh Python 3.10.20 virtual environment with `pip install -e . numpy==1.24.3 scipy==1.11.2` (scikit-learn 1.7.2), CPU only, by `python scripts/run_offline_checks.py`:

1. Steps 1–5 of version 0.1.0 (below) pass unchanged: 336 index sets, 350 + 476 result records, 894 reference statistics with maximum absolute difference 0.
2. `verify_revision.py`: all 10 released summaries of the additional experiments (w2, w4, w5, dev; 497 per-configuration files) are reproduced from their per-configuration files to within 1e-9. Wall-clock timing fields are not compared: three CLIP/BioCLIP 2 encoder summaries were written by an earlier run of the same stage and differ only in timing fields (8–12 per summary).
3. `check_paper_numbers.py`: 297 values printed in Sections 3.3–3.7 and Tables 3–7 of the manuscript are reproduced from the saved results at their printed precision. One printed value disagreed with the results in the manuscript of 21 October 2026 (Table 7, Indian, open 50%, "within 0.5 of best": printed 6/30, results 5/30, i.e. k = 5 with α = 0.7–0.99); it was corrected in the manuscript of 22 October 2026, against which all 297 values match.
4. `make_revision_tables.py --check`: Additional file 1 Tables S16–S26 are regenerated identically; every number in them appears in Additional file 1 of the submitted version.
5. File checksums: the version 0.1.0 `SHA256SUMS` had stale entries for `provenance/formal_reference/experiment.py.txt` and `src/tcmp_repro/original_experiment.py` (the files in the published repository do not match the recorded hashes, most likely because of a whitespace or line-ending change when they were committed; the function definitions of the archived snapshot and of the public module are identical by AST comparison). `SHA256SUMS` and `MANIFEST.json` are regenerated for this version.
6. Server paths in the revision logs and `inputs.json` are replaced by placeholders; `revision/find_inputs.py` now takes the dataset, feature and checkpoint locations from arguments or environment variables instead of searching fixed server folders.

The same eight steps also pass in a fresh Python 3.12.3 environment with the default dependency ranges (NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.7.2).

The additional experiments were run once from images on the reference server (`results/revision/logs/`); that image-to-result path was not repeated on a second machine for this release.

## Version 0.1.0

## Completed locally

1. Checked both dataset manifests: counts, unique/disjoint train/query IDs, disjoint similarity groups, class-balanced supports, and nested four-/sixteen-shot draws.
2. Recomputed all **336** locked fixed-known/fixed-total context index arrays and matched them exactly to archived arrays.
3. Checked the **350** dual-source result contexts, seven classifiers and common supervised-gate invariance; checked **476** original formal result records.
4. Exercised seven classifiers crossed with three gates on synthetic 768-dimensional features, including unknown-query diagnostics and prediction/metric serialization. The synthetic test is a functional test, not an accuracy benchmark.
5. Recomputed all paired summaries, long CSVs and plots from saved results. **894 numerical statistics match the reference exactly** (max absolute difference 0 in the recorded validation). Tolerance of the reusable comparison command is 1e-10.
6. Recomputed the TCMP original closed-table means/sample standard deviations from all per-seed raw JSONs and compared with its archived summary.
7. Compared archived and public core functions using Python ASTs: all learner functions, the five retained supplemental evaluation/aggregation functions, and the original formal experiment body are unchanged. Portable imports and orchestration are separate adaptations. See `core_source_checks.json`.
8. Parsed/compiled all public Python modules and checked the experiment CLI help entry points. The GitHub CPU workflow repeats the portable checks and analysis.

Machine-readable validation reports and exact CPU validation versions are stored beside this document. Saved-result checks do not fit the image encoder or change the paper results.

## Reproducibility boundary

The release was not independently rerun end-to-end from newly downloaded raw archives and model weights on a clean GPU machine. Download/extraction helpers and the full portable experiment commands are provided, but their full fresh-data/GPU path remains to be exercised by a reproducer. Checkpoint/image hash verification precedes feature computation. The original installation was not captured as a complete dependency lock; inspected numerical versions are documented separately.

No new raw-image labeling, new model selection, or formal-query threshold tuning was performed for this packaging task. Large prediction arrays and fitted caches are not redistributed; they are regenerated by the runners. The paper's results remain the archived reference outputs, not the synthetic tests or an unperformed fresh replication.
