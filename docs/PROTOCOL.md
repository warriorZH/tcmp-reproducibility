# Experimental protocol and implementation

## Cohorts and labels

The TCMP manifest contains the 240 evaluation classes after the existing image audit. The separate 60 development classes determined the transferred hyperparameters; this release focuses on the fixed evaluation cohort rather than replaying all earlier exploratory directions. Every support seed has a class-balanced 16-image draw and its nested four-image subset. All remaining eligible training images are unlabeled to the learner. Query images remain outside the training graph.

The Indian manifest contains 60 folder classes, after deterministic candidate ordering, pixel/dHash grouping, cross-label-group exclusion and representative selection. It contains 3,955 training and 988 query images. Original folder names are class labels. Candidate/grouping summary is in `data/audits/data_audit.json`; per-image group IDs are in the manifest. Similarity groups are not specimen identities.

Each source has three open-role assignments: TCMP uses 120 known / 80 pool-unknown / 40 query-only-unknown classes; Indian uses 30 / 20 / 10. Each uses seven support seeds. Role IDs, class order and support ID order are fixed by the manifests. The random generator for pool ordering is NumPy `default_rng(role_seed + support_seed * 10000)`, permuting unknown indices before known indices.

## Pool families

The original TCMP protocol has Clean-full (all known unlabeled images), Add10/Add30 (same known images plus unknowns), Matched30 (approximately fixed total, fewer known images), and Clean-thin (the same reduced known subset without unknowns). Original raw records and code are separate from the capped-gradient family.

For each gradient role/support draw, let `Kstar=min(available_known_unlabeled, available_pool_unknown)`. Fixed-known experiments retain the same `Kstar` known images and add `round(Kstar*rho/(1-rho))` unknown images for `rho=0,.1,.2,.3,.4,.5`. Unknown prefixes are nested without replacement. Fixed-total controls at .3/.5 retain `round(Kstar*(1-rho))` known images plus enough unknowns to total `Kstar`. Thus capped 0% is different from Clean-full. Exact index arrays are in `data/context_indices.zip`; the verifier recomputes all of them.

## Learning and prediction

- Frozen BioCLIP 2 visual features: 768 dimensions, float32 extraction, L2 normalization. Exact checkpoint metadata is in `provenance/model.json`.
- LR: multinomial logistic regression, intercept, L-BFGS, `C=1000`, `tol=1e-5`, `max_iter=1500`. Four support-only stratified folds, random state 5521, generate OOF logits. A temperature is chosen from `.25,.5,1,2,4` by support OOF NLL for self-training.
- Fixed/soft self-training: three rounds; fixed thresholds .9/.7 at four/sixteen-shot; soft weights use the recorded Gaussian confidence weighting. Pool fitting weight is rescaled to equal the total support weight.
- Graph: ten neighbors, cubed nonnegative cosine weights, max-symmetrization and row normalization. The retained implementation requests 11 training neighbors and drops the first returned item. Identical-feature ties can leave rare self-edges; the released kernel preserves this behavior to match the study. It is not described as strict exclusion of every self-neighbor.
- LP: seed reinjection, support clamping; alpha .9/.8 for four/sixteen-shot; maximum 100 steps or max absolute change below 1e-6. Rows without propagated mass use a uniform fallback. Queries combine ten training neighbors inductively.
- Controls: one propagation update with the same normalization/extension, support 10NN with cubed cosine weighting, and normalized nearest-class-mean classification.

## Acceptance and diagnostics

Each predictor crosses all-accept, raw LR MSP and graph MSP rules. The LR threshold is the tenth percentile of support OOF maximum probabilities. The graph threshold uses support-fold held-out-node propagation; query graph scores use the inductive extension. These are different score-generation mechanisms and remain unchanged in the reproduction.

Pool/query truths are present in released metadata for reproducibility, but enter the evaluator only after predictions exist. `learner.py` accepts only features, support indices/labels and numerical settings. Confidence-derived loss/selection does not read pool truth. Query truths support evaluation, pseudo-label diagnostics and retrospective risk–coverage curves; they do not set deployed thresholds.

## Statistics

The paired analysis uses 20,000 shared-support-seed block resamples, seed 202610164, resampling all three fixed roles together. Intervals condition on the fixed images, roles and model; the 21 configurations are not independent external datasets. Accuracy/macro recall are stored as percentages in context records, while coverage/risk/unknown acceptance are fractions. Plot/report generation converts units where needed.

## Code provenance

`learner.py` is the archived formal learner with only a relative import adaptation. `evaluation.py` retains the archived supplemental metric, one-step, nearest-neighbor, evaluation and aggregation function bodies; its server main is replaced by the portable orchestrator. `original_experiment.py` retains the original 476-record experiment body with package imports. Original-source hashes and the adapter description are in `provenance/`. Portable CLI, downloading, extraction and validation helpers were added for this release; they are not evidence of a separate full replication.
