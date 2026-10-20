# Results overview

All rows use the released complete role/support configurations. Accuracies are percent; losses/differences are percentage points.

| Source | Fixed loss 0–50% | LP loss 0–50% | LP minus fixed at 50% (95% conditional interval) |
|---|---:|---:|---|
| tcmp | 1.306 | 0.177 | 1.009 [0.787, 1.263] |
| indian | 1.484 | 0.106 | 0.776 [0.362, 1.129] |

The two sources support lower four-shot contamination losses for iterative LP than for the specified self-training controls. Clean-pool performance, sixteen-shot outcomes and class-tail behavior delimit this benefit. Classification changes and acceptance-score tradeoffs are evaluated separately. Full values, including unfavorable outcomes, are in the reference JSON/CSV files.

Reanalysis command: `python scripts/analyze_results.py`. Original TCMP table command: `python scripts/analyze_formal.py`.
