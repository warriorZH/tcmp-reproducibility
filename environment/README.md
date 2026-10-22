# Environment records

## Reference environment of the published runs

| Component | Version |
|---|---|
| Python | 3.10.12 |
| NumPy / SciPy / scikit-learn | 1.24.3 / 1.11.2 / 1.7.2 |
| PyTorch / torchvision | 2.7.1 (CUDA 11.8) / 0.22.1 |
| GPU (feature extraction, zero-shot text encoding, fine-tuning only) | one NVIDIA Tesla P40, 24 GB, float32 |
| CPU steps (graph, propagation, logistic heads, self-training) | four threads per configuration |

`requirements-inspected.txt` pins the numerical versions. The open_clip and transformers versions of the original runs were not archived; the declared ranges in `pyproject.toml` (`features` and `revision` extras) are the ones the code was written against.

## Recommended installs

CPU-only verification (no images, no GPU), about one minute of computation:

```bash
python -m pip install -e . "numpy==1.24.3" "scipy==1.11.2"
python scripts/run_offline_checks.py
```

This exact combination was tested in a fresh Python 3.10 virtual environment for this release (see `docs/VALIDATION.md`).

Full re-run from images (GPU recommended for feature extraction and fine-tuning):

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[revision]" "numpy==1.24.3" "scipy==1.11.2"
```

## Sources of numerical differences

Prediction ties, BLAS implementation, solver behaviour, image decoding and GPU kernels can change individual predictions across environments. Logistic-regression and graph steps are deterministic given the features; feature extraction and especially end-to-end fine-tuning (random augmentation, cuDNN kernels) are not bit-reproducible across GPUs. Wall-clock timings depend on the machine and its load and are indicative only.
