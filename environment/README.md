# Environment records

The experiment used PyTorch 2.7.1 with CUDA 11.8 on an NVIDIA Tesla P40 24 GB for frozen feature extraction. Logistic heads and sparse graph operations used CPU numerical libraries. An inspected review environment recorded Python 3.10.12, NumPy 1.24.3, SciPy 1.11.2 and scikit-learn 1.7.2. This is not a complete historical pip freeze for every original run.

`requirements-inspected.txt` records those numerical versions for Python 3.10. The default `pyproject.toml` dependency range also permits Python 3.11/3.12 and was used for the release's CPU validation; its exact validation versions are in `docs/validation_environment.json`.

Install GPU wheels explicitly before the `features` extra, as shown in README. Model extraction wrapper dependencies use compatible version ranges; the original open_clip version was not fully archived. Verify the image and checkpoint hashes before extracting. Use separate virtual environments for a reference stack and a modern compatibility check.

Prediction ties, BLAS implementation, solver behavior, image decoding and GPU kernels can affect exact arrays across environments. The package preserves the original neighbor handling and algorithms; it does not silently replace them with a different GPU graph implementation.
