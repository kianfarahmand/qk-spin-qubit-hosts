# qk-spin-qubit-hosts

Code and data for **"Where the analytic spin-coherence law fails: a CCE benchmark and a controlled test of quantum kernels for spin-qubit host screening"** (Kian Farahmand, Department of Materials Science and Engineering, Shiraz University). *Manuscript in preparation.*

> **Status:** pre-release. The package and tests are public. Datasets (D-CCE), per-fold predictions and the pre-registration will be added at submission, with a Zenodo DOI.

## What is here
- `src/qkhosts/` contains:
  - the nuclear-spin-bath scaling law (Kanai et al. 2022)
  - bath descriptors
  - leakage-aware splits
  - a numpy statevector simulator with IQP, hardware-efficient and product-angle feature maps
  - fidelity and projected quantum kernels, with finite-shot and depolarising models
  - classical kernels
  - kernel diagnostics: geometric difference, concentration, alignment, effective dimension
  - KRR, GP and SVR models with equal-budget model selection
  - the pre-registered statistics: corrected resampled t-test, Holm correction, bootstrap
  - pool-based active learning
- `tests/`: unit tests, including dense-matrix and closed-form checks of the simulator.
- `configs/`: experiment settings.

## Install & test
```bash
pip install -e .[dev]
pytest -q
python -m qkhosts.run --config configs/experiments/E_smoke.yaml
```

## Reproducing the paper
Scripts and data for each experiment (E0–E8) will be listed in `experiments/REGISTRY.md` at release.

## Licence
Code is MIT (`LICENSE`). Data, once released, will be CC-BY-4.0 (`LICENSE-DATA`).

## Citation
See `CITATION.cff`.
