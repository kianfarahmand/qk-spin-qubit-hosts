# Experiment registry

Each run writes `results/<EID>/<run-id>/` and gets a row here.

| EID | Purpose | Status | Config | Key result file |
|---|---|---|---|---|
| E0 | PyCCE pilot: benchmarks, convergence, cost | not started | — | — |
| E1 | Reproduce Kanai scaling-law values | not started | — | — |
| E2 | Stratified CCE labels (D-CCE) | not started | — | — |
| E3 | Classical baselines, learning curves | not started | — | — |
| E4 | Quantum kernels, learning curves (H1) | not started | — | — |
| E5 | QK diagnostics | not started | — | — |
| E6 | Ablations | not started | — | — |
| E7 | Active learning (retro → prospective) | not started | — | — |
| E8 | Prospective screening + CCE validation | not started | — | — |
| E9 | (optional) hardware kernel | not started | — | — |
| E_smoke | Synthetic pipeline check | runs (S001) | configs/experiments/E_smoke.yaml | not kept |

## Run log
| Run id | EID | Session | Purpose | Outcome |
|---|---|---|---|---|
| 000-envcheck | E0 | S002 | first pipeline test | workflow OK; artifact not downloadable from sandbox → added fallback branch |
| 001-envcheck | E0 | S002 | pycce probe | ImportError scipy.sparse.sputils → pinned numpy<2, scipy<1.12 |
| 002-envcheck | E0 | S002 | pycce probe | pandas 3 read_csv(delim_whitespace) error → pinned pandas<3 |
| 003-envcheck | E0 | S002 | pycce probe | **OK**: pycce 1.1.1 imports; toy diamond CCE ran (smoke only); see literature/notes/pycce_api.md |
