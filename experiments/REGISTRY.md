# Experiment registry

Definitions are in `proposal/proposal_v2.md` §10. Each run writes `results/<EID>/<run-id>/` and gets a row here.

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
