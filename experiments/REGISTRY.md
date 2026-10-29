# Experiment registry

Each run writes `results/<EID>/<run-id>/` and gets a row here.

| EID | Purpose | Status | Config | Key result file |
|---|---|---|---|---|
| E0 | PyCCE pilot: benchmarks, convergence, cost | **launched S010** (004-e0-bench, 005-e0-diamond on Actions); results pending | compute/requests/004-e0-bench.yaml, 005-e0-diamond.yaml | results/cce/004-e0-bench/, results/cce/005-e0-diamond/ (pending) |
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
| E3_synthetic | Runner check: E3 loop on synthetic grouped data (T-030). Not a result | runs (S005) | configs/experiments/E3_synthetic.yaml | not kept (tests/test_run.py) |

## Run log
| Run id | EID | Session | Purpose | Outcome |
|---|---|---|---|---|
| 000-envcheck | E0 | S002 | first pipeline test | workflow OK; artifact not downloadable from sandbox → added fallback branch |
| 001-envcheck | E0 | S002 | pycce probe | ImportError scipy.sparse.sputils → pinned numpy<2, scipy<1.12 |
| 002-envcheck | E0 | S002 | pycce probe | pandas 3 read_csv(delim_whitespace) error → pinned pandas<3 |
| 003-envcheck | E0 | S002 | pycce probe | **OK**: pycce 1.1.1 imports; toy diamond CCE ran (smoke only); see literature/notes/pycce_api.md |
| 004-e0-bench | E0 | S010 | 4H-SiC (mp-11714), alpha-quartz (mp-6930) vs Kanai Table S1; B along c; 2 shards | launched 2026-10-29, public run 40962961640 (sync 6609a15); pending. No token: pull with `scripts/fetch_cce_results.py` within 7 days |
| 005-e0-diamond | E0 | S010 | diamond (mp-66, conventional cubic) vs Kanai Table S1; B along [111]; 1 shard | **done** (fetched S011): T2 0.746 ms curve-avg (n 1.70), mean-of-fits 0.775 ± 0.030; converged at 65/20 Å (Δ 0.001 dex); 729 s. 14 % below Kanai's like-for-like curve fit 0.868 ms (n 2.38); see experiments/E0_notes.md |
| local, not kept | E3_synthetic | S005 | runner check (2 repeats × 5 folds, N ∈ {25, 50, max}, budget 10) | OK: all 4 models × 3 N, J = 10, 0 failures, ~2 s; synthetic numbers carry no meaning |
| 006-e0-diag | E0 | S011 | diamond shape diagnostics (B dir, r_dipole 8, gCCE, CCE-3) + origin_kanai placement on 3 hosts; 9 shards | **draft** (compute/drafts/), needs Kian's OK + sync |
