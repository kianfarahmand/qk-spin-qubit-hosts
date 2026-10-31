# Experiment registry

Each run writes `results/<EID>/<run-id>/` and gets a row here.

| EID | Purpose | Status | Config | Key result file |
|---|---|---|---|---|
| E0 | PyCCE pilot: benchmarks, convergence, cost | **runs done S012** (004, 005, 006 fetched); ADR-0004 proposed (Q11 = A, S012); E0b 007-e0b done S015 (run 41037639805; T-065; §6 final); ADR-0004 proposed for acceptance S015 (Q14); 008-e0b-dilute drafted (Q13) | compute/requests/004-e0-bench.yaml, 005-e0-diamond.yaml, 006-e0-diag.yaml | results/cce/004-e0-bench/, 005-e0-diamond/, 006-e0-diag/; experiments/E0_notes.md |
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
| 004-e0-bench | E0 | S010 | 4H-SiC (mp-11714), alpha-quartz (mp-6930) vs Kanai Table S1; B along c; 2 shards | **done** (fetched S012): 4H-SiC 1.229 ms (n 2.72), 3 sites, spread 0.003 dex, converged 65/20 Å, 5,364 s; alpha-quartz 2.962 ms (n 2.73), 681 s. 0.91 / 0.87 of Table S1 (mean-of-fits); experiments/E0_notes.md |
| 005-e0-diamond | E0 | S010 | diamond (mp-66, conventional cubic) vs Kanai Table S1; B along [111]; 1 shard | **done** (fetched S011): T2 0.746 ms curve-avg (n 1.70), mean-of-fits 0.775 ± 0.030; converged at 65/20 Å (Δ 0.001 dex); 729 s. 14 % below Kanai's like-for-like curve fit 0.868 ms (n 2.38); see experiments/E0_notes.md |
| local, not kept | E3_synthetic | S005 | runner check (2 repeats × 5 folds, N ∈ {25, 50, max}, budget 10) | OK: all 4 models × 3 N, J = 10, 0 failures, ~2 s; synthetic numbers carry no meaning |
| 006-e0-diag | E0 | S011 | diamond shape diagnostics (B dir, r_dipole 8, gCCE, CCE-3) + origin_kanai placement on 3 hosts; 9 shards | **done** (run 40969623346, fetched S012): B ∥ [001] lifts diamond to 0.990 ms (n 2.62), matching Table S1 0.97(12); placement ≤ 0.010 dex; gCCE = CCE; CCE-3 +0.011 dex; r_dipole 8 Å +0.005 dex. `isotopes` empty (fixed S012). experiments/E0_notes.md §2 |
| 007-e0b | E0 | S013 | E0b at ADR-0004 option A: 3-direction spread on SiC/quartz/diamond, the other 5 Table S1 hosts (Si, CaO, CeO2, ZnO, MgO), alpha-Al2O3 dense bath at caps 750/1,500/3,000; single_most_spinful site; 15 shards | **done** S015: 15/15 ok (launched S013 with Kian's OK, public `89f398b`, run 41037639805); merged in results/cce/007-e0b/; CaO/CeO2 unusable (no floor; T-066, 008); experiments/E0_notes.md §6 (final) |
| 008-e0b-dilute | E0 | S015 | T-066 bath-size floor on the ultra-dilute Table S1 hosts: CaO, CeO2 x 3 directions at floor 1,000; CaO c at floors 500 / 2,000; 4 shards | **done** S015: 2/4 ok (Q13; public `58de29d`, run 41116974041); CeO2 label 43.8 ms (floor works); CaO floors 1,000/2,000 failed (LinAlgError), floor 500 ok; results/cce/008-e0b-dilute/; experiments/E0_notes.md §7 |
| 009-e0b-cao | E0 | S015 | CaO re-run with the T-068 failed-configuration rule + diagnostics: c, a+b, a+b+c at floor 1,000 (one per shard), c at floor 2,000 (10 conf); 4 shards | **launched** S015 with Kian's OK (Q16b; public `6de3ac0`, cce-batch run 41181177145); results pending |
