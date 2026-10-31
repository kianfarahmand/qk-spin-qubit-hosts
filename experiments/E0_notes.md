# E0 PyCCE pilot: results and analysis (T-010, S011–S012)

Runs: `004-e0-bench` (public Actions run 40962961640; 4H-SiC, alpha-quartz; B || c), `005-e0-diamond`
(run 40963132024; B || [111]) and `006-e0-diag` (run 40969623346; 9 one-change variants, single 50/15 Å rung).
Base protocol: S = 1/2, g = 2, B = 5 T, CCE-2 (`method cce`), Hahn echo with t = total time
(`as_delay=False`), 20 isotope configurations, vacancy on up to 3 inequivalent host sites
(multiplicity-weighted log-mean), ladder 50/15 -> 65/20 -> 80/25 Å with |Δlog10 T2| < 0.02 to stop.
Raw records: `results/cce/<id>/merged.jsonl`. Every number below comes from those files,
`data/external/kanai2022/figureS2_Hahn.csv`, or Kanai Table S1 (`literature/notes/kanai2022_si.md`).

## 1. Per-material T2 against Kanai (base protocol, converged at 65/20 Å)

"curve-avg" = fit of the configuration-averaged curve (our `t2_ms`); "mean-of-fits" = mean of the
per-configuration fits. Kanai's curve-avg value is our `fit_t2` on their averaged curve in `figureS2_Hahn.csv`.

| Host | ours curve-avg (n) | ours mean-of-fits ± SEM | Kanai curve-avg fit (n) | Kanai Table S1 (n) | ours / Kanai: curve / fits |
|---|---|---|---|---|---|
| diamond (B ∥ [111]) | 0.746 ms (1.70) | 0.775 ± 0.030 | 0.868 (2.38) | 0.97(12) (2.64) | 0.86 / 0.80 |
| alpha-quartz (B ∥ c) | 2.962 ms (2.73) | 2.96 ± 0.03 | 3.289 (2.69) | 3.42(15) (2.59) | 0.90 / 0.87 |
| 4H-SiC (B ∥ c) | 1.229 ms (2.71–2.73) | 1.228 ± 0.017 | 1.295 (2.68) | 1.35(9) (2.80) | 0.95 / 0.91 |

4H-SiC per site (004 shard 0, 65/20 Å): Si (z 0.938) 1.228 ms, Si (z 0.688) 1.233, C (z 0.750) 1.225;
site spread 0.0029 dex. Ladder 50/15 -> 65/20 moved every site by -0.0023 dex. Nearest bath spin from the
C-vacancy centre 1.88 Å (a 29Si on a nearest-neighbour Si site), from the Si-vacancy centres 3.08 Å. Cost 5,364 s for 3 sites x 2 rungs (50/15: 292–319 s per site; 65/20: 1,480–1,491 s per site).

## 2. 006-e0-diag: one-change variants (single 50/15 Å rung unless noted)

Per-config SD is the spread of the 20 per-configuration T2 fits (compare Kanai's SD in Table S1).
Wall time is per site on one Actions runner (2 vCPU).

| Variant | change | curve-avg T2 (n) | mean-of-fits ± SEM (SD) | mean per-config n | NLS T2 (n) | wall s |
|---|---|---|---|---|---|---|
| diamond-base | B ∥ [111] (= 005 rung 0) | 0.748 (1.70) | 0.778 ± 0.030 (0.133) | 2.26 | 0.771 (1.96) | 162 |
| **diamond-z** | **B ∥ [001]** | **0.990 (2.62)** | **0.989 ± 0.022 (0.099)** | **2.73** | 0.990 (2.61) | 112 |
| diamond-rd8 | r_dipole 8 Å | 0.757 (1.69) | 0.790 ± 0.030 (0.134) | 2.26 | 0.780 (1.95) | 55 |
| diamond-gcce | method gcce | 0.748 (1.70) | identical to base to 8 digits | 2.26 | identical | 212 |
| diamond-cce3 | CCE-3, r_dipole 8 Å | 0.777 (1.65) | 0.798 ± 0.032 (0.144) | 2.17 | **fit failed** (24 ms, n at bound 6) | 89 |
| diamond-origin | Kanai origin (C–C bond centre, no vacancy) | 0.765 (1.99) | 0.776 ± 0.034 (0.152) | 2.39 | 0.772 (2.08) | 113 |
| quartz-base | B ∥ c (= 004 shard 1 rung 0) | 3.026 (2.71) | O 3.031, Si 3.015 ± 0.032 (0.142) | 2.73 | 3.03 / 3.02 | 64–89 |
| quartz-origin | Kanai origin (empty point) | 3.023 (2.69) | 3.023 ± 0.029 (0.132) | 2.71 | 3.019 (2.67) | 80 |
| sic-origin | Kanai origin (= C vacancy, C4 site) | 1.252 (2.70) | 1.252 ± 0.021 (0.092) | 2.75 | 1.253 (2.69) | 315 |

Reference at the same 50/15 rung: 4H-SiC vacancy sites 1.231–1.240 ms (004 rung 0).

## 3. Findings

1. **Field direction is the dominant protocol knob in diamond.** B ∥ [001] instead of [111] raises T2 by
   32 % (0.748 -> 0.990 ms, +0.121 dex) and moves the averaged-curve n from 1.70 to 2.62. With B ∥ [001]
   our diamond reproduces Kanai Table S1 like for like: mean-of-fits 0.989 ± 0.022 vs 0.97(12) ms,
   per-config n 2.73 vs 2.64(29), per-config SD 10 % vs 12 %. With B ∥ [111] it is 20 % low, the SD is 17 %,
   and the per-config n is 2.26. Physical reading (not yet tested): for B ∥ [001] all four C–C bond
   directions sit at the magic angle (54.7°), so nearest-neighbour 13C pairs have no secular flip-flop
   coupling; for B ∥ [111] one bond in four lies along B and those strongly coupled close pairs add a fast,
   configuration-dependent decay that lowers n and widens the spread.
2. **Which direction Kanai used for Table S1 is not settled by the shipped code.** Kanai's CIF loader
   (`load_crystal.m`, `otherwise` branch) puts a ∥ x and b in the xy plane, then rotates `H_direction`
   onto z. For the MP primitive cell of mp-66 (a = b = c = 2.518 Å, all angles 60°), a × b lies along a cubic
   ⟨111⟩ (computed from `data/external/mp_optimade/mp-66.json`); for mp-11714 and mp-6930 it is c. So if
   `H_direction = [0 0 1]`, Kanai's diamond had B ∥ ⟨111⟩, which matches the Fig. 2A caption ("[111] for
   diamond") and our 005 set-up, yet B ∥ [001] is what reproduces Table S1. The script that sets
   `structdata.hdirection` (`batch_density_CCE_v0_from_cif`) is not in the Zenodo record. Side note:
   for H_direction ≠ [0 0 1] that branch rotates the atom positions twice (`r.base` is rotated, then
   `r.index` is built from it and rotated again), so any non-z setting there would distort the lattice.
   This is a question for the CCE collaborator (ADR-0006) or the authors.
3. **Placement rule matters little.** Kanai's origin rule vs our vacancy average: diamond +0.010 dex
   (0.765 vs 0.748), 4H-SiC +0.004 to +0.007 dex against the 50/15 vacancy sites, quartz within 0.001 dex.
   Vacancy site spread within a host: 4H-SiC 0.0029 dex, quartz 0.0021 dex. Both are an order of
   magnitude below the diamond field-direction effect (0.121 dex). The origin rule depends on the CIF's
   arbitrary origin, so it is not reproducible across structure sources.
4. **gCCE = CCE for this central spin.** For S = 1/2 at 5 T the gCCE result equals CCE to 8 digits and
   costs 1.3x more. CCE-2 is converged in order to 0.011 dex (CCE-3 vs CCE-2 at r_dipole 8 Å: 0.777 vs
   0.757 ms), within the 0.02 dex tolerance.
5. **Cut-offs.** r_dipole 15 -> 8 Å changes T2 by +0.005 dex at a third of the cost (55 vs 162 s);
   r_bath/r_dipole 50/15 -> 65/20 changes it by -0.001 (diamond), -0.002 (SiC), -0.009 dex
   (quartz) at 3.3–4.8x the cost. 50/15 holds about 1,000 bath spins inside r_bath here (diamond 994,
   quartz 617–622, 4H-SiC 1,433), matching Kanai's "~1,000 interacting nuclear spins".
6. **Estimator.** When the per-config spread is small (quartz, SiC, diamond with B ∥ [001]) the
   averaged-curve fit and the mean of fits agree within 0.1 %. With B ∥ [111] they differ by 4 %, and the
   averaged curve gets a lower n because it mixes configurations with different T2. The linearised fit
   is robust; the NLS fit failed once (CCE-3: 24 ms with n at its upper bound), so NLS stays a diagnostic.
7. **Residual offset.** With the direction that matches diamond, SiC and quartz still sit 9 % and 13 %
   below Table S1 (5 % and 10 % below Kanai's averaged-curve fit). An offset of about -0.04 to -0.06 dex
   that is uniform across hosts would be absorbed by the intercept of a Δ model; a host-dependent one would
   not. Candidates: isotope abundances (now recorded, see §4), Kanai's unknown cut-offs, the Fig. S2 curves
   coming from a different configuration set than Table S1. E0b (below) adds 5 more Table S1 hosts.
8. **Reproducibility.** 006 repeated 005 rung 0 (diamond 0.748 ms) and 004 rung 0 (quartz O 3.0315 ms,
   Si 3.0157 ms) to all printed digits: the paired seeds make runs deterministic across Actions jobs.
9. **Cost (one site, 50/15 Å, 20 configurations, one Actions runner):** diamond 112–162 s, quartz 64–89 s,
   4H-SiC 292–319 s. All E0 hosts have dilute baths (1–5 % spinful isotopes). Hosts with 100 %-abundant
   spinful isotopes (1H, 19F, 27Al, 31P, 23Na, 51V, ...) have up to about 100x more spins inside 50 Å than diamond; their cost is
   untested and must be capped by a bath-size rule (ADR-0004 open item).

## 4. Real-PyCCE check points (T-010 notes)

| Check | Status |
|---|---|
| `gen_supercell(remove=(el, frac))` semantics | Works: nearest bath spin 3.88 Å (diamond), 4.38–5.01 Å (quartz), 1.88–3.08 Å (4H-SiC); no spin on the centre. |
| centre position (`to_cartesian`; `min_bath_dist_A > 0`) | OK. |
| PyCCE isotope table vs Kanai `isotopetable.csv` | **Not recorded in 006**: `cell.isotopes` is `{}` unless isotopes are added by hand. Fixed in S012: the runner now falls back to `pycce.common_concentrations` for the host's elements (`_isotopes`, test `test_isotopes_fall_back_to_common_concentrations`). Comes with the next request. Kanai: 13C 1.07 %, 29Si 4.6832 %, 17O 0.038 %. |
| field frame | Our runner: b_direction is given in the frame of the structure file (conventional cubic cell for diamond). See finding 2 for Kanai's frame. |

## 5. Decision input for ADR-0004 and next request (E0b, draft in S013)

ADR-0004 (proposed, S012) takes: one vacancy site, CCE-2 `cce`, 50/15 Å, 20 configurations, curve-average
estimator, and asks Kian to choose the field-direction rule. E0b request `007-e0b` should test, at that protocol:
(a) the field-direction spread on 4H-SiC and quartz (c, a+b, a+b+c of the conventional cell) and diamond
[110]; (b) the 5 remaining Table S1 hosts (Si mp-149, MgO mp-1265, ZnO mp-2133, CaO mp-2605,
CeO2 mp-20194) with the isotope record; (c) one dense-bath host (e.g. alpha-Al2O3 mp-1143, 27Al 100 %) with
a bath-size cap, to measure the worst-case cost.

**S013 (T-065): request drafted.** `compute/drafts/007-e0b.yaml` (15 shards, one item each; not launched). Runner
support added in `src/qkhosts/labels/pycce_runner.py` and tested only with the fake PyCCE (`tests/test_pycce_runner.py`):
- `b_directions_lattice`: field along lattice combinations of the CIF cell (all E0/E0b CIFs are conventional cells);
  per-direction records under `sites[i].directions`, host `direction_t2_ms`, `direction_spread_log10`; label = log-mean.
- `site_rule: single_most_spinful` (`pick_site`): element with most spinful nuclei per cell (PyCCE abundances,
  recorded as `spinful_fraction`), then its highest-multiplicity site, ties to the first. Refuses to run if PyCCE's
  abundance table is not found, rather than picking blindly.
- `max_bath_spins` (`cap_radii`): if more spins sit inside r_bath than the cap, r_bath shrinks to enclose `cap`
  spins and r_dipole shrinks by the same factor (pairs per spin stay as in a dilute bath). Recorded:
  `capped`, `r_bath_used_A`, `r_dipole_used_A`, `n_bath_in_rbath_nominal`. Baths are now generated one at a time.
- E0b structures: `compute/structures/e0b/` (Si mp-149, MgO mp-1265, CaO mp-2605, CeO2 mp-20194 as conventional
  cubic cells; ZnO mp-2133 hexagonal; alpha-Al2O3 mp-1143 rhombohedral -> 30-site hexagonal setting); DATA_CARD row
  "MP OPTIMADE E0b structures".
- Points to check on the first real run: `spinful_fraction` matches the PyCCE isotope record; `b_cart_unit` of
  c is (0, 0, 1), and a+b of a hexagonal cell is in-plane, 60 deg from a (`test_field_directions_use_lattice_columns`);
  capped radii for MgO and Al2O3 near the
  pre-launch estimates (MgO ~41/12 Å, Al2O3 ~20/6 Å; rough estimates from density x abundance, not results).

**S013 early shards (cao-A, ceo2-A; `results/cce/007-e0b-early/`): new runner paths work on real PyCCE, but
ultra-dilute baths break the fixed 50 Å protocol.**
- Works: `single_most_spinful` picked Ca (multiplicity 4) in CaO and O in CeO2; `spinful_fraction` and the
  isotope record come from PyCCE's table (43Ca 0.00135, 17O 0.00038; Ce none), which also confirms the S012
  `isotopes` fix; `b_cart_unit` is (0,0,1), (0.707,0.707,0), (0.577,0.577,0.577) for the cubic cells; no cap.
- Problem: only 30 (CaO) and 9 (CeO2) spins sit inside r_bath = 50 Å (`n_bath_in_rbath`), against Kanai's ~1,000.
  Coherence curves of a handful of spins do not follow a stretched exponential: per-configuration fits fail
  (CaO 1 of 20 per direction, CeO2 13–16 of 20), CaO's stretch n is 0.6–1.9 and its curve-fit and mean-of-fits
  T2 disagree, and CeO2's fitted T2 values are meaningless (1e-116 to 1e149 ms). The log-mean accepted them
  because they are positive and finite. These two hosts give no usable E0b number from this run.
- Needed before E2 (new task T-066): (1) a bath-size **floor**, the mirror of the cap: grow r_bath (and the supercell,
  at least 2 r_bath) until about 1,000 spins sit inside; (2) a fit guard: NaN when n is outside [0.5, 6], T2 lies
  outside the sampled window, or the mean curve has fewer than `min_fit_points` usable points; the host label
  is NaN (flagged) rather than a log-mean of bad fits; (3) re-run CaO and CeO2 in a later request.

## 6. E0b results (007-e0b, run 41037639805): final (S015)

**Source:** `results/cce/007-e0b/merged.jsonl` (15 of 15 records ok; `provenance.json`: artifact `cce-007-e0b-merged`,
public `89f398b`, fetched S015 with `python scripts/fetch_cce_results.py 007-e0b`). Tables from
`python scripts/analyse_e0b.py results/cce/007-e0b`. S015 check: every number of the S014 interim section (taken
from the per-shard artifacts) is identical in the merged file; the only new record is `al2o3-c-cap3000`.

Per-direction T2 (curve-avg fit, ms, stretch n; 20 configurations; single 50/15 Å rung unless capped):

| host (site) | c | a+b | a+b+c | label (log-mean) | spread dex | spins in r_bath | wall s per direction |
|---|---|---|---|---|---|---|---|
| 4H-SiC (Si) | 1.23 (2.71) | 1.20 (2.76) | 1.24 (2.73) | 1.22 | 0.016 | 1,404 | 166–183 |
| alpha-quartz (Si) | 3.02 (2.70) | 2.48 (2.60) | 2.07 (2.59) | 2.49 | 0.16 | 622 | 60–81 |
| diamond (C) | 0.990 (2.62) | 0.787 (1.75) | 0.748 (1.70) | 0.835 | 0.12 | 994 | 134–159 |
| Si (Si) | 1.80 (2.64) | 1.11 (2.67) | 1.03 (2.47) | 1.28 | 0.24 | 1,184 | 90–109 |
| ZnO (Zn) | 2.00 (2.60) | 2.05 (2.54) | 2.16 (2.61) | 2.07 | 0.033 | 911 | 3,368–4,040 |
| MgO (Mg), capped 1,500 of 2,930 (r 40.0/12.0 Å) | 0.997 (2.74) | 0.872 (2.71) | 0.874 (2.76) | 0.913 | 0.058 | 1,500 | 6,048–8,915 |
| CaO, CeO2 | no usable fits (30 and 9 spins inside 50 Å; S013, T-066; re-run drafted as 008) | | | | | | |

alpha-Al2O3 (Al site, B ∥ c only, 5 configurations; 27Al 100 %, 23,982 spins inside the nominal 50 Å):

| cap | r_bath/r_dipole Å | T2 µs (n) | Δ vs cap 3,000 dex | wall s | s per configuration |
|---|---|---|---|---|---|
| 750 | 15.7/4.7 | 14.1 (2.58) | +0.019 | 520 | 104 |
| 1,500 | 19.9/6.0 | 13.6 (2.61) | +0.003 | 3,643 | 729 |
| 3,000 | 25.0/7.5 | 13.5 (2.62) | 0 | 13,178 | 2,636 |

Findings:
1. **Kanai's field frame (supersedes the S012 reading of findings 1/2).** Kanai's code puts B along
   z = a x b of the MP primitive cell (if `H_direction = [0 0 1]`). From `data/external/mp_optimade/*.json`: the fcc
   primitives (mp-66, mp-149, mp-1265, mp-2605, mp-20194: a = b = c, all angles 60 deg) give **<111>**; the hexagonal cells
   (mp-11714, mp-6930, mp-2133) give **c**. Si settles it: Si [001] gives 1.80 ms vs Table S1 1.24(23), while Si [111]
   gives 1.03 ms (mean-of-fits 1.04, 0.84 of S1; 0.98 of Kanai's averaged-curve fit 1.06). Si and diamond share one
   structure and one frame in Kanai's code, so diamond's [001] match to Table S1 (0.989 vs 0.97) was a coincidence;
   the like-for-like diamond value is [111] (0.80 of S1).
2. **Offsets vs Table S1 in Kanai's frame (mean-of-fits / S1):** diamond 0.80, Si 0.84, ZnO 0.86, quartz 0.88,
   4H-SiC 0.91, MgO 0.97 (-0.10 to -0.01 dex; all below Kanai). Against Kanai's own averaged curves
   (`figureS2_Hahn.csv`, our `fit_t2`; curve / curve): 0.86, 0.98, 0.90, 0.92, 0.95, 1.01. Not uniform (0.08 dex range), but small
   next to the direction spread. With B along [001] for every host the ratios spread 0.86–1.45 instead.
   Diamond's shape stays off in Kanai's frame (n 1.70 vs Kanai curve 2.38): the open question for the CCE collaborator.
3. **Direction spread supports option A:** 0.016 (SiC) to 0.24 dex (Si). Cubic hosts are not isotropic: Si and
   diamond [001] are the magic-angle direction for all nearest-neighbour bonds (E0 finding 1); MgO 0.058 dex.
4. **Isotopes:** PyCCE `common_concentrations` equal Kanai `isotopetable.csv` for every recorded spinful isotope
   (13C, 29Si, 17O, 43Ca, 25Mg, 67Zn, 27Al; ratio 1.0000). ADR-0004 item 8 closed.
5. **Dense-bath cap (alpha-Al2O3):** cap 1,500 is within 0.003 dex of cap 3,000; cap 750 is +0.019 dex, at the edge of
   the 0.02 tolerance. Cost per configuration grows about 7x (750 -> 1,500) and 3.6x (1,500 -> 3,000). **Cap 1,500** is the
   converged choice. The 5 configurations agree to SEM ~3e-10 ms at every cap: with 27Al at 100 % only rare 17O varies,
   so **one configuration is enough** for such baths.
6. **Cost per nuclear-spin class** (per direction, one Actions runner, from the wall times above):
   I = 1/2 dilute baths (13C, 29Si; 622–1,404 spins, 20 configurations): 1.0–3.1 min (3–9 s per configuration).
   I = 5/2 baths, 20 configurations: ZnO (67Zn, 911 spins) 56–67 min (168–202 s per configuration); MgO (25Mg, cap 1,500)
   101–149 min (302–446 s per configuration). I = 5/2 dense, 1 configuration at cap 1,500 (Al2O3): about 12 min.
   I > 1/2 costs 20–60x more per configuration than I = 1/2 at similar bath size; the ADR-0004 E2 budget is redone per
   class there (S015). SEM of the mean-of-fits for the I = 5/2 hosts at 20 configurations: 0.005–0.009 dex
   (ZnO c 0.0085, MgO a+b 0.0090, a+b+c 0.0050), so 10 configurations would give about 0.007–0.013 dex.

**T-066 (S014):** runner floor + fit guard built (`floor_radii`, `guard_fit`, strict log-mean, `label_ok`; tests in
`tests/test_pycce_runner.py`); CaO/CeO2 re-run drafted as `compute/drafts/008-e0b-dilute.yaml` (floor 1,000, plus CaO c
at floors 500 / 2,000), not launched.

## 7. 008-e0b-dilute (run 41116974041): bath-size floor on CaO and CeO2 (S015)

**Source:** `results/cce/008-e0b-dilute/merged.jsonl` (4 records, 2 ok; public `58de29d`; fetched S015 with
`python scripts/fetch_cce_results.py 008-e0b-dilute`); tables from `python scripts/analyse_e0b.py results/cce/008-e0b-dilute`.
Protocol as 007 plus `min_bath_spins` (runner at public `58de29d`, before T-067/T-068).

| item | B | floor | spins in r_bath (nominal 50 Å) | r_bath/r_dipole Å | T2 curve ms (n) | mean-of-fits ± SEM ms | failed per-conf fits | wall s |
|---|---|---|---|---|---|---|---|---|
| ceo2-A (O site) | c | 1,000 | 1,000 (12) | 234.6/70.4 | 42.8 (1.38) | 31.3 ± 3.5 | 7/20 | 5,281 |
| ceo2-A | a+b | 1,000 | 1,000 (12) | 234.6/70.4 | 43.6 (1.45) | 37.5 ± 2.4 | 4/20 | 5,231 |
| ceo2-A | a+b+c | 1,000 | 1,000 (12) | 234.6/70.4 | 45.1 (1.35) | 32.0 ± 3.2 | 9/20 | 5,283 |
| cao-c-floor500 (Ca site) | c | 500 | 500 (28) | 123.8/37.1 | 26.7 (2.15) | 26.6 ± 1.9 | 0/20 | 1,759 |
| cao-A | c, a+b, a+b+c | 1,000 | — | — | **failed**: `LinAlgError: Eigenvalues did not converge` | | | 205 |
| cao-c-floor2000 | c | 2,000 | — | — | **failed**: same error | | | 19,231 |

Findings:
1. **The floor works for CeO2.** Label (log-mean, 3 directions) 43.8 ms, direction spread 0.023 dex. In Kanai's frame
   (<111>): curve 45.1 ms = **0.985 of Kanai's averaged-curve fit** (45.8 ms, n 1.77) and 0.84 of Table S1 (53.5(13)).
   The averaged curve is strongly stretched (n 1.35–1.45; Kanai's own averaged curve: 1.77), while single configurations
   decay with n ≈ 2.8 (mean per-configuration n 2.74–2.86; Table S1 n 2.80): the ensemble mixes configurations of very
   different T2.
2. **Mean-of-fits is biased low for such ensembles.** 4–9 of 20 per-configuration fits fail the guard; the window
   (t_max 50 ms) is set from the averaged curve, so slow configurations do not decay inside it and drop out. Mean-of-fits
   31–37 ms (0.60 of Table S1) is therefore not a valid comparison here; the curve fit (the ADR-0004 label) is.
3. **CaO fails above floor 500.** Floors 1,000 and 2,000 raise `LinAlgError` inside PyCCE (`simple_propagator` ->
   `numpy.linalg.eigh` of a cluster Hamiltonian); floor 500 (same seeds, same 450 Å supercell, so the same spins inside
   123.8 Å) runs. The record keeps only the traceback, not the configuration or direction. `eigh` failing points to a
   non-finite Hamiltonian (e.g. a zero distance), but this is **not established**; the CaO CIF has no duplicated sites.
   The CaO and CeO2 differences: 43Ca (I = 7/2) bath plus 17O, versus 17O only.
4. **Cost:** CeO2 (17O, I = 5/2, 1,000 spins, r_dipole 70 Å) 1.5 h per direction, in line with the I > 1/2 class;
   CaO floor 500 29 min for one direction.
5. **Floor value:** 1,000 is kept (provisional). Floor convergence is not measured yet (CaO has only the 500 point).

**Next (T-068, S015):** runner change: a configuration that raises `LinAlgError` is skipped and recorded with bath
diagnostics (`conf_errors`: configuration, seed, spins in r_bath, smallest centre-spin and spin-spin distance); more than
`max_conf_fail_frac` = 10 % failures void the step (`fit_flag = "conf_errors"`). Tests in `tests/test_pycce_runner.py`.
Re-run `compute/requests/009-e0b-cao.yaml` (CaO, one direction per shard at floor 1,000; c at floor 2,000 with
10 configurations) launched S015 with Kian's OK (Q16b; public `6de3ac0`, run 41181177145). ADR-0004 Amendment 1 accepted (Q16a).

## 8. 009-e0b-cao (run 41181177145): CaO label, floor convergence, LinAlgError (S017)

**Source:** `results/cce/009-e0b-cao/merged.jsonl` (3 records, 3 ok; public `6de3ac0`; artifact `cce-009-e0b-cao-merged`,
fetched S017 with `python scripts/fetch_cce_results.py 009-e0b-cao`); tables from
`python scripts/analyse_e0b.py results/cce/009-e0b-cao`. Runner = ADR-0004 defaults + T-068 (public `6de3ac0`).
Software on every job: ubuntu-24.04, Python 3.11.17, numpy 1.26.4, scipy 1.11.4, PyCCE 1.1.1, the same as in 008
(job logs of runs 41116974041 and 41181177145).

| item | B | floor | spins in r_bath (nominal 50 Å) | r_bath/r_dipole Å | T2 curve ms (n) | mean-of-fits ± SEM ms | per-conf n (median) | failed confs | wall s |
|---|---|---|---|---|---|---|---|---|---|
| cao-c (Ca site) | c [001] | 1,000 | 1,000 (28) | 157.5/47.3 | 26.6 (2.16) | 26.5 ± 1.9 | 3.19 | 0/20 | 6,959 |
| cao-ab | a+b [110] | 1,000 | 1,000 (28) | 157.5/47.3 | 27.6 (2.22) | 27.4 ± 2.1 | 3.48 | 0/20 | 12,389 |
| cao-abc | a+b+c [111] | 1,000 | 1,000 (28) | 157.5/47.3 | 28.4 (1.92) | 28.2 ± 2.0 | 2.98 | 0/20 | 5,813 |
| cao-c-floor2000 | c | 2,000 | — | — | **no record**: shard 3 hit the 5 h 55 min job limit (10 configurations) | | | | > 21,000 |

Every item: windows tried 50 then 200 ms (t_max 200 ms), no failed per-configuration fit, `fit_flag` empty, `label_ok`.
The shard-3 artifact holds an empty `shard_3.jsonl`: the runner writes a record only when an item ends.

Findings:
1. **CaO label (ADR-0004, log-mean of c, a+b, a+b+c): 27.5 ms**, direction spread 0.028 dex. With this, all 8 Kanai
   Table S1 hosts have an E0/E0b result.
2. **Kanai frame (<111>, fcc primitive):** curve 28.4 ms (n 1.92) = **0.89 of Kanai's averaged-curve fit** (31.8 ms,
   n 2.06; -0.050 dex); mean-of-fits 28.2 ms = **0.76 of Table S1** (37.2(71) ms; -0.121 dex), just below Kanai's
   one-SD band. Here mean-of-fits is valid (0/20 failed fits; the window reached 200 ms, unlike CeO2 in §7), and the
   per-configuration n (median 2.98) is close to Table S1's 2.84. CaO is the lowest of the 7 hosts with a valid
   mean-of-fits (range now 0.76–0.97 of Table S1, -0.12 to -0.01 dex; all below Kanai). Kanai's own averaged curve is
   0.85 of Table S1 for CaO, so part of the gap is Kanai's estimator (curve vs mean-of-fits), as for the other hosts.
3. **Floor convergence (paired), B ∥ c:** floor 500 (008 `cao-c-floor500`) and floor 1,000 (009 `cao-c`) use the same
   seeds and the same 450 Å supercell, so each configuration's inner bath is identical and the floor only adds the
   shell from 123.8 to 157.5 Å. Per configuration, Δlog10 T2 (1,000 - 500) = **-0.0022 dex mean, SD 0.0021, range
   -0.0081 to -0.0001** (20 pairs); curve fit -0.0022 dex, mean-of-fits -0.0027 dex. Doubling the bath changes T2 by
   0.5 %, a tenth of the 0.02 dex tolerance. **Floor 1,000 is converged for CaO.** The 2,000 point is still missing
   (008: LinAlgError after 19,231 s; 009: timed out). It cannot be paired with these runs anyway: 2,000 spins need
   r_bath ≈ 200 Å and a supercell ≥ 500 Å (`SUPERCELL_SAFE`), which redraws every isotope.
4. **Cost of the floor (wall s per configuration evaluation, 21 evaluations per item: 20 configurations + the first
   one's 50 ms window):** floor 500 84 s; floor 1,000 277–590 s (c 331, a+b 590, a+b+c 277; the a+b job was about 2x
   slower with identical work, so runner speed varies); floor 2,000 **> 1,900 s** (at most 10 of 11 evaluations in
   about 21,000 s), so floor 2,000 cannot run 10 configurations in one 6 h job. Each floor doubling costs about 4–6x,
   not 2x. CaO at floor 1,000: 1.6–3.4 h per direction, 7.0 h of runner time for 3 directions (25,161 s).
5. **LinAlgError (Amendment 1, T-068): not reproduced.** No configuration failed in 009 (60 of 60 ran;
   `conf_errors` is empty in all 3 records, so there are no bath diagnostics to report). In 008, `cao-A` failed after
   205 s, less than one evaluation takes at floor 1,000 (277–590 s), so most likely inside the first evaluation of its
   first direction (c; seed 1, 50 ms window). 009's `cao-c` computed that same configuration (same seed, supercell, floor, radii and package versions) without error. So the failure is
   **not a property of the CaO bath** and not deterministic at the input level. The likely source is the job's
   numerical environment (LAPACK `eigh` on a different runner CPU or BLAS kernel path). The CPU model was not logged,
   so this is **not established**. The 10 % rule of Amendment 1 stays as the safety net; no protocol change is needed.
   **Runner change (S017):** `qkhosts.labels.batch.host_info()` now stores the CPU model, CPU count, BLAS build and
   package versions in every CCE record (`host`), so a future failure can be tied to the hardware (`tests/test_batch.py`).
6. **Partial results are lost on a timeout.** The 6 h limit on one item loses all of its configurations. For E2's
   I > 1/2 items (3.4 h per direction in this run) that is a real risk. T-015 (resumable runner) should write one
   line per finished configuration (curve + diagnostics), so that a timed-out item can be resumed or merged.

**Optional floor-2,000 check (not needed to close ADR-0004; proposed to Kian S017):** a paired run in 2 shards on
one 560 Å supercell (seeds 1–3, B ∥ c, `t_max_ms: 200`): floor 1,000 (about 20 min) and floor 2,000 (3 configurations,
about 2–3 h by finding 4). The reason to run it: one more point for the ADR-0004 floor. The reason to skip it: the
paired 500 -> 1,000 step is already 0.0022 dex, and CeO2 at floor 1,000 matches Kanai's curve (0.985). Not drafted
unless Kian asks (Q19).
