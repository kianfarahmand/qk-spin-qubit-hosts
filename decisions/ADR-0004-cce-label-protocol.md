# ADR-0004: CCE label protocol
**Status:** **accepted** by Kian (S015, 2026-10-30; Q14 "Accept"). Item 2 decided by Kian (S012): option A. E0b
(007-e0b) is closed; only the floor value (item 5) is provisional until 008-e0b-dilute (launched S015, Q13). Evidence: `experiments/E0_notes.md`
(runs 004, 005, 006; §6 for 007).

## Context
E2 (D-CCE) needs one fixed PyCCE protocol for 500–3,000 materials. Kanai (2022) gives the physics settings but
no site rule and no cut-offs. The E0 pilot ran diamond, 4H-SiC and alpha-quartz against Kanai Table S1, then
changed one setting at a time (006-e0-diag); E0b (007) added the field-direction spread, 5 more Table S1 hosts and
a dense bath. Sizes of the effects measured, in log10 T2:

| Setting changed | Effect | Source |
|---|---|---|
| B direction c / a+b / a+b+c (spread, max - min) | **0.016–0.24 dex** (SiC 0.016 ... Si 0.24) | 007 |
| B ∥ [111] -> [001] (diamond) | +0.121 dex, n 1.70 -> 2.62 | 006 diamond-base / diamond-z; 007 |
| Kanai origin rule vs vacancy | ≤ 0.010 dex | 006 *-origin |
| vacancy site within a host | ≤ 0.003 dex | 004 site spread |
| CCE-3 vs CCE-2 | +0.011 dex | 006 diamond-cce3 / rd8 |
| gCCE vs CCE (S = 1/2) | 0 | 006 diamond-gcce |
| r_dipole 15 -> 8 Å | +0.005 dex | 006 diamond-rd8 |
| r_bath/r_dipole 50/15 -> 65/20 Å | -0.001 to -0.009 dex | 004, 005 ladders |
| bath cap 1,500 -> 3,000 spins (Al2O3) | -0.003 dex (750 -> 3,000: -0.019) | 007 al2o3-c-cap* |
| 20 configurations: SEM of mean | 0.005–0.017 dex | 006, 007 per-config fits |

## Decision (proposed for acceptance, S015)
1. **Central spin:** S = 1/2, g = 2, B = 5 T, Hahn echo, t = total free-evolution time (`as_delay=False`).
   As Kanai; keeps Δ = log T2_CCE - log T2_SL on the scaling law's own footing.
2. **Field direction: option A, chosen by Kian in S012 (Q11).** Label = log-mean of T2 over 3 fixed directions of the
   conventional standard cell, c, a+b and a+b+c (cubic: [001], [110], [111]); each direction's T2 is stored
   (`direction_t2_ms`), and the spread (`direction_spread_log10`) is a per-material feature and part of σ_y.
   E0b confirms it: the direction spread is 0.016 (4H-SiC) to 0.24 dex (Si), larger than every other knob, and the
   scaling law has no direction in it. Options B (c only) and C (Kanai's code frame) are not used for labels; C is
   used only for benchmarking against Kanai (see Benchmarks).
3. **Site rule:** `site_rule: single_most_spinful`: a vacancy on **one** host site, the inequivalent site with the
   highest multiplicity among the element with the most spinful nuclei per cell (PyCCE abundances); ties go to the
   first site in the standardised structure. Within-host site spread ≤ 0.003 dex (SiC, quartz). Kanai's origin rule
   is rejected: within 0.010 dex, but it depends on the CIF's arbitrary origin. Works on real PyCCE (E0b picked C, Si,
   Mg, Zn, Al, Ca, and O in CeO2). A 3-site audit runs on a 5 % stratified subsample.
4. **Method:** CCE-2, PyCCE `method="cce"` (gCCE identical for S = 1/2, 1.3x slower). CCE-3 audit on the same 5 %
   subsample.
5. **Cut-offs and bath size:** r_bath 50 Å, r_dipole 15 Å, supercell 200 Å, single rung in production, then:
   - **Cap `max_bath_spins: 1500`** (dense baths): r_bath shrinks to enclose 1,500 spins, r_dipole by the same factor.
     Al2O3 (27Al 100 %): cap 1,500 is within 0.003 dex of cap 3,000; cap 750 is +0.019 dex (edge of the 0.02 tolerance).
   - **Floor `min_bath_spins: 1000`** (ultra-dilute baths; **provisional until 008-e0b-dilute**, Q13): r_bath (and the
     supercell) grow to enclose 1,000 spins, r_dipole by the same factor (Kanai: "~1,000 bath spins"). CaO and CeO2
     have 30 and 9 spins inside 50 Å and give no usable fit without it (T-066). 008 tests floors 500/1,000/2,000 on CaO.
   - **Fit guard** (`guard_fit`, T-066): a fit is NaN when n is outside [0.5, 6], T2 lies outside the sampled window,
     or too few points are usable; the host label is NaN (`label_ok = false`) rather than a log-mean of bad fits.
   - Audit: a 65/20 Å rung (scaled the same way under cap or floor) on the 5 % subsample; if |Δlog10 T2| > 0.02 for
     more than 10 % of it, E2 escalates the whole stratum.
6. **Ensemble:** 20 random isotope configurations, seeds `seed + 1 + k`, the same seeds at every direction, site and
   rung (paired comparisons; runs reproduce exactly across Actions jobs, E0_notes finding 8). **Exception (`n_conf`
   rule):** 1 configuration when ≥ 99 % of the expected spins inside r_bath come from isotopes with natural abundance
   ≥ 99 % (e.g. 27Al, 19F, 31P, 23Na, 55Mn, 59Co, 75As, 93Nb, 127I, 133Cs). Basis: Al2O3's 5 configurations agree to
   SEM ~3e-10 ms at caps 750–3,000 (only rare 17O varies). Hosts with partly abundant isotopes (25Mg 10 %, 67Zn 4.1 %,
   29Si 4.7 %) keep 20.
7. **Estimator:** label = linearised stretched-exponential fit of the configuration-averaged curve (`t2_ms`,
   `stretch_n`). Also stored: mean of per-configuration fits with SEM and SD, and the NLS fit (diagnostic only).
   Comparisons with Kanai Table S1 use mean-of-fits; with Kanai Fig. S2 use the curve fit.
8. **Isotopes:** PyCCE natural abundances (`common_concentrations`), recorded per result. **Closed (E0b):** equal to
   Kanai's `isotopetable.csv` for all 7 recorded spinful isotopes (13C, 29Si, 17O, 43Ca, 25Mg, 67Zn, 27Al; ratio 1.0000).

## Benchmarks (E0 + E0b, in Kanai's code frame)
Kanai's code puts B along a x b of the MP primitive cell: **<111> for fcc-primitive hosts** (diamond, Si, MgO, CaO,
CeO2) and **c for hexagonal ones** (4H-SiC, quartz, ZnO). Si settles it ([001] 1.80 ms vs Table S1 1.24(23); [111]
1.03). Ours / Kanai at that direction (`experiments/E0_notes.md` §6):

| host | ours mean-of-fits / Table S1 | ours curve / Kanai Fig. S2 curve fit |
|---|---|---|
| diamond [111] | 0.80 | 0.86 |
| Si [111] | 0.84 | 0.98 |
| ZnO c | 0.86 | 0.90 |
| alpha-quartz c | 0.88 | 0.92 |
| 4H-SiC c | 0.91 | 0.95 |
| MgO [111] (cap 1,500) | 0.97 | 1.01 |
| CeO2 [111] (floor 1,000; 008, added S015) | not valid (biased, see Amendment 1) | 0.98 |
| CaO | pending 009 (008 failed above floor 500) | |

All six are below Table S1 (-0.10 to -0.01 dex). Their range (0.08 dex) is small next to the direction spread, and the
mean offset is absorbed by the Δ intercept. Diamond's decay shape stays off (n 1.70 vs 2.38): question for the CCE
collaborator (ADR-0006). The E0 reading "diamond [001] matches Table S1" was a coincidence (Si shows it).

## Cost and E2 budget per nuclear-spin class (option A, 3 directions, one site, one Actions runner)
From E0b wall times (`experiments/E0_notes.md` §6 finding 6):

| class (dominant bath isotope) | example | per direction | per material (3 directions) |
|---|---|---|---|
| I = 1/2, dilute, 20 configurations | 13C, 29Si (diamond, Si, SiC, quartz) | 1.0–3.1 min | 3–9 min |
| I > 1/2, 20 configurations | 67Zn (ZnO, 911 spins); 25Mg (MgO, cap 1,500) | 56–67 min; 101–149 min | 2.8–3.4 h; 5.1–7.5 h |
| I > 1/2, ~100 % abundant, 1 configuration (item 6), cap 1,500 | 27Al (Al2O3, B ∥ c only) | ~12 min | ~36 min |
| ultra-dilute with floor | CaO, CeO2 | pending 008 | |

E2 for N materials ≈ N x (f_1/2 x 6 min + f_>1/2 x 3–7.5 h + f_dense x 0.6 h). The class mix f is unknown until E1/T-014;
most spinful elements have I > 1/2, so f_>1/2 is likely large. Example: 500 materials all in the I > 1/2 class ≈
1,400–3,700 runner-h, i.e. 3–8 days of wall time at 20 parallel free jobs (500 all I = 1/2: 25–75 runner-h). The S012
estimate (~9 min per material) holds only for I = 1/2. Actions' 6 h job limit means one direction per shard for I > 1/2.
**Levers, to be chosen in T-015 once the class mix is known (not decided here):** (a) 10 configurations for I > 1/2
(SEM of the mean 0.005–0.009 dex at 20, so about 0.007–0.013 at 10; halves the cost); (b) a smaller cap for I > 1/2
(untested; needs a check like Al2O3's); (c) fewer I > 1/2 materials through the stratified sampler (T-014).

## Consequences
- Runner (`src/qkhosts/labels/pycce_runner.py`): `b_directions_lattice`, `site_rule: single_most_spinful`,
  `max_bath_spins`, `min_bath_spins`, `guard_fit` and `label_ok` exist; all but the floor have worked on real PyCCE
  (007; floor pending 008). **Done S015 (T-067):** the `CCEProtocol` defaults are this protocol (single 50/15 Å rung,
  option A directions, single_most_spinful, cap 1,500, floor 1,000, 20 configurations) and the `n_conf` rule (item 6)
  is `dense_bath_share` + `n_conf_dense = 1`, recorded per result as `n_conf_used` and `dense_bath_share`
  (`tests/test_pycce_runner.py`: `test_defaults_are_adr0004`, `test_dense_bath_share`, `test_dense_rule_in_runner`).
  The E0 settings stay available as `E0_PROTOCOL`. Requests 004–008 ran under the old defaults; each result records its full protocol, and request ids are never re-run.
- D-CCE labels will not be Kanai's numbers; the data card states the protocol and the benchmark ratios.
- The diamond shape question goes to the CCE collaborator (ADR-0006, T-008) and, if needed, to Kanai et al.
- History: proposed S012 (E0); option A chosen S012; E0b interim S014; closed, proposed and accepted S015 (Q14).

## Amendments
**Amendment 1 (S015, 2026-10-31; accepted by Kian, Q16a).** Evidence: 008-e0b-dilute (`experiments/E0_notes.md` §7).
- (a) **Failed configurations (T-068):** a configuration whose PyCCE call raises `LinAlgError` is skipped and recorded
  with bath diagnostics (`conf_errors`); if more than 10 % of the configurations of a step fail (`max_conf_fail_frac`),
  the step's T2 is NaN (`fit_flag = "conf_errors"`) and the label is not ok. Reason: CaO at floors 1,000 and 2,000
  failed as a whole on one such error, so no label and no diagnosis were possible. Risk: if failures correlate with
  bath geometry, skipping them biases T2; the 10 % limit bounds that, and the diagnostics will show it.
- (b) **Item 5 floor:** CeO2 at floor 1,000 gives a label that matches Kanai's averaged curve (0.985); 1,000 stays the
  floor, still provisional until a floor-convergence point exists (009: CaO c at 2,000).
- (c) **Item 7 caveat:** for ensembles with a wide T2 spread (ultra-dilute baths), mean-of-fits is biased low because
  slow configurations do not decay inside the window set by the averaged curve (CeO2: 4–9 of 20 fits fail; mean-of-fits
  0.60 of Table S1). Benchmarks against Table S1 for such hosts use the curve fit against Kanai's averaged curve.

## Literature inputs (T-005, S005)
Full notes with locations: `literature/notes/kanai2022_si.md`, `literature/notes/toriyama2025.md`. Summary:
- Kanai: S = ½, g = 2, B = 5 T along z, CCE-2 (checked against CCE-3), no quadrupole (lower bound), thermalised bath, 5–10 (benchmarks) or 10 (single elements) random configurations. Cut-offs "optimized for each condition", with no numbers given (about 1,000 bath spins). **No central-spin site rule stated.** Time axis = total free-evolution time.
- Toriyama 2025 (PyCCE): spin-1 at a vacancy on one host site, 5 T perpendicular to the layer, 40 configurations with curves averaged before the fit, r_bath/r_dipole 50/15 Å (screening) and 120/30 Å (heterostructures), CCE-2, `as_delay=False`.
- Benchmark targets for E0: Kanai Table S1 (8 hosts with SD), e.g. diamond 0.97(12) ms, 4H-SiC 1.35(9) ms, α-quartz 3.42(15) ms.
