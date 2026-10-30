"""PyCCE label generation (E0 pilot / E2 production).

Settings come from the T-005 literature read (S005; `literature/notes/kanai2022_si.md`,
`literature/notes/toriyama2025.md`) and are frozen in ADR-0004 after the E0 pilot (T-010):
- B = 5 T along z (PyCCE takes gauss: 5e4 G); S = 1/2, g = 2 (Kanai Table S1 conditions).
- Hahn echo with `as_delay=False`: t is the total free-evolution time, as in Kanai's T2.
- Central spin at a vacancy on a host site (Toriyama's code), averaged over up to n_sites_max
  inequivalent sites, weighted by multiplicity (our convention; E0 measures whether it matters).
  ADR-0004 item 3 (S013): `site_rule="single_most_spinful"` takes one site (see `pick_site`).
- Field direction: one Cartesian `b_direction`, or (ADR-0004 item 2, option A, S013) a list
  `b_directions_lattice` of lattice-vector combinations of the CIF cell, which is the conventional
  cell (c, a+b, a+b+c = (0,0,1), (1,1,0), (1,1,1)); the host label is the log-mean over directions.
- Bath-size cap (ADR-0004 item 5, S013): `max_bath_spins` shrinks r_bath for dense baths (see `cap_radii`).
- Bath-size floor (T-066, S014): `min_bath_spins` grows r_bath (and the supercell) for ultra-dilute baths
  until that many spins sit inside (see `floor_radii`); r_dipole grows by the same factor.
- Fit guard (T-066, S014): fits with n outside `fit_n_range`, T2 outside the sampled window or too few usable
  points become NaN (`guard_fit`), and a NaN in any site or direction makes the host label NaN (`label_ok`).
- Convergence ladder over (r_bath, r_dipole) starting at Toriyama's screening values 50/15 Å.
- Per site and ladder step, coherence curves are averaged over n_conf isotope configurations before
  the fit (Toriyama); the mean of per-configuration fits is also recorded (ADR-0004 decides).

The PyCCE calls follow the API observed on Actions (pycce 1.1.1, `literature/notes/pycce_api.md`).
Version-sensitive pieces sit in small helpers (`_load_sites`, `_center_position`) so tests can mock
PyCCE and E0 can check them against the real package.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

GAUSS_PER_TESLA = 1e4


@dataclass
class CCEProtocol:
    central_spin: float = 0.5           # S
    central_g: float = 2.0
    site_rule: str = "host_atom_removed_avg_inequivalent"  # ADR-0004; or "origin_kanai" (see origin_site)
    n_sites_max: int = 3
    b_field_T: float = 5.0
    b_direction: tuple = (0.0, 0.0, 1.0)  # Cartesian; pymatgen puts the c axis along z
    b_directions_lattice: tuple | None = None  # e.g. ((0,0,1), (1,1,0), (1,1,1)); overrides b_direction
    max_bath_spins: int | None = None   # cap on spins inside r_bath (dense baths); None = no cap
    min_bath_spins: int | None = None   # floor on spins inside r_bath (ultra-dilute baths); None = no floor
    fit_n_range: tuple = (0.5, 6.0)     # accepted stretch exponents (fit guard)
    order: int = 2
    method: str = "cce"                 # PyCCE compute method: "cce" (conventional) or "gcce"
    ladder: tuple = ((50.0, 15.0), (65.0, 20.0), (80.0, 25.0))  # (r_bath, r_dipole) Å, climbed in order
    supercell: int = 200
    n_conf: int = 20                    # random isotope configurations
    t_max_ms: float = 50.0              # initial window; adapted to the decay (see _adapt_window)
    n_t: int = 201
    min_fit_points: int = 15            # points with 0.05 < L < 0.95 needed for a fit
    max_window_tries: int = 4
    pulses: int = 1                     # Hahn echo
    as_delay: bool = False              # False: time axis is total evolution time (Kanai)
    converge_tol_log10: float = 0.02    # |Δlog10 T2| between successive ladder steps
    seed: int = 0
    extra: dict = field(default_factory=dict)


def fit_t2(t_ms: np.ndarray, coherence: np.ndarray) -> tuple[float, float]:
    """Fit |L(t)| = exp(-(t/T2)^n), t = total evolution time (ms); returns (T2_ms, stretch exponent n).
    Uses linearisation log(-log L) = n log t - n log T2 on 0.05 < L < 0.95."""
    L = np.abs(np.asarray(coherence))
    m = (L > 0.05) & (L < 0.95) & (t_ms > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    x, yv = np.log(t_ms[m]), np.log(-np.log(L[m]))
    n, b = np.polyfit(x, yv, 1)
    return float(np.exp(-b / n)), float(n)


def fit_t2_nls(t_ms: np.ndarray, coherence: np.ndarray) -> tuple[float, float]:
    """Nonlinear least-squares fit of exp(-(t/T2)^n) to the whole curve (t in ms), started from
    `fit_t2`; returns (T2_ms, n) or NaNs. Recorded next to the linearised fit (S011 diagnostic)."""
    from scipy.optimize import curve_fit

    t, L = np.asarray(t_ms, float), np.abs(np.asarray(coherence, float))
    t2_0, n_0 = fit_t2(t, L)
    if not (np.isfinite(t2_0) and np.isfinite(n_0)):
        return float("nan"), float("nan")
    try:
        (t2, n), _ = curve_fit(lambda x, a, k: np.exp(-(x / a) ** k), t, L, p0=(t2_0, float(np.clip(n_0, 0.6, 5.9))),
                               bounds=([1e-12, 0.5], [np.inf, 6.0]), maxfev=5000)
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return float(t2), float(n)


def n_fit_points(coherence: np.ndarray) -> int:
    L = np.abs(np.asarray(coherence))
    return int(((L > 0.05) & (L < 0.95)).sum())


def _load_sites(structure_path: str | Path, n_sites_max: int):  # pragma: no cover - needs pymatgen
    """Return (ase Atoms, [(element, frac_xyz, multiplicity), ...]) for up to n_sites_max
    symmetry-inequivalent sites, most frequent first."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    s = Structure.from_file(str(structure_path))
    groups = SpacegroupAnalyzer(s).get_symmetrized_structure().equivalent_indices
    groups = sorted(groups, key=lambda g: (-len(g), g[0]))[:n_sites_max]
    sites = [(s[g[0]].specie.symbol, np.asarray(s[g[0]].frac_coords, float), len(g)) for g in groups]
    return AseAtomsAdaptor.get_atoms(s), sites


def origin_site(lattice: np.ndarray, frac: np.ndarray, species: list[str], tol_A: float = 0.4):
    """Kanai's placement rule (their load_crystal.m: "electron spin is set at r=0", nuclei closer than
    0.4 Å to it are erased): the centre sits at the CIF origin. Returns (element, frac) of the atom
    within tol_A of the origin (a vacancy there), or (None, zeros) when the origin is an empty point
    (e.g. a bond centre in MP's diamond mp-66). lattice: rows in Å; frac: (n, 3)."""
    L, f = np.asarray(lattice, float), np.asarray(frac, float).reshape(-1, 3)
    w = f - np.round(f)                       # nearest periodic image of each site to the origin
    d = np.linalg.norm(w @ L, axis=1)
    i = int(np.argmin(d))
    if d[i] < tol_A:
        return species[i], f[i] % 1.0
    return None, np.zeros(3)


def _load_origin(structure_path: str | Path, tol_A: float = 0.4):  # pragma: no cover - needs pymatgen
    """(ase Atoms, [(element | None, frac, 1)]) for the origin_kanai rule."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    s = Structure.from_file(str(structure_path))
    el, frac = origin_site(s.lattice.matrix, s.frac_coords, [x.specie.symbol for x in s], tol_A)
    return AseAtomsAdaptor.get_atoms(s), [(el, frac, 1)]


def pick_site(sites: list, spinful_fraction: dict) -> tuple:
    """ADR-0004 item 3: the single vacancy site. sites: all inequivalent [(element, frac, multiplicity)]
    in structure order; spinful_fraction: element -> natural abundance of its spinful isotopes (0-1).
    Takes the element with the most spinful nuclei per cell (sum of multiplicity x fraction), then its
    site with the highest multiplicity; ties go to the first in `sites` order."""
    per_el: dict[str, float] = {}
    for el, _, mult in sites:
        per_el[el] = per_el.get(el, 0.0) + mult * float(spinful_fraction.get(el, 0.0))
    best = max(per_el.values())
    el = next(e for e, _, _ in sites if per_el[e] == best)  # first element (structure order) on a tie
    own = [s for s in sites if s[0] == el]
    top = max(m for _, _, m in own)
    return next(s for s in own if s[2] == top)


def _spinful_fractions(pc, elements) -> dict:
    """element -> total abundance of spinful isotopes in PyCCE's natural-abundance table. Isotopes with
    spin 0 are dropped when PyCCE's `common_isotopes` exposes spins (they should not be in the table)."""
    iso = _isotopes(None, pc, elements) or {}
    spins = getattr(pc, "common_isotopes", None) or getattr(getattr(pc, "bath", None), "common_isotopes", None) or {}
    return {el: float(sum(v for k, v in iso.get(el, {}).items()
                          if k not in spins or float(getattr(spins[k], "s", 1.0)) > 0))
            for el in elements}


def field_directions(cell, proto: "CCEProtocol") -> list[tuple[list | None, np.ndarray]]:
    """[(lattice combination or None, unit Cartesian vector)] for the protocol's field directions.
    Lattice combinations use the lattice vectors of the CIF cell (columns of PyCCE's cell.cell)."""
    if proto.b_directions_lattice is None:
        v = np.asarray(proto.b_direction, float)
        return [(None, v / np.linalg.norm(v))]
    out = []
    for combo in proto.b_directions_lattice:
        v = np.asarray(cell.cell, float) @ np.asarray(combo, float)
        out.append(([float(x) for x in combo], v / np.linalg.norm(v)))
    return out


def cap_radii(dist_A: np.ndarray, r_bath: float, r_dipole: float, cap: int | None) -> tuple[float, float, bool]:
    """Bath-size cap (ADR-0004 item 5). dist_A: centre-to-spin distances (Å) of one configuration.
    If more than `cap` spins lie within r_bath, r_bath shrinks to just enclose the `cap` nearest spins
    and r_dipole shrinks by the same factor, so the pairs per spin stay as in a dilute bath
    (about cap x (r_dipole/r_bath)^3 neighbours). Returns (r_bath, r_dipole, capped)."""
    d = np.sort(np.asarray(dist_A, float))
    if cap is None or (d <= r_bath).sum() <= cap:
        return float(r_bath), float(r_dipole), False
    r_new = float(0.5 * (d[cap - 1] + d[cap]))  # midway between spin cap and spin cap+1
    return r_new, float(r_dipole * r_new / r_bath), True


def floor_radii(dist_A: np.ndarray, r_bath: float, r_dipole: float, floor: int | None,
                r_safe: float) -> tuple[float, float, bool, float | None]:
    """Bath-size floor (T-066), the mirror of `cap_radii`. dist_A: centre-to-spin distances (Å) of one
    configuration; r_safe: radius (Å) inside which the generated supercell is complete. If fewer than `floor`
    spins lie within r_bath, r_bath grows to just enclose the `floor` nearest spins and r_dipole grows by the
    same factor (at least `floor` spins when several share the floor-th distance, as on a lattice shell).
    Returns (r_bath, r_dipole, floored, r_needed): r_needed is None when the supercell is big
    enough, else an estimate (Å, 20 % margin, from the spin density inside r_safe) of the radius to generate."""
    d = np.sort(np.asarray(dist_A, float))
    if floor is None or (d <= r_bath).sum() >= floor:
        return float(r_bath), float(r_dipole), False, None
    if (d <= r_safe).sum() > floor:
        r_new = float(0.5 * (d[floor - 1] + d[floor]))
        return r_new, float(r_dipole * r_new / r_bath), True, None
    n_safe = max(int((d <= r_safe).sum()), 1)
    return float(r_bath), float(r_dipole), False, float(r_safe * (1.2 * floor / n_safe) ** (1.0 / 3.0))


# r_safe = SUPERCELL_SAFE x supercell size (Å). PyCCE's supercell (linear size about `supercell` Å) is centred on
# cell 0 (007 early shards: 30 spins inside 50 Å of 513 in CaO's 200 Å supercell, as a centred cube predicts);
# 0.4 stays inside the inscribed sphere of a hexagonal prism (0.433).
SUPERCELL_SAFE = 0.4


def guard_fit(t2: float, n: float, ts: np.ndarray, n_points: int, min_points: int,
              n_range: tuple = (0.5, 6.0)) -> tuple[float, float, str]:
    """Fit guard (T-066). Returns (T2_ms, n, "") for a usable fit, else (nan, nan, reason): too few points
    with 0.05 < L < 0.95, a failed fit, n outside n_range, or T2 outside the sampled window [ts[1], ts[-1]] ms."""
    if n_points < min_points:
        return float("nan"), float("nan"), "few_points"
    if not (np.isfinite(t2) and np.isfinite(n)):
        return float("nan"), float("nan"), "fit_failed"
    if not n_range[0] <= n <= n_range[1]:
        return float("nan"), float("nan"), "n_out_of_range"
    if not ts[1] <= t2 <= ts[-1]:
        return float("nan"), float("nan"), "t2_outside_window"
    return float(t2), float(n), ""


def _center_position(cell, frac: np.ndarray) -> np.ndarray:
    """Cartesian position (Å) of a cell-0 site. PyCCE stores lattice vectors as columns of cell.cell."""
    if hasattr(cell, "to_cartesian"):
        return np.asarray(cell.to_cartesian(frac), float)
    return np.asarray(cell.cell, float) @ np.asarray(frac, float)


def _coherence(pc, bath, pos, ts, r_bath, r_dipole, proto: CCEProtocol, b_unit=None) -> np.ndarray:
    field_G = np.asarray(proto.b_direction if b_unit is None else b_unit, float)
    field_G = proto.b_field_T * GAUSS_PER_TESLA * field_G / np.linalg.norm(field_G)
    calc = pc.Simulator(spin=proto.central_spin, position=pos, bath=bath, r_bath=r_bath, r_dipole=r_dipole,
                        order=proto.order, magnetic_field=field_G, pulses=proto.pulses, as_delay=proto.as_delay)
    return np.abs(np.asarray(calc.compute(ts, method=proto.method, quantity="coherence"), float))


def _adapt_window(simulate, proto: CCEProtocol) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Find a time window (ms) that resolves the decay: shrink when it decays within a few points,
    widen when L(t_max) > 0.05. Returns (ts, coherence of that window, windows tried)."""
    t_max, tried = proto.t_max_ms, []
    for _ in range(proto.max_window_tries):
        ts = np.linspace(0.0, t_max, proto.n_t)
        L = simulate(ts)
        tried.append(t_max)
        if L[-1] > 0.05:
            t_max *= 4.0
        elif n_fit_points(L) < proto.min_fit_points:
            t2, _ = fit_t2(ts, L)
            below = np.nonzero(L < 0.05)[0]
            t_max = 3.0 * t2 if np.isfinite(t2) else ts[max(int(below[0]), 2)]
        else:
            return ts, L, tried
    return ts, L, tried


def _site_step(pc, cell, element, frac, r_bath, r_dipole, proto, ts=None, b_unit=None) -> dict:
    """Ensemble over n_conf isotope seeds for one site, one field direction and one ladder step.
    Baths are generated one at a time (dense baths are large); the bath-size cap is set from config 1."""
    pos = _center_position(cell, frac)
    remove = [(element, list(frac))] if element else None  # None: centre on an empty point, nothing erased
    size = proto.supercell

    def bath(k):
        return cell.gen_supercell(size, remove=remove, seed=proto.seed + 1 + k)

    def distances(b):
        xyz = np.asarray(getattr(b, "xyz", np.empty((0, 3))), float).reshape(-1, 3)
        return xyz, np.linalg.norm(xyz - pos, axis=1)

    b0 = bath(0)
    xyz, dist = distances(b0)
    n_nominal = int((dist <= r_bath).sum())
    r_bath_set, r_dipole_set = r_bath, r_dipole
    r_bath, r_dipole, capped = cap_radii(dist, r_bath, r_dipole, proto.max_bath_spins)
    floored = False
    for _ in range(4):  # grow the supercell until the floor fits inside it (T-066)
        r_b, r_d, floored, r_need = floor_radii(dist, r_bath, r_dipole, proto.min_bath_spins, SUPERCELL_SAFE * size)
        if r_need is None:
            r_bath, r_dipole = r_b, r_d
            break
        size = int(np.ceil(r_need / SUPERCELL_SAFE))
        del b0
        b0 = bath(0)
        xyz, dist = distances(b0)
    else:
        raise RuntimeError(f"bath floor {proto.min_bath_spins} not reached with supercell {size} A")

    def coh(b, t):
        return _coherence(pc, b, pos, t, r_bath, r_dipole, proto, b_unit)

    tried = None
    if ts is None:
        ts, L0, tried = _adapt_window(lambda t: coh(b0, t), proto)
        curves = [L0]
    else:
        curves = [coh(b0, ts)]
    n_bath = int(len(b0))
    del b0
    curves += [coh(bath(k), ts) for k in range(1, proto.n_conf)]
    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)
    npts, rng = n_fit_points(mean_curve), proto.fit_n_range
    t2_avg, n_avg, flag = guard_fit(*fit_t2(ts, mean_curve), ts, npts, proto.min_fit_points, rng)
    t2_nls, n_nls, _ = guard_fit(*fit_t2_nls(ts, mean_curve), ts, npts, proto.min_fit_points, rng)
    fits = np.array([guard_fit(*fit_t2(ts, c), ts, n_fit_points(c), 3, rng)[:2] for c in curves],
                    float).reshape(-1, 2)
    per = fits[:, 0]
    ok = per[np.isfinite(per)]
    return {"t2_ms": t2_avg, "stretch_n": n_avg, "fit_flag": flag, "n_fit_points": npts,
            "t2_ms_nls": t2_nls, "stretch_n_nls": n_nls,
            "per_conf_t2_ms": per.tolist(), "per_conf_n": fits[:, 1].tolist(),
            "n_bath_in_rbath": int((dist <= r_bath).sum()), "n_bath_in_rbath_nominal": n_nominal,
            "capped": capped, "floored": floored, "supercell_used": size,
            "r_bath_used_A": r_bath, "r_dipole_used_A": r_dipole,
            "r_bath_set_A": r_bath_set, "r_dipole_set_A": r_dipole_set,
            "curve_t_ms": ts.tolist(), "curve_mean": mean_curve.tolist(),
            "t2_ms_mean_of_fits": float(ok.mean()) if len(ok) else float("nan"),
            "t2_sem_ms": float(ok.std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else float("nan"),
            "n_fit_failed": int(len(per) - len(ok)), "n_bath": n_bath,  # whole supercell, not r_bath
            "min_bath_dist_A": float(dist.min()) if len(xyz) else float("nan"),
            "t_max_ms": float(ts[-1]), "windows_tried_ms": tried, "_ts": ts}


def _converge_site(pc, cell, element, frac, proto: CCEProtocol, b_unit=None) -> dict:
    """Climb the (r_bath, r_dipole) ladder until |Δlog10 T2| < tol between successive steps.
    The same isotope seeds and time window are used at every step (paired comparison)."""
    steps, ts = [], None
    for r_bath, r_dipole in proto.ladder:
        t0 = time.time()
        st = _site_step(pc, cell, element, frac, r_bath, r_dipole, proto, ts, b_unit)
        ts = st.pop("_ts")
        st.update(r_bath_A=r_bath, r_dipole_A=r_dipole, wall_s=round(time.time() - t0, 2))
        steps.append(st)
        if len(steps) >= 2 and np.isfinite(steps[-1]["t2_ms"]) and np.isfinite(steps[-2]["t2_ms"]):
            if abs(np.log10(steps[-1]["t2_ms"] / steps[-2]["t2_ms"])) < proto.converge_tol_log10:
                return {"converged": True, "final": steps[-1], "ladder": steps}
    return {"converged": False, "final": steps[-1], "ladder": steps}


def run_material(structure_path: str | Path, proto: CCEProtocol, out_dir: str | Path, name: str | None = None) -> dict:
    """Compute ensemble-averaged Hahn-echo T2 (ms) for one host. Resumable: skips if the result exists.
    `name` sets the output file stem (default: the CIF stem), so protocol variants of one CIF do not collide.

    Output JSON: protocol, per-site results (each with its convergence ladder), and the
    multiplicity-weighted host values t2_ms (log-mean over sites), converged (all sites), cost.
    With `b_directions_lattice`, each site holds one record per direction (`directions`), its `final`
    summarises them (log-mean T2), and `direction_t2_ms` gives the host T2 per direction (log-mean over
    sites), so a single-direction label can be extracted. Host t2_ms = log-mean over sites and directions."""
    out = Path(out_dir) / ((name or Path(structure_path).stem) + ".json")
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    try:
        import pycce as pc
    except ImportError as e:  # pragma: no cover
        raise ImportError("install extras: pip install -e .[cce]") from e
    t0 = time.time()
    spinful = None
    if proto.site_rule == "origin_kanai":
        atoms, sites = _load_origin(structure_path)
    elif proto.site_rule == "single_most_spinful":
        atoms, sites = _load_sites(structure_path, 10 ** 6)
        spinful = _spinful_fractions(pc, sorted({el for el, _, _ in sites}))
        if not any(v > 0 for v in spinful.values()):  # table not found: never pick a site blindly
            raise RuntimeError(f"no spinful-isotope abundances from PyCCE for {sorted(spinful)}")
        sites = [pick_site(sites, spinful)]
    else:
        atoms, sites = _load_sites(structure_path, proto.n_sites_max)
    cell = pc.read_ase(atoms) if hasattr(pc, "read_ase") else pc.BathCell.from_ase(atoms)
    dirs = field_directions(cell, proto)
    per_site = []
    for element, frac, mult in sites:
        recs = []
        for combo, b_unit in dirs:
            r = _converge_site(pc, cell, element, np.asarray(frac, float), proto, None if combo is None else b_unit)
            recs.append({"b_lattice": combo, "b_cart_unit": [float(x) for x in b_unit], **r})
        site = {"element": element or "", "frac": [float(x) for x in frac], "multiplicity": int(mult)}
        site.update(recs[0] if len(recs) == 1 else _combine_directions(recs))
        per_site.append(site)
    w = np.array([s["multiplicity"] for s in per_site], float)
    t2s = np.array([s["final"]["t2_ms"] for s in per_site], float)
    good = np.isfinite(t2s) & (t2s > 0)
    t2 = _log_mean(t2s, w, strict=True)  # T-066: one guarded (NaN) site or direction voids the label
    res = {
        "structure": Path(structure_path).name, "protocol": asdict(proto), "protocol_hash": protocol_hash(proto),
        "t2_ms": t2, "log10_t2_ms": float(np.log10(t2)) if np.isfinite(t2) else float("nan"),
        "site_spread_log10": float(np.ptp(np.log10(t2s[good]))) if good.sum() > 1 else 0.0,
        "converged": bool(good.all() and all(s["converged"] for s in per_site)),
        "label_ok": bool(np.isfinite(t2)),
        "n_sites": len(per_site), "sites": per_site, "wall_s": round(time.time() - t0, 2),
        "isotopes": _isotopes(cell, pc, _symbols(atoms)),
    }
    if spinful is not None:
        res["spinful_fraction"] = spinful
    if len(dirs) > 1:
        per_dir = [_log_mean(np.array([s["directions"][j]["final"]["t2_ms"] for s in per_site], float), w)
                   for j in range(len(dirs))]
        lg = np.log10([x for x in per_dir if np.isfinite(x) and x > 0])
        res.update(n_directions=len(dirs), direction_t2_ms=per_dir,
                   direction_spread_log10=float(np.ptp(lg)) if len(lg) > 1 else 0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    return res


def _log_mean(t2: np.ndarray, w: np.ndarray | None = None, strict: bool = False) -> float:
    """Weighted log-mean (ms) of the positive finite T2 values (ms); NaN if there are none, or, with
    strict=True, if any value is not positive and finite."""
    t2 = np.asarray(t2, float)
    w = np.ones_like(t2) if w is None else np.asarray(w, float)
    good = np.isfinite(t2) & (t2 > 0)
    if strict and not good.all():
        return float("nan")
    return float(10 ** np.average(np.log10(t2[good]), weights=w[good])) if good.any() else float("nan")


def _combine_directions(recs: list[dict]) -> dict:
    """Site record for several field directions: the per-direction records plus a `final` summary
    (log-mean T2 of the curve fits and of the mean-of-fits, mean stretch n, summed wall time in s)."""
    fin = [r["final"] for r in recs]
    t2 = np.array([f["t2_ms"] for f in fin], float)
    good = np.isfinite(t2) & (t2 > 0)
    return {"directions": recs, "converged": all(r["converged"] for r in recs),
            "final": {"t2_ms": _log_mean(t2, strict=True),
                      "t2_ms_mean_of_fits": _log_mean(np.array([f["t2_ms_mean_of_fits"] for f in fin], float)),
                      "stretch_n": float(np.nanmean([f["stretch_n"] for f in fin])),
                      "direction_t2_ms": t2.tolist(),
                      "direction_spread_log10": float(np.ptp(np.log10(t2[good]))) if good.sum() > 1 else 0.0,
                      "wall_s": round(sum(st.get("wall_s", 0.0) for r in recs for st in r["ladder"]), 2)}}


def _isotopes(cell, pc=None, elements=None) -> dict | None:
    """PyCCE's isotope abundances for the cell ({element: {isotope: fraction}}), for the comparison
    with Kanai's isotopetable.csv (T-010 check point 3). None if neither source exposes them.

    `cell.isotopes` holds only user-added isotopes and is {} by default (006-e0-diag, S012);
    gen_supercell then uses PyCCE's natural abundances, `common_concentrations`, read here for
    `elements`."""
    iso = getattr(cell, "isotopes", None)
    if not (isinstance(iso, dict) and iso) and pc is not None:
        common = getattr(pc, "common_concentrations", None)
        if common is None:
            common = getattr(getattr(pc, "bath", None), "common_concentrations", None)
        if common is None:  # pycce.bath.array, where PyCCE 1.1 defines it
            common = getattr(getattr(getattr(pc, "bath", None), "array", None), "common_concentrations", None)
        if isinstance(common, dict):
            iso = {el: common[el] for el in sorted(set(elements or common)) if el in common}
    if not isinstance(iso, dict) or not iso:
        return None
    return {str(el): {str(k): float(v) for k, v in dict(d).items()} for el, d in iso.items()}


def _symbols(atoms) -> list[str] | None:
    """Chemical symbols of an ase Atoms (None for the test fakes, which pass atoms=None)."""
    get = getattr(atoms, "get_chemical_symbols", None)
    return list(get()) if get else None


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def protocol_hash(proto: CCEProtocol) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(asdict(proto), sort_keys=True, default=list).encode()).hexdigest()[:12]
