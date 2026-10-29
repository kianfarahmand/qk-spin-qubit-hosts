"""PyCCE label generation (E0 pilot / E2 production).

Settings come from the T-005 literature read (S005; `literature/notes/kanai2022_si.md`,
`literature/notes/toriyama2025.md`) and are frozen in ADR-0004 after the E0 pilot (T-010):
- B = 5 T along z (PyCCE takes gauss: 5e4 G); S = 1/2, g = 2 (Kanai Table S1 conditions).
- Hahn echo with `as_delay=False`: t is the total free-evolution time, as in Kanai's T2.
- Central spin at a vacancy on a host site (Toriyama's code), averaged over up to n_sites_max
  inequivalent sites, weighted by multiplicity (our convention; E0 measures whether it matters).
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
    site_rule: str = "host_atom_removed_avg_inequivalent"  # ADR-0004
    n_sites_max: int = 3
    b_field_T: float = 5.0
    b_direction: tuple = (0.0, 0.0, 1.0)  # Cartesian; pymatgen puts the c axis along z
    order: int = 2
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


def _center_position(cell, frac: np.ndarray) -> np.ndarray:
    """Cartesian position (Å) of a cell-0 site. PyCCE stores lattice vectors as columns of cell.cell."""
    if hasattr(cell, "to_cartesian"):
        return np.asarray(cell.to_cartesian(frac), float)
    return np.asarray(cell.cell, float) @ np.asarray(frac, float)


def _coherence(pc, bath, pos, ts, r_bath, r_dipole, proto: CCEProtocol) -> np.ndarray:
    field_G = np.asarray(proto.b_direction, float)
    field_G = proto.b_field_T * GAUSS_PER_TESLA * field_G / np.linalg.norm(field_G)
    calc = pc.Simulator(spin=proto.central_spin, position=pos, bath=bath, r_bath=r_bath, r_dipole=r_dipole,
                        order=proto.order, magnetic_field=field_G, pulses=proto.pulses, as_delay=proto.as_delay)
    return np.abs(np.asarray(calc.compute(ts, method="cce", quantity="coherence"), float))


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


def _site_step(pc, cell, element, frac, r_bath, r_dipole, proto, ts=None) -> dict:
    """Ensemble over n_conf isotope seeds for one site and one ladder step."""
    pos = _center_position(cell, frac)
    baths = [cell.gen_supercell(proto.supercell, remove=[(element, list(frac))], seed=proto.seed + 1 + k)
             for k in range(proto.n_conf)]
    tried = None
    if ts is None:
        ts, L0, tried = _adapt_window(lambda t: _coherence(pc, baths[0], pos, t, r_bath, r_dipole, proto), proto)
        curves = [L0]
    else:
        curves = [_coherence(pc, baths[0], pos, ts, r_bath, r_dipole, proto)]
    curves += [_coherence(pc, b, pos, ts, r_bath, r_dipole, proto) for b in baths[1:]]
    curves = np.array(curves)
    t2_avg, n_avg = fit_t2(ts, curves.mean(axis=0))
    per = np.array([fit_t2(ts, c)[0] for c in curves])
    ok = per[np.isfinite(per)]
    xyz = np.asarray(getattr(baths[0], "xyz", np.empty((0, 3))), float).reshape(-1, 3)
    return {"t2_ms": t2_avg, "stretch_n": n_avg,
            "t2_ms_mean_of_fits": float(ok.mean()) if len(ok) else float("nan"),
            "t2_sem_ms": float(ok.std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else float("nan"),
            "n_fit_failed": int(len(per) - len(ok)), "n_bath": int(len(baths[0])),
            "min_bath_dist_A": float(np.linalg.norm(xyz - pos, axis=1).min()) if len(xyz) else float("nan"),
            "t_max_ms": float(ts[-1]), "windows_tried_ms": tried, "_ts": ts}


def _converge_site(pc, cell, element, frac, proto: CCEProtocol) -> dict:
    """Climb the (r_bath, r_dipole) ladder until |Δlog10 T2| < tol between successive steps.
    The same isotope seeds and time window are used at every step (paired comparison)."""
    steps, ts = [], None
    for r_bath, r_dipole in proto.ladder:
        t0 = time.time()
        st = _site_step(pc, cell, element, frac, r_bath, r_dipole, proto, ts)
        ts = st.pop("_ts")
        st.update(r_bath_A=r_bath, r_dipole_A=r_dipole, wall_s=round(time.time() - t0, 2))
        steps.append(st)
        if len(steps) >= 2 and np.isfinite(steps[-1]["t2_ms"]) and np.isfinite(steps[-2]["t2_ms"]):
            if abs(np.log10(steps[-1]["t2_ms"] / steps[-2]["t2_ms"])) < proto.converge_tol_log10:
                return {"converged": True, "final": steps[-1], "ladder": steps}
    return {"converged": False, "final": steps[-1], "ladder": steps}


def run_material(structure_path: str | Path, proto: CCEProtocol, out_dir: str | Path) -> dict:
    """Compute ensemble-averaged Hahn-echo T2 (ms) for one host. Resumable: skips if the result exists.

    Output JSON: protocol, per-site results (each with its convergence ladder), and the
    multiplicity-weighted host values t2_ms (log-mean over sites), converged (all sites), cost."""
    out = Path(out_dir) / (Path(structure_path).stem + ".json")
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    try:
        import pycce as pc
    except ImportError as e:  # pragma: no cover
        raise ImportError("install extras: pip install -e .[cce]") from e
    t0 = time.time()
    atoms, sites = _load_sites(structure_path, proto.n_sites_max)
    cell = pc.read_ase(atoms) if hasattr(pc, "read_ase") else pc.BathCell.from_ase(atoms)
    per_site = []
    for element, frac, mult in sites:
        r = _converge_site(pc, cell, element, np.asarray(frac, float), proto)
        per_site.append({"element": element, "frac": [float(x) for x in frac], "multiplicity": int(mult), **r})
    w = np.array([s["multiplicity"] for s in per_site], float)
    t2s = np.array([s["final"]["t2_ms"] for s in per_site], float)
    good = np.isfinite(t2s) & (t2s > 0)
    t2 = float(10 ** np.average(np.log10(t2s[good]), weights=w[good])) if good.any() else float("nan")
    res = {
        "structure": Path(structure_path).name, "protocol": asdict(proto), "protocol_hash": protocol_hash(proto),
        "t2_ms": t2, "log10_t2_ms": float(np.log10(t2)) if np.isfinite(t2) else float("nan"),
        "site_spread_log10": float(np.ptp(np.log10(t2s[good]))) if good.sum() > 1 else 0.0,
        "converged": bool(good.all() and all(s["converged"] for s in per_site)),
        "n_sites": len(per_site), "sites": per_site, "wall_s": round(time.time() - t0, 2),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    return res


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def protocol_hash(proto: CCEProtocol) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(asdict(proto), sort_keys=True, default=list).encode()).hexdigest()[:12]
