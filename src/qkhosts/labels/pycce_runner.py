"""PyCCE label generation (E0 pilot / E2 production).

Design is fixed here; numeric settings come from ADR-0004 after the E0 pilot.
The PyCCE calls below follow the documented PyCCE workflow (BathCell -> gen_supercell ->
Simulator -> compute) but MUST be checked against the installed PyCCE version during E0
(task T-010) — argument names have changed between releases.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CCEProtocol:
    central_spin: float = 0.5           # S
    central_g: float = 2.0
    site_rule: str = "host_atom_removed_avg_inequivalent"  # ADR-0004
    n_sites_max: int = 3
    b_field_T: float = 5.0
    b_direction: tuple = (0.0, 0.0, 1.0)  # relative to lattice c axis; ADR-0004
    order: int = 2
    r_bath_A: float = 40.0
    r_dipole_A: float = 8.0
    supercell: int = 200
    n_conf: int = 20                    # random isotope configurations
    t_max_ms: float = 50.0
    n_t: int = 201
    pulses: int = 1                     # Hahn echo
    converge_tol_log10: float = 0.02    # |Δlog10 T2| between successive (order, r_bath) steps
    seed: int = 0
    extra: dict = field(default_factory=dict)


def fit_t2(t_ms: np.ndarray, coherence: np.ndarray) -> tuple[float, float]:
    """Fit |L(t)| = exp(-(t/T2)^n), t = total evolution time; returns (T2_ms, stretch exponent n).
    Uses linearisation log(-log L) = n log t - n log T2 on 0.05 < L < 0.95.
    NOTE (E0): confirm whether PyCCE's time axis is total or half echo time, and match Kanai's
    T2 definition, before any label is produced."""
    L = np.abs(np.asarray(coherence))
    m = (L > 0.05) & (L < 0.95) & (t_ms > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    x, yv = np.log(t_ms[m]), np.log(-np.log(L[m]))
    n, b = np.polyfit(x, yv, 1)
    return float(np.exp(-b / n)), float(n)


def run_material(structure_path: str | Path, proto: CCEProtocol, out_dir: str | Path) -> dict:
    """Compute ensemble-averaged Hahn-echo T2 for one host. Resumable: skips if result exists."""
    out = Path(out_dir) / (Path(structure_path).stem + ".json")
    if out.exists():
        return json.loads(out.read_text())
    try:
        import pycce as pc  # noqa: F401
        from pymatgen.core import Structure  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("install extras: pip install -e .[cce]") from e
    raise NotImplementedError(
        "Implement in T-010 (E0): Structure -> ase Atoms -> pc.BathCell.from_ase -> gen_supercell(remove site, "
        "isotope seed) -> pc.Simulator(spin=CenterArray(S, g), bath, r_bath, r_dipole, order, magnetic_field, "
        "pulses) -> compute(timespace, method='cce', quantity='coherence'); loop n_conf seeds; fit_t2; "
        "convergence escalation; write JSON with protocol=asdict(proto), t2_ms, t2_sem, stretch_n, converged.")


def protocol_hash(proto: CCEProtocol) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(asdict(proto), sort_keys=True, default=list).encode()).hexdigest()[:12]
