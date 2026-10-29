"""Analytic nuclear-spin-bath T2 scaling law (Kanai et al., PNAS 2022).

Per isotope i (dilute regime, n_i < ~1e22 cm^-3):
    T2_i = A * |g_i|**alpha * I_i**beta * n_i**gamma            [s], n_i in cm^-3
Compound:
    T2 = ( sum_i T2_i**(-eta) )**(-1/eta)                       (eta ~ 2)

Constants (verified in T-003, ADR-0002): A=1.5e18 s cm^-3, alpha=-1.65, beta=-1.09, gamma=-1.0,
eta=2. Eq. 2 of the PNAS paper prints alpha, beta rounded to -1.6, -1.1; the authors' Zenodo script
T2_Listup_v1.0.py uses -1.65, -1.09, and only those reproduce the published values. Read them from
configs/base.yaml; do not hard-code them at call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

DENSE_THRESHOLD_CM3 = 1e22  # upper validity bound of the n^-1 regime stated by Kanai et al.


@dataclass(frozen=True)
class ScalingLawParams:
    A: float = 1.5e18
    alpha: float = -1.65
    beta: float = -1.09
    gamma: float = -1.0
    eta: float = 2.0
    verified: bool = False  # True only via configs/base.yaml (T-003 done)

    @classmethod
    def from_config(cls, cfg: Mapping) -> "ScalingLawParams":
        # float(): PyYAML (YAML 1.1) loads "1.5e18" without a sign/dot rule match as a string.
        vals = {k: float(cfg[k]) for k in ("A", "alpha", "beta", "gamma", "eta") if k in cfg}
        return cls(**vals, verified=bool(cfg.get("verified", False)))


def t2_isotope(g: np.ndarray, spin_I: np.ndarray, n_cm3: np.ndarray, p: ScalingLawParams) -> np.ndarray:
    """Per-isotope T2 contribution in seconds. Spinless isotopes (I=0) or n=0 give +inf."""
    g, spin_I, n = (np.asarray(a, dtype=float) for a in (g, spin_I, n_cm3))
    out = np.full(np.broadcast(g, spin_I, n).shape, np.inf)
    ok = (spin_I > 0) & (n > 0) & (np.abs(g) > 0)
    out[ok] = p.A * np.abs(g[ok]) ** p.alpha * spin_I[ok] ** p.beta * n[ok] ** p.gamma
    return out


def combine(t2_i: np.ndarray, eta: float) -> float:
    """Combine per-isotope T2_i into compound T2 (s). All-inf (spin-free bath) -> inf."""
    t2_i = np.asarray(t2_i, dtype=float)
    rates = np.where(np.isfinite(t2_i), t2_i ** (-eta), 0.0)
    s = rates.sum()
    return float("inf") if s == 0 else float(s ** (-1.0 / eta))


def t2_compound(g, spin_I, n_cm3, p: ScalingLawParams) -> float:
    return combine(t2_isotope(g, spin_I, n_cm3, p), p.eta)


def dominant_share(t2_i: np.ndarray, eta: float) -> float:
    """Fraction of the total decoherence 'rate' sum_i T2_i^-eta carried by the largest term."""
    t2_i = np.asarray(t2_i, dtype=float)
    r = np.where(np.isfinite(t2_i), t2_i ** (-eta), 0.0)
    return float(r.max() / r.sum()) if r.sum() > 0 else 0.0
