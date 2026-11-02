"""Wall-time model for one CCE label on a free Actions runner (E2 budget; T-070 -> T-015, S019).

One *evaluation* is one PyCCE `compute` call: one bath configuration, one field direction, one time window
(CCE-2, n_t = 201). Model, in seconds:

    t_eval = a * n_pairs * W(p) + b * n_supercell

- n_pairs = N^2 (r_dipole / r_bath)^3 / 2: pair clusters inside r_dipole for N spins inside r_bath of a uniform
  bath. The ADR-0004 cap and floor scale r_bath and r_dipole together, so r_dipole / r_bath stays 15/50 = 0.3 and
  n_pairs depends on N only (N = the nominal count clipped to [floor, cap]).
- W(p) = sum_{k,l} f_k f_l (d_k d_l)^p = (sum_k f_k d_k^p)^2: mean cost weight of a pair, with f_k the share of bath
  spins from isotope k and d_k = 2 I_k + 1 (a pair's Hilbert space is 2 d_k d_l).
- n_supercell: spins in the generated supercell (bath generation and the distance filter scale with it).

A label is n_directions x (n_conf + extra windows) evaluations: the window search (`pycce_runner._adapt_window`)
runs on the first configuration only, so each extra window costs one evaluation.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

R_DIP_OVER_R_BATH = 15.0 / 50.0
SUPERCELL_SAFE = 0.4      # = pycce_runner.SUPERCELL_SAFE
SUPERCELL_A = 200.0       # = CCEProtocol.supercell default
R_BATH_A = 50.0


def bath_mix(counts: Mapping[str, float], isotopes: Mapping[str, Mapping[str, float]],
             spin_I: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """(f, d): share of bath spins per spinful isotope and its dimension 2I+1.
    counts: atoms per cell (any consistent unit); isotopes: element -> {isotope: abundance fraction};
    spin_I: isotope -> I. Elements without an entry in `isotopes` carry no bath spins."""
    w, d = [], []
    for el, n in counts.items():
        for iso, ab in isotopes.get(el, {}).items():
            I = float(spin_I[iso])
            if I > 0 and ab > 0:
                w.append(float(n) * float(ab))
                d.append(2.0 * I + 1.0)
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return np.zeros(0), np.zeros(0)
    return w / w.sum(), np.asarray(d, float)


def isotopes_from_table(iso: pd.DataFrame, elements) -> dict[str, dict[str, float]]:
    """element -> {isotope: abundance} for the spinful isotopes of `iso` (`load_kanai_isotope_table` frame)."""
    sub = iso[iso["element"].isin(list(elements)) & (iso["spin_I"] > 0) & (iso["abundance"] > 0)]
    out: dict[str, dict[str, float]] = {}
    for el, name, ab in zip(sub["element"], sub["isotope"], sub["abundance"]):
        out.setdefault(el, {})[name] = float(ab)
    return out


def pair_weight(f: np.ndarray, d: np.ndarray, p: float) -> float:
    """W(p) = (sum_k f_k d_k^p)^2 (dimensionless; 1 for a pure I = 0 bath, 2^(2p) for a pure I = 1/2 bath)."""
    return float(np.dot(f, np.asarray(d, float) ** p) ** 2)


def n_pairs(n_in_rbath: float, ratio: float = R_DIP_OVER_R_BATH) -> float:
    """Expected pair clusters inside r_dipole for n spins inside r_bath (uniform bath, edge effects ignored)."""
    return 0.5 * float(n_in_rbath) ** 2 * ratio ** 3


def bath_used(n_nominal: float, floor: float | None, cap: float | None) -> float:
    """Spins inside r_bath after the ADR-0004 floor and cap (nominal = count inside 50 Å)."""
    n = float(n_nominal)
    if cap is not None and n > cap:
        n = float(cap)
    if floor is not None and n < floor:
        n = float(floor)
    return n


def supercell_spins(n_nominal: float, floor: float | None, k_sc: float) -> float:
    """Spins in the generated supercell. k_sc = supercell spins per nominal spin at the default size (measured,
    ~16). A floored bath whose floor radius exceeds SUPERCELL_SAFE x size grows the supercell to enclose 1.2 x floor
    spins (`pycce_runner.floor_radii`), so the count is at least k_sc * 1.2 * floor * (R_BATH / (SAFE * size))^3."""
    lo = 0.0 if floor is None else 1.2 * float(floor) * (R_BATH_A / (SUPERCELL_SAFE * SUPERCELL_A)) ** 3
    return float(k_sc) * max(float(n_nominal), lo)


def extra_windows(t2_ms: float) -> int:
    """Extra evaluations from the window search, from an a-priori T2 (ms): the first window is 50 ms; it shrinks
    once for 0.03 ms <= T2 < ~10 ms and twice below, and may widen once for slow baths (010, 007-009 records).
    Conservative: 2 below 0.03 ms, otherwise 1."""
    return 2 if t2_ms < 0.03 else 1


def eval_seconds(params: Mapping[str, float], pairs: float, w: float, n_sc: float) -> float:
    """t_eval (s) = a * pairs * w + b * n_sc; params keys a, b (p is already inside w)."""
    return float(params["a"]) * pairs * w + float(params["b"]) * n_sc


def fit_cost_model(pairs: np.ndarray, f_d: list[tuple[np.ndarray, np.ndarray]], n_sc: np.ndarray,
                   t_eval: np.ndarray, p0: float = 2.0) -> dict:
    """Least squares in log t_eval for (a, b, p). Returns a, b, p and the rms log10 residual (dex)."""
    from scipy.optimize import least_squares

    pairs, n_sc, t_eval = (np.asarray(x, float) for x in (pairs, n_sc, t_eval))

    def model(th):
        a, b, p = np.exp(th[0]), np.exp(th[1]), th[2]
        w = np.array([pair_weight(f, d, p) for f, d in f_d])
        return a * pairs * w + b * n_sc

    def resid(th):
        return np.log10(model(th)) - np.log10(t_eval)

    w0 = np.array([pair_weight(f, d, p0) for f, d in f_d])
    a0 = np.median(t_eval / (pairs * w0))
    sol = least_squares(resid, x0=[np.log(a0), np.log(1e-6), p0], bounds=([-60, -60, 0.0], [10, 10, 4.0]))
    r = resid(sol.x)
    return {"a": float(np.exp(sol.x[0])), "b": float(np.exp(sol.x[1])), "p": float(sol.x[2]),
            "rms_dex": float(np.sqrt(np.mean(r ** 2))), "max_abs_dex": float(np.max(np.abs(r))), "n": int(len(r))}
