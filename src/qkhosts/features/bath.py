"""Nuclear-spin-bath descriptors from composition + cell volume.

Inputs per material: element counts per cell (dict element -> count) and cell volume (A^3).
Isotope number density n_i [cm^-3] = count_el * abundance_i / (V_A3 * 1e-24).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from qkhosts.physics.scaling_law import (DENSE_THRESHOLD_CM3, ScalingLawParams, combine,
                                         dominant_share, t2_isotope)

A3_TO_CM3 = 1e-24


def isotope_densities(counts: Mapping[str, float], volume_A3: float, iso: pd.DataFrame) -> pd.DataFrame:
    """Rows = spinful isotopes present, with n_cm3 added."""
    if volume_A3 <= 0:
        raise ValueError("volume must be positive")
    rows = iso[iso["element"].isin(counts) & (iso["spin_I"] > 0) & (iso["abundance"] > 0)].copy()
    rows["n_cm3"] = [counts[e] * a / (volume_A3 * A3_TO_CM3) for e, a in zip(rows["element"], rows["abundance"])]
    return rows.reset_index(drop=True)


def bath_descriptors(counts: Mapping[str, float], volume_A3: float, iso: pd.DataFrame,
                     p: ScalingLawParams) -> dict[str, float]:
    r = isotope_densities(counts, volume_A3, iso)
    if len(r) == 0:
        return {"log10_t2_sl": np.inf, "n_spinful_species": 0, "n_spinful_total": 0.0, "log10_n_max": -np.inf,
                "dominant_share": 0.0, "is_dense": 0.0, "min_rel_gamma_gap": np.inf, "g_abs_max": 0.0, "I_max": 0.0}
    t2i = t2_isotope(r["g_factor"].to_numpy(), r["spin_I"].to_numpy(), r["n_cm3"].to_numpy(), p)
    t2 = combine(t2i, p.eta)
    g = np.abs(r["g_factor"].to_numpy())
    gaps = [abs(a - b) / max(a, b) for i, a in enumerate(g) for b in g[i + 1:] if max(a, b) > 0]
    return {
        "log10_t2_sl": float(np.log10(t2)),
        "n_spinful_species": int(len(r)),
        "n_spinful_total": float(r["n_cm3"].sum()),
        "log10_n_max": float(np.log10(r["n_cm3"].max())),
        "dominant_share": dominant_share(t2i, p.eta),
        "is_dense": float((r["n_cm3"] > DENSE_THRESHOLD_CM3).any()),
        "min_rel_gamma_gap": float(min(gaps)) if gaps else np.inf,
        "g_abs_max": float(g.max()),
        "I_max": float(r["spin_I"].max()),
    }
