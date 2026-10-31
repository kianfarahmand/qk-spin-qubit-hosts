"""Nuclear-spin-class census for the E2 CCE budget (T-011, S016; ADR-0004 cost table).

From composition and volume only (no structure): the expected spinful nuclei per Å^3, the expected count inside
a sphere of the production r_bath (50 Å), the share of those spins with I > 1/2, and the share from isotopes with
natural abundance >= 99 % (the ADR-0004 item 6 `n_conf` rule; same arithmetic as
`labels.pycce_runner.dense_bath_share`). A host's cost class follows the ADR-0004 cost table:

| class | rule | ADR-0004 cost (3 directions) |
|---|---|---|
| `excluded` | an element missing from Kanai's isotope table (Kanai T2 = 1e-20 s), or no spinful nuclei | not labelled |
| `dense_half` / `dense_gt_half` | dense share >= 0.99; split by the I of the dense spins | 1 configuration; ~36 min measured for 27Al |
| `half` | share of spins with I > 1/2 below `gt_half_min_share` | 3-9 min |
| `gt_half` | otherwise | 2.8-7.5 h (ZnO, MgO) |

Flags: `floor` (fewer than `floor_spins` spins inside r_bath; r_bath grows) and `cap` (more than `cap_spins`).
`gt_half_min_share` (default 0.05) is a census choice, not an ADR value: alpha-quartz (17O share 0.016) ran at the
I = 1/2 cost in E0, ZnO and MgO (share 1.0) at the I > 1/2 cost; hosts in between are unmeasured.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

R_BATH_A = 50.0
FLOOR_SPINS = 1000
CAP_SPINS = 1500
DENSE_MIN_ABUNDANCE = 0.99
DENSE_MIN_SHARE = 0.99


def spin_census(counts: Mapping[str, float], volume_A3: float, iso: pd.DataFrame, r_bath_A: float = R_BATH_A,
                gt_half_min_share: float = 0.05, floor_spins: int = FLOOR_SPINS, cap_spins: int = CAP_SPINS) -> dict:
    """Census row for one host. counts: atoms per volume_A3 (any consistent unit, e.g. per formula unit);
    iso: `load_kanai_isotope_table` frame (abundance as a fraction, spinful isotopes only)."""
    known = set(iso["element"])
    missing = sorted(e for e in counts if e not in known and e not in ("Ce", "Ar"))
    rows = iso[iso["element"].isin(counts) & (iso["spin_I"] > 0) & (iso["abundance"] > 0)]
    per_cell = np.array([counts[e] * a for e, a in zip(rows["element"], rows["abundance"])], float)
    tot = float(per_cell.sum())
    spin = rows["spin_I"].to_numpy(float)
    dense_mask = rows["abundance"].to_numpy(float) >= DENSE_MIN_ABUNDANCE
    rho = tot / volume_A3 if volume_A3 > 0 else float("nan")  # spins per Å^3
    n_bath = rho * 4.0 / 3.0 * np.pi * r_bath_A ** 3
    share_gt = float(per_cell[spin > 0.5].sum() / tot) if tot > 0 else float("nan")
    dense_share = float(per_cell[dense_mask].sum() / tot) if tot > 0 else 0.0
    if missing or tot == 0:
        cls = "excluded"
    elif dense_share >= DENSE_MIN_SHARE:
        dense_gt = per_cell[dense_mask & (spin > 0.5)].sum() > 0.5 * per_cell[dense_mask].sum()
        cls = "dense_gt_half" if dense_gt else "dense_half"
    elif share_gt < gt_half_min_share:
        cls = "half"
    else:
        cls = "gt_half"
    dom = int(np.argmax(per_cell)) if tot > 0 else None
    return {"spins_per_A3": rho, "n_bath_50A": float(n_bath), "share_I_gt_half": share_gt, "dense_share": dense_share,
            "dominant_isotope": rows["isotope"].iloc[dom] if dom is not None else "",
            "I_max": float(spin.max()) if len(spin) else 0.0, "missing_elements": ",".join(missing),
            "cost_class": cls, "floor": bool(tot > 0 and n_bath < floor_spins), "cap": bool(n_bath > cap_spins)}
