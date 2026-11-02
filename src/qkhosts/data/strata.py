"""Regime strata S1-S4 and the stratified sampler for the E2 label set (proposal v2 §4 "Stratified design"; T-014).

Strata (membership flags can overlap; each host gets one primary stratum by `priority`):
- S1 dense: any spinful isotope with n_i > 1e22 cm^-3 (`bath_is_dense`; Kanai's validity bound of the n^-1 law).
- S2 heteronuclear with close γ (`s2_rule="gap"`, proposal v2): smallest relative |γ| gap between spinful
  isotopes (`bath_min_rel_gamma_gap`, dimensionless) below `gap_max`. Caveat (S018): at 5 T, without quadrupole
  terms, two species flip-flop only if |Δγ|B is about as small as their dipolar coupling, i.e. a relative gap of
  ~1e-5-1e-4; no D-SL host goes below 2e-4, so any practical `gap_max` selects frozen pairs.
  Adopted instead (Q22, ADR-0007; `s2_rule="mixed"`, the default): a mixed bath, i.e. >= 2 spinful isotopes and the largest one's share of the
  rate sum Σ T2_i^-η (`bath_dominant_share`) below `mixed_max`; this is where the scaling law's combination rule
  carries the prediction.
- S3 anisotropic or layered: bond-graph dimensionality below 3 (`is_low_dim` at the chosen bond scale) or
  reduced-cell anisotropy (`reduced_anisotropy`, dimensionless) at or above `aniso_min`. ADR-0007 uses
  dimensionality alone at scale 1.2 (`aniso_min` = inf): anisotropy flags polytypes such as 4H-SiC.
- S4 dilute control: none of S1-S3.

Sampler: within each stratum a Latin hypercube over (log10 T2_SL, chemistry), both rank-transformed to [0, 1];
chemistry is the composition-weighted mean atomic number Z. Each LHS point takes the nearest unused host whose
chemical system (set of elements) is still under `cap_per_chemsys`. A stratum with fewer hosts than its share
gives the rest to the other strata in proportion to their shares.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

STRATA = ("S1", "S2", "S3", "S4")


@dataclass(frozen=True)
class StrataThresholds:
    """Defaults = Q22 (Kian, S018; ADR-0007): S2 mixed bath < 0.6, S3 low-dim at bond scale 1.2 only."""
    s2_rule: str = "mixed"              # "mixed" (ADR-0007) or "gap" (proposal v2 original, kept for comparison)
    gap_max: float = 0.01               # S2 "gap": relative γ gap (dimensionless)
    mixed_max: float = 0.6              # S2 "mixed": largest isotope's share of Σ T2_i^-η (dimensionless)
    aniso_min: float = float("inf")     # S3: reduced-cell max/min length (dimensionless); inf = not used
    bond_scale: float = 1.2             # S3: which is_low_dim column (bond tolerance of T-013)
    priority: tuple[str, ...] = ("S3", "S2", "S1")  # primary stratum: first matching flag, else S4


def low_dim_col(scale: float) -> str:
    """Column name of is_low_dim at a bond scale, e.g. 1.2 -> 'is_low_dim_s120'."""
    return f"is_low_dim_s{round(scale * 100):d}"


def assign_strata(df: pd.DataFrame, th: StrataThresholds = StrataThresholds()) -> pd.DataFrame:
    """Add in_S1, in_S2, in_S3 (bool) and `stratum` (primary, 'S1'..'S4'). Needs bath_is_dense,
    bath_min_rel_gamma_gap, reduced_anisotropy and the is_low_dim column of th.bond_scale. A missing
    structural value (NaN) counts as not in S3."""
    out = df.copy()
    out["in_S1"] = out["bath_is_dense"].fillna(0).astype(float) > 0
    if th.s2_rule == "gap":
        out["in_S2"] = out["bath_min_rel_gamma_gap"].astype(float) < th.gap_max
    elif th.s2_rule == "mixed":
        out["in_S2"] = (out["bath_dominant_share"].astype(float) < th.mixed_max) & (out["bath_n_spinful_species"] >= 2)
    else:
        raise ValueError(f"unknown s2_rule {th.s2_rule!r}")
    low = out[low_dim_col(th.bond_scale)].fillna(0).astype(float) > 0
    out["in_S3"] = low | (out["reduced_anisotropy"].astype(float) >= th.aniso_min)
    stratum = pd.Series("S4", index=out.index)
    for s in reversed(th.priority):
        stratum[out[f"in_{s}"]] = s
    out["stratum"] = stratum
    return out


def parse_composition(c: str | Mapping[str, float]) -> dict[str, float]:
    """D-SL's `composition` column (JSON element -> count per cell) as a dict."""
    return dict(c) if isinstance(c, Mapping) else json.loads(c)


def chemical_system(c: str | Mapping[str, float]) -> str:
    """Sorted element set joined by '-', e.g. 'Ca-O-S'."""
    return "-".join(sorted(parse_composition(c)))


def mean_z(c: str | Mapping[str, float], z: Mapping[str, int]) -> float:
    """Composition-weighted mean atomic number (dimensionless); NaN if an element has no Z."""
    comp = parse_composition(c)
    if any(e not in z for e in comp):
        return float("nan")
    n = sum(comp.values())
    return float(sum(z[e] * k for e, k in comp.items()) / n)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank transform to (0, 1): (rank + 0.5) / n, ties by order of appearance."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return (r + 0.5) / max(len(x), 1)


def lhs_pick(u: np.ndarray, chemsys: Sequence[str], m: int, rng: np.random.Generator,
             cap_per_chemsys: int, used: Mapping[str, int] | None = None) -> list[int]:
    """Pick up to m row indices of u (n x d, in [0, 1]) nearest to m Latin-hypercube points, each chemical system
    at most cap_per_chemsys times (counting `used`). Returns positions into u."""
    n, d = u.shape
    m = min(m, n)
    pts = (np.stack([rng.permutation(m) for _ in range(d)], axis=1) + rng.random((m, d))) / m
    count = dict(used or {})
    free = np.ones(n, bool)
    picked: list[int] = []
    for p in pts[rng.permutation(m)]:
        ok = free & np.array([count.get(cs, 0) < cap_per_chemsys for cs in chemsys])
        if not ok.any():
            break
        cand = np.flatnonzero(ok)
        j = int(cand[np.argmin(((u[cand] - p) ** 2).sum(axis=1))])
        picked.append(j)
        free[j] = False
        count[chemsys[j]] = count.get(chemsys[j], 0) + 1
    return picked


def allocate(n_total: int, sizes: Mapping[str, int], shares: Mapping[str, float]) -> dict[str, int]:
    """Integer allocation per stratum: n_total split by shares (largest remainder), no stratum above its size,
    any excess passed on to the strata with room, in proportion to their shares."""
    alloc = {s: 0 for s in shares}
    left, open_ = n_total, {s for s in shares if sizes.get(s, 0) > 0}
    while left > 0 and open_:
        tot = sum(shares[s] for s in open_)
        raw = {s: left * shares[s] / tot for s in open_}
        add = {s: int(np.floor(v)) for s, v in raw.items()}
        for s in sorted(open_, key=lambda s: raw[s] - add[s], reverse=True)[: left - sum(add.values())]:
            add[s] += 1
        for s in open_:
            alloc[s] += add[s]
        over = {s: alloc[s] - sizes[s] for s in open_ if alloc[s] > sizes[s]}
        for s, k in over.items():
            alloc[s] = sizes[s]
        left = sum(over.values())
        open_ = {s for s in open_ if alloc[s] < sizes[s]}
    return alloc


@dataclass(frozen=True)
class SamplerConfig:
    n_total: int = 500
    shares: Mapping[str, float] = field(default_factory=lambda: {s: 0.25 for s in STRATA})
    cap_per_chemsys: int = 2
    max_e_hull_eV: float = 0.0          # keep hosts on the current MP hull (Kanai's universe is on-hull)
    force_top_n: int = 25               # Q22(d): the n highest-T2_SL hosts are always in (rank LHS misses the tail)
    seed: int = 0


def stratified_sample(df: pd.DataFrame, cfg: SamplerConfig = SamplerConfig()) -> pd.DataFrame:
    """Rows of df (with `stratum`, `chemsys`, `mean_z`, `bath_log10_t2_sl`) chosen by the stratified LHS sampler.
    Excluded before sampling: cost_class 'excluded', a missing element, e_hull above max_e_hull_eV, NaN axes.
    First the `force_top_n` hosts with the highest bath_log10_t2_sl (chemical-system cap respected) are taken and
    counted against their own stratum's allocation; the LHS fills the rest. Adds `sample_stratum` and
    `forced_tail` (bool). The chemical-system cap is global across strata."""
    pool = df[(df["cost_class"] != "excluded") & df["missing_elements"].isna()
              & (df["e_hull_now_eV"] <= cfg.max_e_hull_eV + 1e-12)
              & np.isfinite(df["bath_log10_t2_sl"]) & np.isfinite(df["mean_z"])]
    sizes = pool["stratum"].value_counts().to_dict()
    alloc = allocate(cfg.n_total, sizes, cfg.shares)
    rng = np.random.default_rng(cfg.seed)
    used: dict[str, int] = {}
    tail_idx = []
    for i, cs in pool.sort_values("bath_log10_t2_sl", ascending=False, kind="stable")["chemsys"].items():
        if len(tail_idx) >= min(cfg.force_top_n, cfg.n_total):
            break
        if used.get(cs, 0) < cfg.cap_per_chemsys:
            tail_idx.append(i)
            used[cs] = used.get(cs, 0) + 1
    tail = pool.loc[tail_idx]
    parts = [tail.assign(sample_stratum=tail["stratum"], forced_tail=True)] if len(tail) else []
    for s, k in tail["stratum"].value_counts().items():
        alloc[s] = max(alloc.get(s, 0) - int(k), 0)
    pool = pool.drop(index=tail_idx)
    for s in STRATA:
        sub = pool[pool["stratum"] == s]
        if alloc.get(s, 0) == 0 or len(sub) == 0:
            continue
        u = np.column_stack([_rank01(sub["bath_log10_t2_sl"].to_numpy()), _rank01(sub["mean_z"].to_numpy())])
        cs = sub["chemsys"].tolist()
        idx = lhs_pick(u, cs, alloc[s], rng, cfg.cap_per_chemsys, used)
        for j in idx:
            used[cs[j]] = used.get(cs[j], 0) + 1
        parts.append(sub.iloc[idx].assign(sample_stratum=s, forced_tail=False))
    if not parts:
        return pool.iloc[:0].assign(sample_stratum=pd.Series(dtype=str), forced_tail=pd.Series(dtype=bool))
    return pd.concat(parts)
