"""Strata S1-S4, the stratified LHS sampler (T-014) and the reduced-cell anisotropy. Toy tables only."""
import json

import numpy as np
import pandas as pd
import pytest

from qkhosts.data.strata import (SamplerConfig, StrataThresholds, allocate, assign_strata, chemical_system,
                                 lhs_pick, low_dim_col, mean_z, stratified_sample)
from qkhosts.features.structural import Crystal, reduced_anisotropy, reduced_basis


def test_low_dim_col():
    assert low_dim_col(1.2) == "is_low_dim_s120"
    assert low_dim_col(1.1) == "is_low_dim_s110"


def _toy():
    return pd.DataFrame({
        "bath_is_dense": [1, 0, 0, 0, 1, 0],
        "bath_min_rel_gamma_gap": [0.5, 0.001, 0.5, 0.5, 0.001, np.inf],
        "is_low_dim_s120": [0, 0, 1, 0, 0, np.nan],
        "reduced_anisotropy": [1.0, 1.0, 1.0, 5.0, 1.0, np.nan],
    })


def test_assign_strata_flags_and_priority():
    s = assign_strata(_toy(), StrataThresholds(s2_rule="gap", gap_max=0.01, aniso_min=3.0))
    assert s["in_S1"].tolist() == [True, False, False, False, True, False]
    assert s["in_S2"].tolist() == [False, True, False, False, True, False]
    assert s["in_S3"].tolist() == [False, False, True, True, False, False]  # NaN structure -> not S3
    assert s["stratum"].tolist() == ["S1", "S2", "S3", "S3", "S2", "S4"]   # row 4: S2 outranks S1
    s2 = assign_strata(_toy(), StrataThresholds(s2_rule="gap", priority=("S1", "S2", "S3")))
    assert s2["stratum"].tolist()[4] == "S1"


def test_adr0007_defaults():
    th = StrataThresholds()
    assert (th.s2_rule, th.mixed_max, th.aniso_min, th.bond_scale) == ("mixed", 0.6, float("inf"), 1.2)
    t = _toy().assign(bath_dominant_share=[1.0] * 6, bath_n_spinful_species=[1] * 6)
    assert assign_strata(t)["in_S3"].tolist() == [False, False, True, False, False, False]  # anisotropy 5 not S3


def test_assign_strata_mixed_rule():
    t = _toy().assign(bath_dominant_share=[0.9, 0.9, 0.5, 0.5, 0.5, 1.0], bath_n_spinful_species=[2, 2, 2, 1, 3, 1])
    s = assign_strata(t, StrataThresholds(s2_rule="mixed", mixed_max=0.6))
    assert s["in_S2"].tolist() == [False, False, True, False, True, False]  # row 3: one species is never mixed
    with pytest.raises(ValueError):
        assign_strata(t, StrataThresholds(s2_rule="nope"))


def test_composition_helpers():
    c = json.dumps({"O": 2, "Ca": 1})
    assert chemical_system(c) == "Ca-O"
    assert mean_z(c, {"O": 8, "Ca": 20}) == pytest.approx((20 + 16) / 3)
    assert np.isnan(mean_z(c, {"O": 8}))


def test_allocate_redistributes_excess():
    shares = {"S1": 0.25, "S2": 0.25, "S3": 0.25, "S4": 0.25}
    assert allocate(100, {"S1": 100, "S2": 100, "S3": 100, "S4": 100}, shares) == {s: 25 for s in shares}
    a = allocate(100, {"S1": 100, "S2": 5, "S3": 100, "S4": 100}, shares)
    assert a["S2"] == 5 and sum(a.values()) == 100 and max(a.values()) - min(a["S1"], a["S3"], a["S4"]) <= 1
    assert sum(allocate(1000, {"S1": 10, "S2": 10, "S3": 0, "S4": 10}, shares).values()) == 30


def test_lhs_pick_spreads_and_caps():
    rng = np.random.default_rng(0)
    u = np.column_stack([np.linspace(0.005, 0.995, 100), np.random.default_rng(1).random(100)])
    idx = lhs_pick(u, [f"c{i}" for i in range(100)], 10, rng, cap_per_chemsys=1)
    assert len(set(idx)) == 10
    # one pick per tenth of the first axis (LHS marginal), up to nearest-neighbour slack
    assert len({int(u[i, 0] * 10) for i in idx}) >= 8
    capped = lhs_pick(u, ["same"] * 100, 10, np.random.default_rng(0), cap_per_chemsys=2)
    assert len(capped) == 2


def test_stratified_sample_end_to_end():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "stratum": rng.choice(["S1", "S2", "S3", "S4"], n, p=[0.7, 0.1, 0.1, 0.1]),
        "chemsys": [f"A-B{i % 150}" for i in range(n)],
        "mean_z": rng.random(n) * 50, "bath_log10_t2_sl": rng.normal(-3, 1, n),
        "cost_class": ["excluded" if i % 50 == 0 else "gt_half" for i in range(n)],
        "missing_elements": [np.nan] * n, "e_hull_now_eV": [0.0 if i % 40 else 0.01 for i in range(n)],
    })
    cfg = SamplerConfig(n_total=80, cap_per_chemsys=2, seed=7, force_top_n=0)
    a = stratified_sample(df, cfg)
    b = stratified_sample(df, cfg)
    assert a.index.tolist() == b.index.tolist()                      # deterministic for a seed
    assert len(a) == 80 and a.index.is_unique
    assert (a["sample_stratum"] == a["stratum"]).all()
    assert (a["cost_class"] != "excluded").all() and (a["e_hull_now_eV"] == 0).all()
    assert a["chemsys"].value_counts().max() <= 2
    assert (a["sample_stratum"].value_counts() >= 19).all()           # ~25 % each when every stratum has room
    assert not a["forced_tail"].any()
    # forced tail: the top-10 T2_SL hosts of the pool (cap respected) are in, the total stays n_total
    t = stratified_sample(df, SamplerConfig(n_total=80, cap_per_chemsys=2, seed=7, force_top_n=10))
    pool = df[(df.cost_class != "excluded") & (df.e_hull_now_eV == 0)]
    top = pool.sort_values("bath_log10_t2_sl", ascending=False).index[:10]
    assert len(t) == 80 and t.index.is_unique and t["forced_tail"].sum() == 10
    assert set(top) <= set(t.index)
    assert t["chemsys"].value_counts().max() <= 2


def test_reduced_anisotropy_cell_independent():
    a = 3.0
    fcc = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    skew = np.array([[1, 0, 0], [3, 1, 0], [-2, 5, 1]]) @ fcc       # same lattice, unimodular change of basis
    for lat in (fcc, a * np.eye(3), skew):
        assert reduced_anisotropy(Crystal(lat, np.zeros((1, 3)), ("X",))) == pytest.approx(1.0)
    hexa = np.array([[2.0, 0, 0], [-1.0, np.sqrt(3), 0], [0, 0, 7.0]])
    sheared = np.array([[1, 0, 0], [0, 1, 0], [4, -3, 1]]) @ hexa
    assert reduced_anisotropy(Crystal(sheared, np.zeros((1, 3)), ("X",))) == pytest.approx(3.5)
    assert abs(np.linalg.det(reduced_basis(skew))) == pytest.approx(abs(np.linalg.det(fcc)))
