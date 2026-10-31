"""T-011 (S016): Kanai universe loader, formula parser, missing-element rule and the nuclear-spin-class census."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qkhosts.data.kanai_universe import parse_formula, split_isotope, volume_per_formula_A3
from qkhosts.features.bath import KANAI_MISSING_T2_S, t2_kanai_rule
from qkhosts.labels.census import spin_census
from qkhosts.physics.scaling_law import ScalingLawParams

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "external" / "kanai2022_zip" / "extended_data.xlsx"
ISO = ROOT / "data" / "external" / "kanai2022" / "isotopetable.csv"


def _iso() -> pd.DataFrame:
    """Toy spinful-isotope table (abundance fractions)."""
    return pd.DataFrame([
        ("C", "13C", 0.0107, 0.5, 1.4048), ("O", "17O", 0.00038, 2.5, -0.7575),
        ("Al", "27Al", 1.0, 2.5, 1.4566), ("H", "1H", 0.99988, 0.5, 5.5857), ("Si", "29Si", 0.0468, 0.5, -1.1106),
    ], columns=["element", "isotope", "abundance", "spin_I", "g_factor"])


@pytest.mark.parametrize("f, want", [
    ("CaO", {"Ca": 1, "O": 1}), ("Ce(SO4)2", {"Ce": 1, "S": 2, "O": 8}),
    ("Sr2Si(S2O7)4", {"Sr": 2, "Si": 1, "S": 8, "O": 28}), ("Mn7P6(HO6)4", {"Mn": 7, "P": 6, "H": 4, "O": 24}),
    ("K2((CN)2)3", {"K": 2, "C": 6, "N": 6}), ("Li0.5O", {"Li": 0.5, "O": 1})])
def test_parse_formula(f, want):
    assert parse_formula(f) == want


@pytest.mark.parametrize("bad", ["Ca)O", "Ca(O", "cao", "Ca-O"])
def test_parse_formula_rejects(bad):
    with pytest.raises(ValueError):
        parse_formula(bad)


def test_split_isotope_and_volume():
    assert split_isotope("17O") == (17, "O")
    iso = _iso()
    # 2 O per formula unit, 17O density 1e19 cm^-3 -> V = 2 * 0.00038 / 1e19 cm^3 = 76 Å^3
    assert volume_per_formula_A3({"Si": 1, "O": 2}, "17O", 1e19, iso) == pytest.approx(76.0)
    assert np.isnan(volume_per_formula_A3({"Si": 1}, "17O", 1e19, iso))


def test_t2_kanai_rule_missing_elements():
    iso, p = _iso(), ScalingLawParams()
    base = t2_kanai_rule({"Si": 1, "O": 2}, 40.0, iso, p)
    assert np.isfinite(base) and base > 0
    assert t2_kanai_rule({"Ce": 1, "Si": 1, "O": 2}, 40.0, iso, p) == pytest.approx(base)  # Ce adds nothing
    assert t2_kanai_rule({"Th": 1, "O": 2}, 40.0, iso, p) == pytest.approx(KANAI_MISSING_T2_S, rel=1e-9)


def test_spin_census_classes():
    iso = _iso()
    assert spin_census({"Al": 2, "O": 3}, 42.5, iso)["cost_class"] == "dense_gt_half"  # 27Al 100 %
    assert spin_census({"C": 1, "H": 4}, 40.0, iso)["cost_class"] == "dense_half"
    q = spin_census({"Si": 1, "O": 2}, 37.7, iso)  # quartz-like: 17O share ~0.016
    assert q["cost_class"] == "half" and 0.01 < q["share_I_gt_half"] < 0.02
    assert spin_census({"Si": 1, "O": 2}, 37.7, iso, gt_half_min_share=0.01)["cost_class"] == "gt_half"
    ce = spin_census({"Ce": 1, "O": 2}, 40.0, iso)
    assert ce["cost_class"] == "gt_half" and ce["floor"] and not ce["cap"]  # 17O only, ultra-dilute
    assert spin_census({"Th": 1, "O": 2}, 40.0, iso)["cost_class"] == "excluded"
    # counts per formula unit vs per cell give the same census
    a, b = spin_census({"Si": 1, "O": 2}, 37.7, iso), spin_census({"Si": 3, "O": 6}, 113.1, iso)
    assert a["n_bath_50A"] == pytest.approx(b["n_bath_50A"]) and a["share_I_gt_half"] == pytest.approx(b["share_I_gt_half"])


@pytest.mark.skipif(not (XLSX.exists() and ISO.exists()), reason="Kanai zip members not present (fetch_kanai2022_zip.py)")
def test_universe_reproduces_kanai_t2():
    """Formula + densest-isotope volume + the authors' rule reproduce Kanai's printed T2 (first 300 hosts and
    every excluded one); the benchmark hosts get the expected classes."""
    from qkhosts.data.kanai_universe import load_table_s1
    from qkhosts.physics.isotopes import load_kanai_isotope_table
    df, iso, p = load_table_s1(XLSX), load_kanai_isotope_table(ISO), ScalingLawParams()
    assert len(df) == 12859 and df["mp_id"].is_unique and (df["band_gap_eV"] > 1.0).all()
    sub = pd.concat([df.head(300), df[df["t2_kanai_s"] <= 1e-19]])
    for r in sub.itertuples(index=False):
        c = parse_formula(r.material)
        v = volume_per_formula_A3(c, r.densest_isotope, r.densest_n_cm3, iso)
        assert t2_kanai_rule(c, v, iso, p) == pytest.approx(r.t2_kanai_s, rel=1e-9), r.material
    # Table 1 ranks 1-26 are the sheet's first 26 rows; rank 27 FeS2 is not in the sheet (sheet 27 = Ca2GeS4).
    # Printed T2 (ms) equals the sheet's T2 at 2 significant figures except S (18.45 -> 19) and MgO (0.625 -> 0.6);
    # every matched row is within 0.02 dex (E1 match report, experiments/E1_notes.md).
    import csv
    with (ROOT / "data" / "external" / "kanai2022" / "table1.csv").open(encoding="utf-8-sig", newline="") as f:
        t1 = [r for r in csv.reader(f) if r and r[0].strip().isdigit()]
    names = list(df["material"])
    for (rank, name, t2_ms, *_), r in zip(t1[:26], df.head(26).itertuples(index=False)):
        assert name == r.material, rank
    misses = []
    for rank, name, t2_ms, *_ in t1:
        if name not in names:
            continue
        ms = float(df.loc[df["material"] == name, "t2_kanai_s"].iloc[0]) * 1e3
        assert abs(np.log10(float(t2_ms) / ms)) <= 0.02, name
        if float(t2_ms) != float(f"{ms:.2g}"):
            misses.append(name)
    assert misses == ["S", "MgO"]
    assert "FeS2" not in names
    by = df.set_index("mp_id")
    # diamond, 4H-SiC, alpha-quartz, Si and ZnO are not in the universe (one hull entry per composition, GGA gap > 1 eV)
    assert not {"mp-66", "mp-11714", "mp-6930", "mp-149", "mp-2133"} & set(df["mp_id"])
    want = {"mp-20194": "gt_half", "mp-2605": "gt_half", "mp-1265": "gt_half", "mp-546794": "half",
            "mp-1143": "dense_gt_half"}
    for mp, cls in want.items():
        r = by.loc[mp]
        c = parse_formula(r["material"])
        v = volume_per_formula_A3(c, r["densest_isotope"], r["densest_n_cm3"], iso)
        assert spin_census(c, v, iso)["cost_class"] == cls, mp


def test_optimade_id_filter():
    from qkhosts.data.optimade import id_filter
    assert id_filter(["mp-66", "mvc-13324"]) == 'id="mp-66" OR id="mvc-13324"'
    for bad in ("mp-6a", 'mp-1" OR 1', "xx-1"):
        with pytest.raises(ValueError):
            id_filter([bad])
