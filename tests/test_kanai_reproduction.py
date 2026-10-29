"""T-003: our scaling law + configs/base.yaml constants reproduce Kanai et al. (PNAS 2022) published values.

Fig. 3 of the paper gives T2 of every element at element density 1.0e23 cm^-3, natural abundance,
from Eqs. 1-2 (data/external/kanai2022/figure3.csv, seconds, 3 significant figures).
"""
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from qkhosts.physics.isotopes import load_kanai_isotope_table
from qkhosts.physics.scaling_law import ScalingLawParams, combine, t2_isotope

ROOT = Path(__file__).resolve().parents[1]
KANAI = ROOT / "data" / "external" / "kanai2022"
N_ELEMENT_CM3 = 1.0e23  # Fig. 3 assumption stated in the paper text
# data/external/ is not in public/MANIFEST, so the public repo skips these tests.
pytestmark = pytest.mark.skipif(not (KANAI / "isotopetable.csv").exists(), reason="Kanai Zenodo files not present (T-004)")


def _cfg_params() -> ScalingLawParams:
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    return ScalingLawParams.from_config(cfg["scaling_law"])


def _fig3() -> dict[str, float]:
    with (KANAI / "figure3.csv").open(encoding="utf-8", newline="") as f:
        return {r[0]: float(r[1]) for r in csv.reader(f) if r and r[0]}  # float("Inf") -> inf


def _element_t2(iso, element: str, p: ScalingLawParams) -> float:
    r = iso[iso["element"] == element]
    return combine(t2_isotope(r["g_factor"], r["spin_I"], r["abundance"] * N_ELEMENT_CM3, p), p.eta)


def _max_dlog10(p: ScalingLawParams) -> float:
    iso = load_kanai_isotope_table(KANAI / "isotopetable.csv")
    worst = 0.0
    for el, ref in _fig3().items():
        ours = _element_t2(iso, el, p)
        if np.isinf(ref) or np.isinf(ours):
            assert np.isinf(ref) and np.isinf(ours), el  # only Ce: no spinful stable isotope
            continue
        worst = max(worst, abs(np.log10(ours / ref)))
    return worst


def test_kanai_isotope_table_loads():
    iso = load_kanai_isotope_table(KANAI / "isotopetable.csv")
    assert len(iso) == 119 and iso["element"].nunique() == 80
    c13 = iso[iso["isotope"] == "13C"].iloc[0]
    assert c13["spin_I"] == 0.5 and 0 < c13["abundance"] < 0.02


def test_config_is_verified_and_reproduces_fig3():
    p = _cfg_params()
    assert p.verified
    assert _max_dlog10(p) < 0.005  # 3-sig-fig rounding of figure3.csv is <= ~0.0022 in log10


def test_rounded_eq2_exponents_do_not_reproduce_fig3():
    # Documents the T-003 finding: the rounded Eq. 2 exponents (-1.6, -1.1) miss by up to ~0.05 dex.
    p = ScalingLawParams(A=1.5e18, alpha=-1.6, beta=-1.1, gamma=-1.0, eta=2.0)
    assert _max_dlog10(p) > 0.03
