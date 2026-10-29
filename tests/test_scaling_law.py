import numpy as np
import pandas as pd
import pytest

from qkhosts.features.bath import bath_descriptors, isotope_densities
from qkhosts.physics.isotopes import normalise
from qkhosts.physics.scaling_law import ScalingLawParams, combine, dominant_share, t2_isotope

P = ScalingLawParams()


def test_powerlaw_exponents():
    base = t2_isotope(1.0, 0.5, 1e20, P)
    assert np.isclose(t2_isotope(1.0, 0.5, 2e20, P) / base, 2 ** P.gamma)
    assert np.isclose(t2_isotope(2.0, 0.5, 1e20, P) / base, 2 ** P.alpha)
    assert np.isclose(t2_isotope(1.0, 1.0, 1e20, P) / base, 2 ** P.beta)


def test_spinless_is_inf_and_combine():
    t = t2_isotope(np.array([1.0, 1.0]), np.array([0.0, 0.5]), np.array([1e20, 1e20]), P)
    assert np.isinf(t[0]) and np.isfinite(t[1])
    assert combine(t, 2.0) == pytest.approx(t[1])
    assert combine(np.array([np.inf]), 2.0) == np.inf
    # two equal contributions -> T2 / sqrt(2) for eta=2
    assert combine(np.array([1.0, 1.0]), 2.0) == pytest.approx(2 ** -0.5)
    assert dominant_share(np.array([1.0, 1.0]), 2.0) == pytest.approx(0.5)


def _toy_table():
    # SYNTHETIC values for unit tests only — not physical data.
    return normalise(pd.DataFrame({
        "element": ["A", "A", "B"], "isotope": ["A1", "A2", "B1"],
        "abundance": [0.9, 0.1, 1.0], "spin_I": [0.0, 0.5, 1.5], "g_factor": [0.0, 1.0, 0.5]}))


def test_densities_and_descriptors():
    iso = _toy_table()
    r = isotope_densities({"A": 2, "B": 1}, 100.0, iso)
    # A2: 2 * 0.1 / (100e-24) = 2e21 cm^-3
    assert r.loc[r.isotope == "A2", "n_cm3"].item() == pytest.approx(2e21)
    d = bath_descriptors({"A": 2, "B": 1}, 100.0, iso, P)
    assert d["n_spinful_species"] == 2 and np.isfinite(d["log10_t2_sl"])
    assert d["min_rel_gamma_gap"] == pytest.approx(0.5)


def test_isotope_table_validation():
    with pytest.raises(ValueError):
        normalise(pd.DataFrame({"element": ["A"], "isotope": ["A1"], "abundance": [0.5], "spin_I": [0.5], "g_factor": [1.0]}))


def test_from_config_casts_yaml_strings():
    # PyYAML loads "1.5e18" (no sign in the exponent) as a str; from_config must cast.
    import yaml
    p = ScalingLawParams.from_config(yaml.safe_load("A: 1.5e18\nalpha: -1.65\nverified: true"))
    assert isinstance(p.A, float) and p.A == 1.5e18 and p.alpha == -1.65 and p.verified
