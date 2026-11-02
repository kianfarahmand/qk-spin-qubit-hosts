"""CCE cost model for the E2 budget (S019, T-070): bath mix, pair weight, sizes, fit recovery."""
import numpy as np
import pandas as pd
import pytest

from qkhosts.labels.cost import (bath_mix, bath_used, eval_seconds, extra_windows, fit_cost_model,
                                 isotopes_from_table, n_pairs, pair_weight, supercell_spins)

SPIN = {"31P": 0.5, "69Ga": 1.5, "71Ga": 1.5, "25Mg": 2.5, "33S": 1.5}
GAP = {"Ga": {"69Ga": 0.6, "71Ga": 0.4}, "P": {"31P": 1.0}}


def test_bath_mix_shares_and_dims():
    f, d = bath_mix({"Ga": 1, "P": 1}, GAP, SPIN)
    assert f.sum() == pytest.approx(1.0)
    assert len(f) == 3
    # Ga isotopes both d = 4: share 0.5 together; 31P d = 2: 0.5
    assert f[d == 4.0].sum() == pytest.approx(0.5) and f[d == 2.0].sum() == pytest.approx(0.5)
    f0, d0 = bath_mix({"Ce": 1, "O": 2}, {}, SPIN)
    assert len(f0) == 0 and len(d0) == 0


def test_pair_weight():
    f, d = np.array([1.0]), np.array([2.0])  # pure I = 1/2
    assert pair_weight(f, d, 2.0) == pytest.approx(16.0)  # (2^2)^2
    assert pair_weight(np.array([0.5, 0.5]), np.array([2.0, 4.0]), 1.0) == pytest.approx(9.0)  # (1 + 2)^2
    assert pair_weight(f, d, 0.0) == pytest.approx(1.0)


def test_sizes():
    assert n_pairs(1500) == pytest.approx(0.5 * 1500 ** 2 * 0.027)
    assert bath_used(25770, 1000, 1500) == 1500 and bath_used(92, 1000, 1500) == 1000
    assert bath_used(1068, 1000, 1500) == 1068 and bath_used(50, None, None) == 50
    lo = 16.0 * 1.2 * 1000 * (50 / 80) ** 3  # floor radius beyond 0.4 x 200 Å: supercell grows
    assert supercell_spins(92, 1000, 16.0) == pytest.approx(lo)
    assert supercell_spins(25770, 1000, 16.0) == pytest.approx(16.0 * 25770)
    assert extra_windows(0.01) == 2 and extra_windows(1.0) == 1 and extra_windows(40.0) == 1


def test_eval_seconds_and_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    true = {"a": 2e-6, "b": 3e-5, "p": 2.5}
    fd = [(np.array([1.0]), np.array([float(x)])) for x in rng.choice([2, 3, 4, 6, 8, 10], 30)]
    pairs = rng.uniform(5e3, 6e4, 30)
    n_sc = rng.uniform(5e3, 9e5, 30)
    t = np.array([eval_seconds(true, pp, pair_weight(f, d, true["p"]), n) for pp, (f, d), n in zip(pairs, fd, n_sc)])
    fit = fit_cost_model(pairs, fd, n_sc, t)
    assert fit["p"] == pytest.approx(2.5, abs=1e-3) and fit["a"] == pytest.approx(2e-6, rel=1e-2)
    assert fit["rms_dex"] < 1e-4


def test_isotopes_from_table():
    iso = pd.DataFrame([("Ga", "69Ga", 0.6, 1.5), ("Ga", "71Ga", 0.4, 1.5), ("O", "16O", 0.99, 0.0),
                        ("P", "31P", 1.0, 0.5)], columns=["element", "isotope", "abundance", "spin_I"])
    assert isotopes_from_table(iso, ["Ga", "O"]) == {"Ga": {"69Ga": 0.6, "71Ga": 0.4}}
