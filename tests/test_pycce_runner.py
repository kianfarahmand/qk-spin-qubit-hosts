"""run_material (T-010 prep) against a fake PyCCE. The fake returns exp(-(t/T2)^2) with a T2 that
depends on the removed element, r_bath and the isotope seed, so window adaptation, ensembles,
convergence and site averaging can be checked without the real package."""
import json
import sys
import types

import numpy as np
import pytest

from qkhosts.labels import pycce_runner as pr
from qkhosts.labels.pycce_runner import CCEProtocol, run_material

CALLS: list[dict] = []


class FakeBath:
    def __init__(self, seed, removed):
        self.seed, self.removed = seed, removed
        self.xyz = np.array([[1.0, 0, 0], [0, 2.0, 0]])

    def __len__(self):
        return 1000


class FakeCell:
    cell = np.eye(3) * 4.0  # columns = lattice vectors (PyCCE convention)

    def gen_supercell(self, size, remove=None, seed=None):
        return FakeBath(seed, remove[0][0])


def make_fake(t2_base, r_scale=10.0, amp=2.0):
    """t2_base: element -> T2 at r_bath -> inf (ms). T2(r) = t2_base * (1 + amp exp(-r / r_scale))."""
    class Sim:
        def __init__(self, **kw):
            self.kw = kw
            CALLS.append(kw)

        def compute(self, ts, method, quantity):
            b = self.kw["bath"]
            t2 = t2_base[b.removed] * (1 + amp * np.exp(-self.kw["r_bath"] / r_scale))
            t2 *= 1 + 0.01 * ((b.seed % 3) - 1)
            return np.exp(-(np.asarray(ts) / t2) ** 2)

    return types.SimpleNamespace(read_ase=lambda atoms: FakeCell(), Simulator=Sim)


@pytest.fixture
def fake(monkeypatch):
    CALLS.clear()

    def install(t2_base, sites, **kw):
        monkeypatch.setitem(sys.modules, "pycce", make_fake(t2_base, **kw))
        monkeypatch.setattr(pr, "_load_sites", lambda path, n: (None, sites[:n]))
    return install


SI = [("Si", np.zeros(3), 1)]


def test_single_site_converges_and_fits(fake, tmp_path):
    fake({"Si": 1.0}, SI)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=3), tmp_path / "out")
    site = res["sites"][0]
    assert res["converged"] and site["converged"]
    assert len(site["ladder"]) == 2  # 50 -> 65 Å changes log10 T2 by < 0.02
    # seeds 1..3 give factors 1.00, 1.01, 0.99 around T2(65 Å); the mean curve fit lands close to it
    expect = 1.0 * (1 + 2 * np.exp(-6.5))
    assert res["t2_ms"] == pytest.approx(expect, rel=0.01)
    assert site["final"]["stretch_n"] == pytest.approx(2.0, abs=0.05)
    assert site["final"]["t2_sem_ms"] > 0 and site["final"]["n_fit_failed"] == 0
    assert site["final"]["min_bath_dist_A"] == pytest.approx(1.0)


def test_protocol_settings_reach_simulator(fake, tmp_path):
    fake({"Si": 1.0}, SI)
    run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2), tmp_path / "out")
    kw = CALLS[0]
    assert np.allclose(kw["magnetic_field"], [0, 0, 5e4])  # 5 T in gauss
    assert kw["as_delay"] is False and kw["pulses"] == 1 and kw["order"] == 2
    assert (kw["r_bath"], kw["r_dipole"]) == (50.0, 15.0)
    assert kw["spin"] == 0.5


def test_window_shrinks_for_fast_decay(fake, tmp_path):
    fake({"Si": 1.0}, SI)  # 50 ms initial window, T2 ~ 1 ms -> too few fit points
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1), tmp_path / "out")
    st = res["sites"][0]["ladder"][0]
    assert st["windows_tried_ms"][0] == 50.0 and len(st["windows_tried_ms"]) >= 2
    assert st["t_max_ms"] < 10


def test_window_widens_for_slow_decay(fake, tmp_path):
    fake({"Si": 100.0}, SI)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1), tmp_path / "out")
    assert res["sites"][0]["ladder"][0]["t_max_ms"] > 50.0
    assert np.isfinite(res["t2_ms"])


def test_unconverged_is_flagged(fake, tmp_path):
    fake({"Si": 1.0}, SI, r_scale=200.0, amp=5.0)  # strong r_bath dependence across the ladder
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1), tmp_path / "out")
    assert not res["converged"]
    assert len(res["sites"][0]["ladder"]) == len(CCEProtocol().ladder)


def test_site_average_weighted_log_mean(fake, tmp_path):
    sites = [("Si", np.zeros(3), 3), ("C", np.array([0.25, 0.25, 0.25]), 1)]
    fake({"Si": 1.0, "C": 4.0}, sites)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1), tmp_path / "out")
    t = [s["final"]["t2_ms"] for s in res["sites"]]
    assert res["t2_ms"] == pytest.approx(10 ** ((3 * np.log10(t[0]) + np.log10(t[1])) / 4))
    assert res["site_spread_log10"] == pytest.approx(abs(np.log10(t[1] / t[0])))
    # the C-site centre sits at frac (1/4,1/4,1/4) of a 4 Å cubic cell -> (1,1,1) Å
    assert any(np.allclose(kw["position"], [1, 1, 1]) for kw in CALLS)


def test_n_sites_max(fake, tmp_path):
    sites = [("Si", np.zeros(3), 3), ("C", np.zeros(3), 1)]
    fake({"Si": 1.0, "C": 4.0}, sites)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1, n_sites_max=1), tmp_path / "out")
    assert res["n_sites"] == 1


def test_resumable_and_json(fake, tmp_path):
    fake({"Si": 1.0}, SI)
    out = tmp_path / "out"
    a = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2), out)
    n = len(CALLS)
    b = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2), out)
    assert len(CALLS) == n  # no recomputation
    assert b["t2_ms"] == pytest.approx(a["t2_ms"]) and b["protocol_hash"] == a["protocol_hash"]
    assert json.loads((out / "x.json").read_text(encoding="utf-8"))["n_sites"] == 1


def test_ladder_reuses_window_and_seeds(fake, tmp_path):
    fake({"Si": 1.0}, SI)
    run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2), tmp_path / "out")
    seeds = {kw["r_bath"]: [] for kw in CALLS}
    for kw in CALLS:
        seeds[kw["r_bath"]].append(kw["bath"].seed)
    assert set(seeds[65.0]) == set(seeds[50.0]) == {1, 2}
