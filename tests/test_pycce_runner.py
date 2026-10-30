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
        return FakeBath(seed, remove[0][0] if remove else "")


def make_fake(t2_base, r_scale=10.0, amp=2.0):
    """t2_base: element -> T2 at r_bath -> inf (ms). T2(r) = t2_base * (1 + amp exp(-r / r_scale))."""
    class Sim:
        def __init__(self, **kw):
            self.kw = kw
            CALLS.append(kw)

        def compute(self, ts, method, quantity):
            self.kw["method"] = method
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


def test_nls_fit_recovers_stretched_exponential():
    t = np.linspace(0, 3, 201)
    t2, n = pr.fit_t2_nls(t, np.exp(-(t / 0.97) ** 2.64))
    assert t2 == pytest.approx(0.97, rel=1e-4) and n == pytest.approx(2.64, rel=1e-4)
    assert np.isnan(pr.fit_t2_nls(t, np.ones_like(t))[0])


def test_diagnostics_recorded(fake, tmp_path):
    """S011: per-config fits, NLS fit, mean curve, spins inside r_bath and the method are recorded."""
    fake({"Si": 1.0}, SI)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=3, method="gcce"), tmp_path / "out")
    st = res["sites"][0]["final"]
    assert len(st["per_conf_t2_ms"]) == len(st["per_conf_n"]) == 3
    assert all(n == pytest.approx(2.0, abs=1e-6) for n in st["per_conf_n"])
    assert st["t2_ms_nls"] == pytest.approx(st["t2_ms"], rel=0.01)
    assert len(st["curve_t_ms"]) == len(st["curve_mean"]) == CCEProtocol().n_t
    assert st["n_bath_in_rbath"] == 2  # both fake spins lie within 65 Å of the centre
    assert all(kw["method"] == "gcce" for kw in CALLS)


def test_origin_site_kanai_rule():
    """Kanai's centre sits at the CIF origin (load_crystal.m). Fractional sites copied from our E0 CIFs."""
    a = 3.567
    diamond = np.array([[.625, .625, .125], [.625, .125, .625], [.125, .625, .625], [.125, .125, .125],
                        [.875, .875, .875], [.875, .375, .375], [.375, .875, .375], [.375, .375, .875]])
    el, frac = pr.origin_site(np.eye(3) * a, diamond, ["C"] * 8)
    assert el is None and np.allclose(frac, 0)  # bond centre, 0.77 Å from two C atoms
    hexL = np.array([[3.0757, 0, 0], [-1.53785, 2.66364, 0], [0, 0, 10.0706]])
    sic = np.array([[2/3, 1/3, .93784], [1/3, 2/3, .43784], [0, 0, .68789], [0, 0, .18789],
                    [2/3, 1/3, .74980], [1/3, 2/3, .24980], [0, 0, .50046], [0, 0, .00046]])
    el, frac = pr.origin_site(hexL, sic, ["Si"] * 4 + ["C"] * 4)
    assert el == "C" and np.allclose(frac, [0, 0, .00046])  # C vacancy at the origin
    # an atom just across the cell boundary (frac 0.9999) still counts as at the origin
    assert pr.origin_site(np.eye(3) * 4, np.array([[0.9999, 0, 0]]), ["X"])[0] == "X"


def test_origin_rule_without_removal(fake, monkeypatch, tmp_path):
    fake({"": 1.0}, SI)
    monkeypatch.setattr(pr, "_load_origin", lambda path: (None, [(None, np.zeros(3), 1)]))
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2, site_rule="origin_kanai"), tmp_path / "out")
    assert res["n_sites"] == 1 and res["sites"][0]["element"] == "" and np.isfinite(res["t2_ms"])
    assert all(np.allclose(kw["position"], 0) for kw in CALLS)


def test_isotopes_fall_back_to_common_concentrations():
    """cell.isotopes is {} unless isotopes were added (006-e0-diag); the record must then come from
    PyCCE's natural-abundance table, restricted to the host's elements."""
    cell = types.SimpleNamespace(isotopes={})
    common = {"C": {"13C": 0.0107}, "O": {"17O": 0.00038}, "Si": {"29Si": 0.04685}}
    pc = types.SimpleNamespace(common_concentrations=common)
    assert pr._isotopes(cell, pc, ["Si", "C", "C"]) == {"C": {"13C": 0.0107}, "Si": {"29Si": 0.04685}}
    pc_nested = types.SimpleNamespace(bath=types.SimpleNamespace(common_concentrations=common))
    assert pr._isotopes(cell, pc_nested, ["O"]) == {"O": {"17O": 0.00038}}
    assert pr._isotopes(cell) is None
    added = types.SimpleNamespace(isotopes={"C": {"13C": 0.5}})
    assert pr._isotopes(added, pc, ["C"]) == {"C": {"13C": 0.5}}


# --- S013 (T-065, ADR-0004 option A): field-direction list, single-site rule, bath-size cap ---

def make_dir_fake(t2_by_axis):
    """Fake whose T2 depends on the field direction: T2 = sum_i t2_by_axis[i] * b_i^2 (b a unit vector)."""
    class Sim:
        def __init__(self, **kw):
            self.kw = kw
            CALLS.append(kw)

        def compute(self, ts, method, quantity):
            b = np.asarray(self.kw["magnetic_field"], float)
            b = b / np.linalg.norm(b)
            t2 = float(np.dot(t2_by_axis, b ** 2))
            return np.exp(-(np.asarray(ts) / t2) ** 2)

    return types.SimpleNamespace(read_ase=lambda atoms: FakeCell(), Simulator=Sim)


DIRS = ((0, 0, 1), (1, 1, 0), (1, 1, 1))


def test_direction_list_log_mean(monkeypatch, tmp_path):
    CALLS.clear()
    monkeypatch.setitem(sys.modules, "pycce", make_dir_fake([1.0, 2.0, 4.0]))
    monkeypatch.setattr(pr, "_load_sites", lambda path, n: (None, SI))
    proto = CCEProtocol(n_conf=2, ladder=((50.0, 15.0),), b_directions_lattice=DIRS)
    res = run_material(tmp_path / "x.cif", proto, tmp_path / "out")
    # FakeCell is cubic: c -> z (T2 4), a+b -> (x+y)/sqrt2 (T2 1.5), a+b+c (T2 7/3)
    expect = [4.0, 1.5, 7 / 3]
    assert res["n_directions"] == 3
    assert res["direction_t2_ms"] == pytest.approx(expect, rel=1e-3)
    assert res["t2_ms"] == pytest.approx(np.prod(expect) ** (1 / 3), rel=1e-3)
    assert res["direction_spread_log10"] == pytest.approx(np.log10(4.0 / 1.5), rel=1e-3)
    site = res["sites"][0]
    assert [d["b_lattice"] for d in site["directions"]] == [list(map(float, d)) for d in DIRS]
    assert np.allclose(site["directions"][1]["b_cart_unit"], np.array([1, 1, 0]) / np.sqrt(2))
    assert site["final"]["t2_ms"] == pytest.approx(res["t2_ms"])
    # every direction reaches the simulator at 5 T, with the same isotope seeds
    fields = {tuple(np.round(kw["magnetic_field"], 6)) for kw in CALLS}
    assert len(fields) == 3 and all(np.linalg.norm(f) == pytest.approx(5e4) for f in fields)
    seeds = {}
    for kw in CALLS:
        seeds.setdefault(tuple(np.round(kw["magnetic_field"], 6)), set()).add(kw["bath"].seed)
    assert all(s == {1, 2} for s in seeds.values())


def test_single_direction_keeps_legacy_layout(fake, tmp_path):
    fake({"Si": 1.0}, SI)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1), tmp_path / "out")
    site = res["sites"][0]
    assert "directions" not in site and "ladder" in site and site["b_lattice"] is None
    assert "n_directions" not in res and "spinful_fraction" not in res


def test_field_directions_use_lattice_columns():
    hexcell = types.SimpleNamespace(cell=np.array([[3.0, -1.5, 0.0], [0.0, 1.5 * np.sqrt(3), 0.0], [0.0, 0.0, 10.0]]))
    d = pr.field_directions(hexcell, CCEProtocol(b_directions_lattice=DIRS))
    assert np.allclose(d[0][1], [0, 0, 1])
    assert np.allclose(d[1][1], [0.5, np.sqrt(3) / 2, 0])  # a + b of a hexagonal cell, 60° from a
    assert d[2][1][2] > 0.9  # a + b + c leans towards c for c/a = 3.33
    assert pr.field_directions(hexcell, CCEProtocol())[0][0] is None


def test_pick_site_rule():
    # Al2O3-like: Al (mult 12, 100 % 27Al) beats O (mult 18, 0.038 % 17O)
    sites = [("O", np.zeros(3), 18), ("Al", np.ones(3) * 0.35, 12)]
    assert pr.pick_site(sites, {"Al": 1.0, "O": 0.00038})[0] == "Al"
    # SiO2: Si (3 x 4.7 %) beats O (6 x 0.038 %)
    assert pr.pick_site([("O", np.zeros(3), 6), ("Si", np.zeros(3), 3)], {"Si": 0.0468, "O": 0.00038})[0] == "Si"
    # within the element, the highest multiplicity; ties go to the first in structure order
    a, b, c = ("C", np.zeros(3), 2), ("C", np.ones(3) * .5, 4), ("C", np.ones(3) * .25, 4)
    assert pr.pick_site([a, b, c], {"C": 0.0107}) is b
    # element tie: first element in structure order
    assert pr.pick_site([("Si", np.zeros(3), 4), ("C", np.zeros(3), 4)], {"Si": 0.01, "C": 0.01})[0] == "Si"


def test_single_most_spinful_in_runner(monkeypatch, tmp_path):
    CALLS.clear()
    fk = make_fake({"Si": 1.0, "C": 4.0})
    fk.common_concentrations = {"C": {"13C": 0.0107}, "Si": {"29Si": 0.0468}}
    fk.common_isotopes = {"13C": types.SimpleNamespace(s=0.5), "29Si": types.SimpleNamespace(s=0.5)}
    monkeypatch.setitem(sys.modules, "pycce", fk)
    sites = [("C", np.zeros(3), 6), ("Si", np.ones(3) * .5, 2), ("Si", np.ones(3) * .25, 4)]
    seen = {}

    def load(path, n):
        seen["n"] = n
        return None, sites[:n]

    monkeypatch.setattr(pr, "_load_sites", load)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1, site_rule="single_most_spinful"), tmp_path / "out")
    assert seen["n"] >= len(sites)  # the rule sees every inequivalent site, not the first n_sites_max
    assert res["n_sites"] == 1 and res["sites"][0]["element"] == "Si" and res["sites"][0]["multiplicity"] == 4
    assert res["spinful_fraction"] == pytest.approx({"C": 0.0107, "Si": 0.0468})


def test_spinful_fractions_drop_spin_zero():
    pc = types.SimpleNamespace(common_concentrations={"X": {"1X": 0.3, "2X": 0.7}},
                               common_isotopes={"1X": types.SimpleNamespace(s=0.0), "2X": types.SimpleNamespace(s=1.5)})
    assert pr._spinful_fractions(pc, ["X", "Y"]) == {"X": pytest.approx(0.7), "Y": 0.0}


def test_cap_radii():
    d = np.arange(1, 101, dtype=float)  # spins at 1, 2, ..., 100 Å
    assert pr.cap_radii(d, 50.0, 15.0, None) == (50.0, 15.0, False)
    assert pr.cap_radii(d, 50.0, 15.0, 60) == (50.0, 15.0, False)  # 50 spins inside, under the cap
    rb, rd, capped = pr.cap_radii(d, 50.0, 15.0, 20)
    assert capped and rb == pytest.approx(20.5) and rd == pytest.approx(15.0 * 20.5 / 50.0)
    assert (d <= rb).sum() == 20


def test_bath_cap_in_runner(monkeypatch, tmp_path):
    """A dense fake bath (spins every 1 Å along x) is capped: r_bath and r_dipole shrink together."""
    CALLS.clear()
    fk = make_fake({"Al": 1.0})

    class DenseBath(FakeBath):
        def __init__(self, seed, removed):
            super().__init__(seed, removed)
            self.xyz = np.c_[np.arange(1, 3001, dtype=float), np.zeros(3000), np.zeros(3000)]

    class DenseCell(FakeCell):
        def gen_supercell(self, size, remove=None, seed=None):
            return DenseBath(seed, remove[0][0])

    fk.read_ase = lambda atoms: DenseCell()
    monkeypatch.setitem(sys.modules, "pycce", fk)
    monkeypatch.setattr(pr, "_load_sites", lambda path, n: (None, [("Al", np.zeros(3), 1)]))
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2, ladder=((50.0, 15.0),), max_bath_spins=10),
                       tmp_path / "out")
    st = res["sites"][0]["final"]
    assert st["capped"] and st["n_bath_in_rbath_nominal"] == 50 and st["n_bath_in_rbath"] == 10
    assert st["r_bath_used_A"] == pytest.approx(10.5) and st["r_dipole_used_A"] == pytest.approx(15 * 10.5 / 50)
    assert st["r_bath_set_A"] == 50.0 and st["r_bath_A"] == 50.0
    assert all(kw["r_bath"] == pytest.approx(10.5) for kw in CALLS)


def test_single_site_rule_refuses_without_abundances(fake, monkeypatch, tmp_path):
    fake({"Si": 1.0}, SI)  # the fake PyCCE has no common_concentrations
    with pytest.raises(RuntimeError, match="spinful"):
        run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1, site_rule="single_most_spinful"), tmp_path / "out")


# --- S014 (T-066): bath-size floor for ultra-dilute baths, fit guard, strict host label ---

def test_guard_fit_reasons():
    ts = np.linspace(0.0, 10.0, 101)
    assert pr.guard_fit(2.0, 2.5, ts, 40, 15) == (2.0, 2.5, "")
    assert pr.guard_fit(2.0, 2.5, ts, 10, 15)[2] == "few_points"
    assert pr.guard_fit(float("nan"), 2.5, ts, 40, 15)[2] == "fit_failed"
    assert pr.guard_fit(2.0, 0.3, ts, 40, 15)[2] == "n_out_of_range"
    assert pr.guard_fit(2.0, -0.01, ts, 40, 15)[2] == "n_out_of_range"   # CeO2 007 early shard
    assert pr.guard_fit(6.9e149, 0.9, ts, 40, 15)[2] == "t2_outside_window"
    assert pr.guard_fit(1e-3, 0.9, ts, 40, 15)[2] == "t2_outside_window"
    t2, n, flag = pr.guard_fit(2.0, 7.0, ts, 40, 15)
    assert np.isnan(t2) and np.isnan(n) and flag == "n_out_of_range"


def test_floor_radii():
    d = np.arange(1, 101, dtype=float)  # spins at 1, 2, ..., 100 Å
    assert pr.floor_radii(d, 50.0, 15.0, None, 90.0) == (50.0, 15.0, False, None)
    assert pr.floor_radii(d, 50.0, 15.0, 40, 90.0) == (50.0, 15.0, False, None)  # 50 inside, above the floor
    rb, rd, floored, need = pr.floor_radii(d, 50.0, 15.0, 70, 90.0)
    assert floored and need is None and rb == pytest.approx(70.5) and rd == pytest.approx(15.0 * 70.5 / 50.0)
    assert (d <= rb).sum() == 70
    rb, rd, floored, need = pr.floor_radii(d, 50.0, 15.0, 70, 60.0)  # only 60 spins inside the safe radius
    assert not floored and (rb, rd) == (50.0, 15.0) and need == pytest.approx(60.0 * (1.2 * 70 / 60) ** (1 / 3))


class GridBath(FakeBath):
    """Ultra-dilute fake bath: one spin per (10 Å)^3, uniform in the cube of side `size` Å around 0 (random
    positions, so no two spins share a distance from the centre)."""
    def __init__(self, seed, removed, size):
        super().__init__(seed, removed)
        rng = np.random.default_rng(seed)
        self.xyz = rng.uniform(-size / 2, size / 2, (int(size ** 3 / 1000), 3))


def _install_grid(monkeypatch):
    CALLS.clear()
    fk = make_fake({"Si": 1.0})
    sizes = []

    class GridCell(FakeCell):
        def gen_supercell(self, size, remove=None, seed=None):
            sizes.append(size)
            return GridBath(seed, remove[0][0], size)

    fk.read_ase = lambda atoms: GridCell()
    monkeypatch.setitem(sys.modules, "pycce", fk)
    monkeypatch.setattr(pr, "_load_sites", lambda path, n: (None, SI))
    return sizes


def test_bath_floor_in_runner(monkeypatch, tmp_path):
    """Too few spins inside r_bath: r_bath grows to enclose the floor and r_dipole grows with it."""
    _install_grid(monkeypatch)
    proto = CCEProtocol(n_conf=2, ladder=((20.0, 6.0),), min_bath_spins=100)
    st = run_material(tmp_path / "x.cif", proto, tmp_path / "out")["sites"][0]["final"]
    assert st["floored"] and not st["capped"] and st["supercell_used"] == 200
    assert st["n_bath_in_rbath_nominal"] < 100 and st["n_bath_in_rbath"] == 100
    assert st["r_bath_used_A"] > 20.0 and st["r_dipole_used_A"] == pytest.approx(6.0 * st["r_bath_used_A"] / 20.0)
    assert all(kw["r_bath"] == pytest.approx(st["r_bath_used_A"]) for kw in CALLS)


def test_bath_floor_grows_supercell(monkeypatch, tmp_path):
    """The floor does not fit inside the requested supercell: it is regenerated larger for every configuration."""
    sizes = _install_grid(monkeypatch)
    proto = CCEProtocol(n_conf=3, ladder=((10.0, 3.0),), min_bath_spins=100, supercell=40)
    st = run_material(tmp_path / "x.cif", proto, tmp_path / "out")["sites"][0]["final"]
    assert st["floored"] and st["supercell_used"] > 40 and st["n_bath_in_rbath"] == 100
    assert st["r_bath_used_A"] <= pr.SUPERCELL_SAFE * st["supercell_used"]
    assert sizes[0] == 40 and set(sizes[1:]) == {st["supercell_used"]}


def test_no_floor_by_default(monkeypatch, tmp_path):
    _install_grid(monkeypatch)
    st = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=1, ladder=((20.0, 6.0),)),
                      tmp_path / "out")["sites"][0]["final"]
    assert not st["floored"] and st["r_bath_used_A"] == 20.0 and st["supercell_used"] == 200


def test_fit_guard_voids_label(monkeypatch, tmp_path):
    """A decay that is no stretched exponential in range (n = 0.2) gives NaN fits and a NaN, flagged label."""
    CALLS.clear()

    class Sim:
        def __init__(self, **kw):
            self.kw = kw

        def compute(self, ts, method, quantity):
            return np.exp(-(np.asarray(ts) / 1.0) ** 0.2)

    monkeypatch.setitem(sys.modules, "pycce", types.SimpleNamespace(read_ase=lambda a: FakeCell(), Simulator=Sim))
    monkeypatch.setattr(pr, "_load_sites", lambda path, n: (None, SI))
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2, ladder=((50.0, 15.0),)), tmp_path / "out")
    st = res["sites"][0]["final"]
    assert st["fit_flag"] == "n_out_of_range" and np.isnan(st["t2_ms"]) and np.isnan(st["stretch_n"])
    assert all(np.isnan(st["per_conf_t2_ms"])) and st["n_fit_failed"] == 2
    assert np.isnan(res["t2_ms"]) and not res["label_ok"] and not res["converged"]
    assert json.loads((tmp_path / "out" / "x.json").read_text(encoding="utf-8"))["label_ok"] is False


def test_one_bad_direction_voids_label(monkeypatch, tmp_path):
    """Before T-066 the log-mean silently skipped bad directions; now one NaN direction voids the label."""
    CALLS.clear()

    class Sim:
        def __init__(self, **kw):
            self.kw = kw

        def compute(self, ts, method, quantity):
            b = np.asarray(self.kw["magnetic_field"], float)
            n = 0.2 if abs(b[2]) > 0.99 * np.linalg.norm(b) else 2.0  # only B || c decays badly
            return np.exp(-(np.asarray(ts) / 1.0) ** n)

    monkeypatch.setitem(sys.modules, "pycce", types.SimpleNamespace(read_ase=lambda a: FakeCell(), Simulator=Sim))
    monkeypatch.setattr(pr, "_load_sites", lambda path, n: (None, SI))
    proto = CCEProtocol(n_conf=1, ladder=((50.0, 15.0),), b_directions_lattice=DIRS)
    res = run_material(tmp_path / "x.cif", proto, tmp_path / "out")
    d = res["direction_t2_ms"]
    assert np.isnan(d[0]) and d[1] == pytest.approx(1.0, rel=1e-2) and d[2] == pytest.approx(1.0, rel=1e-2)
    assert np.isnan(res["sites"][0]["final"]["t2_ms"]) and np.isnan(res["t2_ms"]) and not res["label_ok"]


def test_good_label_is_ok(fake, tmp_path):
    fake({"Si": 1.0}, SI)
    res = run_material(tmp_path / "x.cif", CCEProtocol(n_conf=2), tmp_path / "out")
    assert res["label_ok"] and res["sites"][0]["final"]["fit_flag"] == ""
