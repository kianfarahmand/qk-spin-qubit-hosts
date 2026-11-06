"""S024: guard against PyCCE 1.1.1's isotope-sampling overflow, which voided 7 K hosts in 013-e2-b01
("Cannot take a larger sample than population when replace is False" in BathCell.gen_supercell).
`pycce_sample` copies PyCCE 1.1.1's per-element draw (pycce/bath/cell.py, gen_supercell) so the
failure and the fix are tested without the package."""
import numpy as np
import pytest

from qkhosts.labels import pycce_runner as pr

K = {"39K": 0.932581, "40K": 0.000117, "41K": 0.067302}  # PyCCE common_concentrations (b01 records)
O = {"17O": 0.00038}


def pycce_sample(natoms: int, iso: dict, seed: int = 1) -> dict:
    """PyCCE 1.1.1's draw for one element: int(round(N c)) sites per isotope, without replacement."""
    rgen = np.random.default_rng(seed)
    sites = np.arange(natoms, dtype=np.int32)
    mask = np.zeros(natoms, dtype=bool)
    out = {}
    for i, c in iso.items():
        n = int(round(np.int32(natoms) * c))
        pick = rgen.choice(sites[~mask], n, replace=False, shuffle=False)
        mask += np.isin(sites, pick)
        out[i] = n
    return out


def overflowing_n(iso: dict, lo: int = 1000) -> int:
    n = lo
    while sum(int(round(np.int32(n) * c)) for c in iso.values()) <= n:
        n += 1
    return n


def test_overflow_reproduces_pycce_error():
    n = overflowing_n(K)
    with pytest.raises(ValueError, match="larger sample than population"):
        pycce_sample(n, K)


def test_guard_fixes_overflow_and_changes_abundance_by_excess_over_n():
    n = overflowing_n(K)
    safe, excess = pr.sampling_safe_isotopes({"K": n, "O": 4 * n}, {"K": K, "O": O})
    assert excess == {"K": 1}
    assert list(safe) == ["K", "O"] and list(safe["K"]) == list(K)  # order fixes the random stream
    got = pycce_sample(n, safe["K"])
    assert sum(got.values()) == n  # every K site carries exactly one isotope
    assert abs(safe["K"]["39K"] - K["39K"]) <= 1.0 / n
    assert safe["K"]["40K"] == K["40K"] and safe["K"]["41K"] == K["41K"] and safe["O"] == O


def test_guard_is_inactive_when_nothing_overflows():
    n = overflowing_n(K)
    ok = next(m for m in range(n + 1, n + 100)
              if sum(int(round(np.int32(m) * c)) for c in K.values()) <= m)
    assert pr.sampling_safe_isotopes({"K": ok}, {"K": K}) == (None, {})


def test_guard_share_matches_b01():
    """About 1 in 8 supercell sizes overflow for K (b01: 7 of 57 K hosts)."""
    n = np.arange(10_000, 200_000)
    s = sum(np.round(n * c).astype(int) for c in K.values())
    assert 0.10 < (s > n).mean() < 0.15
    for m in n[s > n][:200]:
        safe, _ = pr.sampling_safe_isotopes({"K": int(m)}, {"K": K})
        assert sum(int(round(np.int32(m) * c)) for c in safe["K"].values()) == m


def test_supercell_site_counts_matches_pycce_formula():
    cols = np.diag([4.0, 5.0, 10.0])
    # size 60 A: int(60/4 + 1) = 16, int(60/5 + 1) = 13, int(60/10 + 1) = 7
    assert pr.supercell_site_counts(cols, {"K": 2, "O": 3}, 60) == {"K": 16 * 13 * 7 * 2, "O": 16 * 13 * 7 * 3}


class _Cell:
    """Minimal BathCell: records the isotopes in force at each gen_supercell call."""
    def __init__(self, n_k_sites):
        self.cell = np.eye(3) * 1.0
        self.atoms = {"K": [[0, 0, 0]] * n_k_sites}
        self.isotopes = {}
        self.seen = []

    def gen_supercell(self, size, remove=None, seed=None):
        self.seen.append(dict(self.isotopes))
        iso = self.isotopes or {"K": K}
        counts = pr.supercell_site_counts(self.cell, {"K": len(self.atoms["K"])}, size)
        pycce_sample(counts["K"], iso["K"], seed)
        return np.zeros((0, 3))


def test_bath_uses_guard_and_restores_cell_isotopes(monkeypatch):
    """_site_step's bath() swaps in the safe dict only for the draw and restores cell.isotopes."""
    import types
    pc = types.SimpleNamespace(common_concentrations={"K": K})
    size = 1
    n_sites = overflowing_n(K) // 8  # 1 A supercell on a 1 A cell: int(1 + 1)^3 = 8 cells
    while sum(int(round(np.int32(8 * n_sites) * c)) for c in K.values()) <= 8 * n_sites:
        n_sites += 1
    cell = _Cell(n_sites)
    with pytest.raises(ValueError):
        cell.gen_supercell(size, seed=1)
    assert pr._default_isotopes(pc, cell) == {"K": K}
    proto = pr.CCEProtocol(supercell=size, min_bath_spins=None, max_bath_spins=None, n_conf=1)
    with pytest.raises(Exception) as ei:  # the draw succeeds; the run stops later on the empty bath
        pr._site_step(pc, cell, "K", np.zeros(3), 50.0, 15.0, proto, ts=np.linspace(0, 1, 5),
                      b_unit=np.array([0, 0, 1.0]))
    assert "larger sample" not in str(ei.value)
    assert cell.seen[-1]["K"]["39K"] < K["39K"] and cell.isotopes == {}
