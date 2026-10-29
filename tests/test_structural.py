"""Structural descriptors (T-013) on hand-checkable toy crystals. Radii and isotope rows are test
fixtures, not physical data."""
import numpy as np
import pandas as pd
import pytest

from qkhosts.features.structural import (Crystal, component_dimensionalities, dimensionality,
                                         lattice_anisotropy, nn_spinful, structural_descriptors)

R = {"X": 1.0, "Y": 1.0}
# X has one spinful isotope (p = 0.5, g = 2, I = 1/2); Y has none.
ISO = pd.DataFrame({"element": ["X", "X", "Y"], "isotope": ["1X", "2X", "1Y"], "abundance": [0.5, 0.5, 1.0],
                    "spin_I": [0.5, 0.0, 0.0], "g_factor": [2.0, 0.0, 0.0]})


def ortho(a, b, c, frac=((0, 0, 0),), species=("X",)):
    return Crystal(np.diag([a, b, c]), np.array(frac, float), species)


def test_anisotropy():
    assert lattice_anisotropy(ortho(2, 2, 2)) == pytest.approx(1.0)
    assert lattice_anisotropy(ortho(2, 3, 8)) == pytest.approx(4.0)


@pytest.mark.parametrize("abc,dim", [((2, 2, 2), 3), ((2, 2, 9), 2), ((2, 9, 9), 1), ((9, 9, 9), 0)])
def test_dimensionality_single_atom(abc, dim):
    # bond length 2 Å < 1.2 * 2 Å bonds along short axes only
    assert component_dimensionalities(ortho(*abc), R) == [dim]


def test_dimensionality_molecule_in_layered_host():
    # a 2D sheet of X (a = b = 2) plus a Y-Y dimer far from it in a tall cell
    c = Crystal(np.diag([2.0, 2.0, 20.0]),
                np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.55]]), ("X", "Y", "Y"))
    # the Y dimer is 1 Å apart and its images are 2 Å apart in-plane -> the Y component is also a sheet
    assert sorted(component_dimensionalities(c, R)) == [2, 2]
    # Y radius 0.45: dimer bond 1 Å < 1.08 Å kept, Y images at 2 Å not bonded -> one 0D molecule
    assert sorted(component_dimensionalities(c, {"X": 1.0, "Y": 0.45})) == [0, 2]
    d = dimensionality(c, {"X": 1.0, "Y": 0.45})
    assert d == {"dim_max": 2.0, "dim_min": 0.0, "is_low_dim": 1.0}


def test_dimensionality_cycle_through_two_atoms():
    # rock-salt-like 1D chain X-Y-X-Y along a with a = 4, X-Y = 2 Å: the cycle closes only via 2 bonds
    c = Crystal(np.diag([4.0, 10.0, 10.0]), np.array([[0, 0, 0], [0.5, 0, 0]]), ("X", "Y"))
    assert component_dimensionalities(c, R) == [1]


def test_dimensionality_skewed_cell():
    # same simple-cubic lattice in a sheared basis (a, a+b, c): still 3D
    lat = np.array([[2.0, 0, 0], [2.0, 2.0, 0], [0, 0, 2.0]])
    assert component_dimensionalities(Crystal(lat, np.zeros((1, 3)), ("X",)), R) == [3]


def test_nn_spinful_simple_cubic():
    # two-site cell: X at origin, Y at body centre; centre = Y site (removed)
    c = Crystal(np.diag([3.0, 3.0, 3.0]), np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), ("X", "Y"))
    d1 = 3.0 * np.sqrt(3) / 2  # 8 X corners
    out = nn_spinful(c, 1, ISO, r_count=3.0, r_max=3.0)
    assert out["nn_spinful_d1_A"] == pytest.approx(d1)
    assert out["nn_spinful_dk_mean_A"] == pytest.approx(d1)
    assert out["n_spinful_exp_r"] == pytest.approx(8 * 0.5)
    m2 = 8 * 0.5 * 4 * 0.75 / d1 ** 6
    assert out["log10_m2_local"] == pytest.approx(np.log10(m2))


def test_nn_spinful_removes_center_atom():
    c = ortho(3, 3, 3)  # X at the origin is the removed site; nearest X images at 3 Å (6 of them)
    out = nn_spinful(c, 0, ISO, r_count=3.5, r_max=3.5)
    assert out["nn_spinful_d1_A"] == pytest.approx(3.0)
    assert out["n_spinful_exp_r"] == pytest.approx(6 * 0.5)


def test_nn_spinful_none():
    out = nn_spinful(ortho(3, 3, 3, species=("Y",)), 0, ISO)
    assert out["nn_spinful_d1_A"] == np.inf and out["log10_m2_local"] == -np.inf


def test_structural_descriptors_keys_and_center_average():
    c = Crystal(np.diag([3.0, 3.0, 3.0]), np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), ("X", "Y"))
    d = structural_descriptors(c, ISO, R, r_count=3.0, r_max=3.0)
    one = nn_spinful(c, 0, ISO, r_count=3.0, r_max=3.0)["nn_spinful_d1_A"]
    two = nn_spinful(c, 1, ISO, r_count=3.0, r_max=3.0)["nn_spinful_d1_A"]
    assert d["nn_spinful_d1_A"] == pytest.approx((one + two) / 2)
    assert {"lattice_anisotropy", "dim_max", "dim_min", "is_low_dim"} <= set(d)


def test_crystal_validation():
    with pytest.raises(ValueError):
        Crystal(np.zeros((3, 3)), np.zeros((1, 3)), ("X",))
    with pytest.raises(ValueError):
        Crystal(np.eye(3), np.zeros((2, 3)), ("X",))
