import numpy as np
import pytest

from qkhosts.kernels import statevector as sv
from qkhosts.kernels.classical import matern52, product_cos2, rbf
from qkhosts.kernels.diagnostics import (effective_dimension, geometric_difference, kernel_target_alignment,
                                         offdiag_variance)
from qkhosts.kernels.feature_maps import FeatureMap
from qkhosts.kernels.quantum import (fidelity_kernel, global_depolarizing, projected_kernel, psd_project,
                                     shot_noise_fidelity)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("name", ["product_angle", "iqp", "hea"])
@pytest.mark.parametrize("q", [1, 2, 3, 5])
def test_states_normalised_and_kernel_props(name, q):
    X = rng.uniform(-1, 1, (7, q))
    fm = FeatureMap(name, layers=2, bandwidth=0.7)
    S = fm.states(X)
    assert np.allclose(np.linalg.norm(S, axis=1), 1)
    K = fidelity_kernel(fm, X)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_product_angle_matches_closed_form():
    X, Y = rng.uniform(-1, 1, (6, 4)), rng.uniform(-1, 1, (5, 4))
    fm = FeatureMap("product_angle", bandwidth=0.9)
    assert np.allclose(fidelity_kernel(fm, X, Y), product_cos2(X, Y, 0.9), atol=1e-12)


def test_single_qubit_gates_against_dense():
    q, j = 3, 1
    psi = rng.normal(size=(2, 8)) + 1j * rng.normal(size=(2, 8))
    U = sv.ry(np.array(0.37))
    dense = np.kron(np.kron(np.eye(2), U), np.eye(2))
    assert np.allclose(sv.apply_1q(psi, q, j, U), psi @ dense.T)


def test_cnot_against_dense():
    q = 2
    psi = rng.normal(size=(1, 4)) + 0j
    CN = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(sv.cnot(psi, q, 0, 1), psi @ CN.T)


def test_pauli_expectations_bloch():
    th = np.array([0.3, 1.1])
    psi = sv.apply_1q(sv.zero_state(2, 1), 1, 0, sv.ry(th))
    P = sv.pauli_expectations(psi, 1)
    assert np.allclose(P[:, 0, 0], np.sin(th)) and np.allclose(P[:, 0, 2], np.cos(th)) and np.allclose(P[:, 0, 1], 0)


def test_projected_kernel_props():
    X = rng.uniform(-1, 1, (8, 4))
    K = projected_kernel(FeatureMap("iqp", 1, 0.5), X, gamma=2.0)
    assert np.allclose(np.diag(K), 1) and np.linalg.eigvalsh(K).min() > -1e-10


def test_bandwidth_to_zero_gives_constant_kernel():
    X = rng.uniform(-1, 1, (5, 3))
    K = fidelity_kernel(FeatureMap("iqp", 1, 1e-6), X)
    # IQP has a pi^2 pair term independent of x, but the kernel depends only on differences -> ~1
    assert np.allclose(K, 1, atol=1e-6)


def test_noise_models():
    X = rng.uniform(-1, 1, (10, 3))
    K = fidelity_kernel(FeatureMap("hea", 2, 1.0), X)
    Kn = shot_noise_fidelity(K, 200, np.random.default_rng(1))
    assert np.allclose(np.diag(Kn), 1) and np.allclose(Kn, Kn.T)
    assert np.linalg.eigvalsh(psd_project(Kn)).min() > -1e-10
    Kd = global_depolarizing(K, 1.0, 3)
    assert np.allclose(Kd, K)


def test_classical_kernels_and_diagnostics():
    X = rng.uniform(-1, 1, (12, 3))
    y = X[:, 0] ** 2
    for k in (rbf(X, lengthscale=0.8), matern52(X, lengthscale=[0.5, 1, 2])):
        assert np.allclose(np.diag(k), 1)
    K = rbf(X, lengthscale=0.8)
    assert geometric_difference(K, K, lam=1e-6) == pytest.approx(1.0, rel=1e-2)
    assert -1 <= kernel_target_alignment(K, y) <= 1
    assert 0 < effective_dimension(K) <= 12
    assert offdiag_variance(np.eye(5)) == 0
