"""T-041: numpy statevector kernels agree with independent PennyLane circuits to < 1e-10 (q <= 8).

Skips where PennyLane is not installed (e.g. the public CI without the [qml] extra).
"""
import numpy as np
import pytest

pytest.importorskip("pennylane")

from qkhosts.kernels import pennylane_ref as pl  # noqa: E402
from qkhosts.kernels import statevector as sv  # noqa: E402
from qkhosts.kernels.feature_maps import FeatureMap  # noqa: E402
from qkhosts.kernels.quantum import fidelity_kernel, projected_kernel  # noqa: E402

TOL = 1e-10
MAPS = [("product_angle", 1, "ring"), ("iqp", 1, "ring"), ("iqp", 2, "ring"), ("iqp", 2, "full"),
        ("hea", 1, "ring"), ("hea", 3, "ring")]


@pytest.mark.parametrize("name,layers,pairs", MAPS)
@pytest.mark.parametrize("q", [1, 2, 3, 5, 8])
def test_states_and_kernels_match_pennylane(name, layers, pairs, q):
    rng = np.random.default_rng(1000 * q + layers)
    X = rng.uniform(-1, 1, (4, q))
    fm = FeatureMap(name, layers=layers, bandwidth=float(rng.uniform(0.3, 1.5)), pairs=pairs)
    S_np, S_pl = fm.states(X), pl.states(fm, X)
    assert np.max(np.abs(S_np - S_pl)) < TOL
    K_pl = np.abs(S_pl.conj() @ S_pl.T) ** 2
    assert np.max(np.abs(fidelity_kernel(fm, X) - K_pl)) < TOL
    P_pl = pl.pauli_expectations(fm, X)
    assert np.max(np.abs(sv.pauli_expectations(S_np, q) - P_pl)) < TOL
    d2 = ((P_pl.reshape(4, -1)[:, None, :] - P_pl.reshape(4, -1)[None, :, :]) ** 2).sum(-1)
    assert np.max(np.abs(projected_kernel(fm, X, gamma=0.7) - np.exp(-0.7 * 0.5 * d2))) < TOL


@pytest.mark.parametrize("name", ["iqp", "hea"])
def test_compute_uncompute_estimator_matches(name):
    rng = np.random.default_rng(7)
    q = 4
    X = rng.uniform(-1, 1, (3, q))
    fm = FeatureMap(name, layers=2, bandwidth=0.8)
    K = fidelity_kernel(fm, X)
    for i in range(3):
        for j in range(3):
            assert abs(K[i, j] - pl.fidelity_kernel_overlap(fm, X[i], X[j])) < TOL
