"""Fidelity (FQK) and projected (PQK) quantum kernels, with finite-shot and noise models."""
from __future__ import annotations

import numpy as np

from qkhosts.kernels import statevector as sv
from qkhosts.kernels.feature_maps import FeatureMap


def fidelity_kernel(fm: FeatureMap, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """K_ij = |<psi(x_i)|psi(y_j)>|^2 (exact statevector)."""
    SX = fm.states(X)
    SY = SX if Y is None else fm.states(Y)
    return np.abs(SX.conj() @ SY.T) ** 2


def projected_kernel(fm: FeatureMap, X: np.ndarray, Y: np.ndarray | None = None, gamma: float = 1.0) -> np.ndarray:
    """K = exp(-gamma * sum_k ||rho_k(x) - rho_k(y)||_F^2), ||.||_F^2 = 0.5 * sum_P (<P>_x - <P>_y)^2."""
    q = np.atleast_2d(X).shape[1]
    PX = sv.pauli_expectations(fm.states(X), q).reshape(len(X), -1)
    PY = PX if Y is None else sv.pauli_expectations(fm.states(Y), q).reshape(len(Y), -1)
    d2 = (PX ** 2).sum(1)[:, None] + (PY ** 2).sum(1)[None, :] - 2 * PX @ PY.T
    return np.exp(-gamma * 0.5 * np.clip(d2, 0, None))


def shot_noise_fidelity(K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool = True) -> np.ndarray:
    """Binomial estimate of each fidelity entry (e.g. compute-uncompute, P(0...0))."""
    Kn = rng.binomial(shots, np.clip(K, 0, 1)) / shots
    if symmetric and K.shape[0] == K.shape[1]:
        Kn = np.triu(Kn, 1)
        Kn = Kn + Kn.T + np.eye(K.shape[0])
    return Kn


def global_depolarizing(K: np.ndarray, p: float, q: int) -> np.ndarray:
    """Each state rho -> p rho + (1-p) I/2^q  =>  Tr(rho rho') -> p^2 K + (1-p^2)/2^q. (Approximation.)"""
    return p ** 2 * K + (1 - p ** 2) / 2 ** q


def psd_project(K: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Clip negative eigenvalues (needed after shot noise)."""
    Ks = (K + K.T) / 2
    w, V = np.linalg.eigh(Ks)
    return (V * np.clip(w, eps, None)) @ V.T
