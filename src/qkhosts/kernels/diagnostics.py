"""Kernel diagnostics used to interpret (and bound) quantum-kernel performance."""
from __future__ import annotations

import numpy as np


def _sqrtm_psd(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def _trace_normalise(K):
    return K * (K.shape[0] / np.trace(K))


def offdiag_variance(K: np.ndarray) -> float:
    """Exponential concentration indicator (Thanasilp et al. 2024): Var of off-diagonal entries."""
    m = ~np.eye(K.shape[0], dtype=bool)
    return float(np.var(K[m]))


def geometric_difference(K_c: np.ndarray, K_q: np.ndarray, lam: float = 1e-3) -> float:
    """g(K_C || K_Q) = sqrt(|| sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf), trace-normalised to N
    (Huang et al. 2021). Small g => classical kernel can match the quantum kernel on this data."""
    Kc, Kq = _trace_normalise(K_c), _trace_normalise(K_q)
    s = _sqrtm_psd(Kq)
    M = s @ np.linalg.solve(Kc + lam * np.eye(len(Kc)), s)
    return float(np.sqrt(np.linalg.norm((M + M.T) / 2, 2)))


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment (Cortes et al. 2012)."""
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    yc = y - y.mean()
    Y = np.outer(yc, yc)
    return float((Kc * Y).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Y) + 1e-300))


def effective_dimension(K: np.ndarray, lam: float = 1e-3) -> float:
    """d_eff = sum_i mu_i/(mu_i + lam), mu = eigenvalues of K/N."""
    mu = np.clip(np.linalg.eigvalsh((K + K.T) / 2) / K.shape[0], 0, None)
    return float(np.sum(mu / (mu + lam)))
