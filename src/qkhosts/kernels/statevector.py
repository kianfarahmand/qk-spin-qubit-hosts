"""Minimal batched statevector simulator (numpy) for kernel feature maps.

Conventions: q qubits, qubit 0 is the most significant bit of the basis index.
States are complex arrays of shape (N, 2**q). Intended for q <= ~16.
Cross-checked against PennyLane in task T-041.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SQRT2 = np.sqrt(2.0)


@lru_cache(maxsize=None)
def z_signs(q: int) -> np.ndarray:
    """(2**q, q) array of Z eigenvalues (+1 for |0>, -1 for |1>)."""
    b = np.arange(2 ** q)[:, None] >> (q - 1 - np.arange(q))[None, :] & 1
    return (1 - 2 * b).astype(float)


def zero_state(n: int, q: int) -> np.ndarray:
    psi = np.zeros((n, 2 ** q), dtype=complex)
    psi[:, 0] = 1.0
    return psi


def apply_1q(psi: np.ndarray, q: int, j: int, U: np.ndarray) -> np.ndarray:
    """Apply single-qubit gate on qubit j. U shape (2,2) or per-sample (N,2,2)."""
    n = psi.shape[0]
    v = psi.reshape(n, 2 ** j, 2, 2 ** (q - j - 1))
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    if U.ndim == 2:
        U = np.broadcast_to(U, (n, 2, 2))
    u = U[:, :, :, None, None]
    out = np.stack([u[:, 0, 0] * a + u[:, 0, 1] * b, u[:, 1, 0] * a + u[:, 1, 1] * b], axis=2)
    return out.reshape(n, 2 ** q)


def hadamard_all(psi: np.ndarray, q: int) -> np.ndarray:
    H = np.array([[1, 1], [1, -1]], dtype=complex) / SQRT2
    for j in range(q):
        psi = apply_1q(psi, q, j, H)
    return psi


def ry(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def rz(theta: np.ndarray) -> np.ndarray:
    e = np.exp(-0.5j * theta)
    z = np.zeros_like(e)
    return np.stack([np.stack([e, z], -1), np.stack([z, np.conj(e)], -1)], -2)


@lru_cache(maxsize=None)
def _cnot_perm(q: int, c: int, t: int) -> np.ndarray:
    idx = np.arange(2 ** q)
    cbit = (idx >> (q - 1 - c)) & 1
    return idx ^ (cbit << (q - 1 - t))


def cnot(psi: np.ndarray, q: int, c: int, t: int) -> np.ndarray:
    return psi[:, _cnot_perm(q, c, t)]


def pauli_expectations(psi: np.ndarray, q: int) -> np.ndarray:
    """(N, q, 3) array of <X_k>, <Y_k>, <Z_k> for each qubit."""
    n = psi.shape[0]
    out = np.empty((n, q, 3))
    for k in range(q):
        v = psi.reshape(n, 2 ** k, 2, 2 ** (q - k - 1))
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        ab = np.sum(np.conj(a) * b, axis=(1, 2))
        out[:, k, 0] = 2 * ab.real
        out[:, k, 1] = 2 * ab.imag
        out[:, k, 2] = np.sum(np.abs(a) ** 2 - np.abs(b) ** 2, axis=(1, 2))
    return out
