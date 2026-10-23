"""Quantum feature maps (data-encoding circuits).

Inputs X: (N, d) scaled to [-1, 1] on the training fold; d must equal q (one feature per qubit).
Encoded angle theta = bandwidth * pi * x. `bandwidth` is the key hyperparameter (Shaydulin & Wild 2022).

Maps:
  product_angle : RY(theta_j)|0> per qubit. No entanglement -> fidelity kernel is the classical closed
                  form prod_j cos^2((theta_j - theta'_j)/2). Kept as the 'classically trivial' ablation.
  iqp           : (U_Phi(theta) H^{(x)q})^L, U_Phi = exp(i[sum_j theta_j Z_j + sum_(j,k) (pi-theta_j)(pi-theta_k) Z_j Z_k])
                  (Havlicek et al. 2019 style), pairs = ring or full.
  hea           : L x [RY(theta_j) RZ(theta_j) on all qubits; CNOT ring]  (data re-uploading).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qkhosts.kernels import statevector as sv


@dataclass(frozen=True)
class FeatureMap:
    name: str = "iqp"
    layers: int = 2
    bandwidth: float = 1.0
    pairs: str = "ring"  # iqp only: ring | full

    def angles(self, X: np.ndarray) -> np.ndarray:
        return self.bandwidth * np.pi * np.asarray(X, dtype=float)

    def states(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        q = X.shape[1]
        th = self.angles(X)
        if self.name == "product_angle":
            return _product_angle(th, q)
        if self.name == "iqp":
            return _iqp(th, q, self.layers, self.pairs)
        if self.name == "hea":
            return _hea(th, q, self.layers)
        raise ValueError(f"unknown feature map {self.name}")


def _pairs(q: int, kind: str) -> list[tuple[int, int]]:
    if q < 2:
        return []
    if kind == "full":
        return [(j, k) for j in range(q) for k in range(j + 1, q)]
    ring = [(j, j + 1) for j in range(q - 1)]
    return ring + ([(q - 1, 0)] if q > 2 else [])


def _product_angle(th: np.ndarray, q: int) -> np.ndarray:
    psi = sv.zero_state(th.shape[0], q)
    for j in range(q):
        psi = sv.apply_1q(psi, q, j, sv.ry(th[:, j]))
    return psi


def _iqp(th: np.ndarray, q: int, layers: int, pairs: str) -> np.ndarray:
    Z = sv.z_signs(q)  # (2^q, q)
    phase = th @ Z.T  # sum_j theta_j z_j
    for j, k in _pairs(q, pairs):
        phase += ((np.pi - th[:, j]) * (np.pi - th[:, k]))[:, None] * (Z[:, j] * Z[:, k])[None, :]
    diag = np.exp(1j * phase)
    psi = sv.zero_state(th.shape[0], q)
    for _ in range(layers):
        psi = sv.hadamard_all(psi, q)
        psi = psi * diag
    return psi


def _hea(th: np.ndarray, q: int, layers: int) -> np.ndarray:
    psi = sv.zero_state(th.shape[0], q)
    for _ in range(layers):
        for j in range(q):
            psi = sv.apply_1q(psi, q, j, sv.ry(th[:, j]))
            psi = sv.apply_1q(psi, q, j, sv.rz(th[:, j]))
        if q > 1:
            for j in range(q if q > 2 else 1):
                psi = sv.cnot(psi, q, j, (j + 1) % q)
    return psi
