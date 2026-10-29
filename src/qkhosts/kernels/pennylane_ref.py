"""PennyLane reference circuits for the feature maps in `feature_maps.py` (task T-041).

Written from the gate definitions, not from the numpy code, so that the two implementations check
each other. PennyLane's wire 0 is the most significant bit of the basis index, which is the same
convention as `statevector.py`, so states can be compared entry by entry (no global phase is dropped).

Gate identities used (PennyLane definitions):
  RZ(a)       = exp(-i a/2 Z)        ->  exp(i theta Z)     = RZ(-2 theta)
  IsingZZ(a)  = exp(-i a/2 Z(x)Z)    ->  exp(i phi Z(x)Z)   = IsingZZ(-2 phi)

Needs the optional extra `[qml]` (pennylane). Angles theta = bandwidth * pi * x, as in FeatureMap.
"""
from __future__ import annotations

import numpy as np

from qkhosts.kernels.feature_maps import FeatureMap


def _qml():
    import pennylane as qml  # optional dependency

    return qml


def _ring_or_full(q: int, kind: str) -> list[tuple[int, int]]:
    if q < 2:
        return []
    if kind == "full":
        return [(j, k) for j in range(q) for k in range(j + 1, q)]
    return [(j, j + 1) for j in range(q - 1)] + ([(q - 1, 0)] if q > 2 else [])


def apply_feature_map(fm: FeatureMap, theta: np.ndarray) -> None:
    """Queue the gates of `fm` for one sample; theta has shape (q,) in radians."""
    qml = _qml()
    q = len(theta)
    if fm.name == "product_angle":
        for j in range(q):
            qml.RY(theta[j], wires=j)
    elif fm.name == "iqp":
        for _ in range(fm.layers):
            for j in range(q):
                qml.Hadamard(wires=j)
            for j in range(q):
                qml.RZ(-2.0 * theta[j], wires=j)
            for j, k in _ring_or_full(q, fm.pairs):
                qml.IsingZZ(-2.0 * (np.pi - theta[j]) * (np.pi - theta[k]), wires=[j, k])
    elif fm.name == "hea":
        for _ in range(fm.layers):
            for j in range(q):
                qml.RY(theta[j], wires=j)
                qml.RZ(theta[j], wires=j)
            if q == 2:
                qml.CNOT(wires=[0, 1])
            elif q > 2:
                for j in range(q):
                    qml.CNOT(wires=[j, (j + 1) % q])
    else:
        raise ValueError(f"unknown feature map {fm.name}")


def states(fm: FeatureMap, X: np.ndarray) -> np.ndarray:
    """(N, 2**q) complex statevectors from PennyLane default.qubit."""
    qml = _qml()
    X = np.atleast_2d(X)
    q = X.shape[1]
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def circuit(theta):
        apply_feature_map(fm, theta)
        return qml.state()

    return np.stack([np.asarray(circuit(t), dtype=complex) for t in fm.angles(X)])


def pauli_expectations(fm: FeatureMap, X: np.ndarray) -> np.ndarray:
    """(N, q, 3) array of <X_k>, <Y_k>, <Z_k> measured with qml.expval."""
    qml = _qml()
    X = np.atleast_2d(X)
    q = X.shape[1]
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def circuit(theta):
        apply_feature_map(fm, theta)
        return [qml.expval(P(k)) for k in range(q) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]

    return np.stack([np.asarray(circuit(t), dtype=float).reshape(q, 3) for t in fm.angles(X)])


def fidelity_kernel_overlap(fm: FeatureMap, x: np.ndarray, y: np.ndarray) -> float:
    """K(x, y) as P(0...0) of the compute-uncompute circuit U(y)^dagger U(x)|0> (the hardware estimator)."""
    qml = _qml()
    q = len(x)
    dev = qml.device("default.qubit", wires=q)
    tx, ty = fm.angles(x), fm.angles(y)

    @qml.qnode(dev)
    def circuit():
        apply_feature_map(fm, tx)
        qml.adjoint(apply_feature_map)(fm, ty)
        return qml.probs(wires=range(q))

    return float(np.asarray(circuit())[0])
