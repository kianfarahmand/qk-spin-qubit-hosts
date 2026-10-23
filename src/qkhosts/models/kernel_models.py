"""Estimators on precomputed kernels + equal-budget hyperparameter selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.svm import SVR

KernelFn = Callable[[dict, np.ndarray, np.ndarray | None], np.ndarray]


@dataclass
class KRR:
    alpha: float = 1e-3
    _a: np.ndarray | None = field(default=None, repr=False)

    def fit(self, K: np.ndarray, y: np.ndarray) -> "KRR":
        self._mu = float(np.mean(y))
        c = cho_factor(K + self.alpha * np.eye(len(K)))
        self._a = cho_solve(c, y - self._mu)
        return self

    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        return K_test_train @ self._a + self._mu


@dataclass
class GP:
    """Zero-mean GP on centred targets with amplitude s2 and noise variance `noise`."""
    noise: float = 1e-2
    amplitude: float = 1.0

    def fit(self, K: np.ndarray, y: np.ndarray) -> "GP":
        self._mu = float(np.mean(y))
        self._yc = y - self._mu
        self._c = cho_factor(self.amplitude * K + self.noise * np.eye(len(K)))
        self._a = cho_solve(self._c, self._yc)
        return self

    def predict(self, K_test_train: np.ndarray, k_test_diag: np.ndarray | None = None, return_std: bool = False):
        Ks = self.amplitude * K_test_train
        mean = Ks @ self._a + self._mu
        if not return_std:
            return mean
        kd = self.amplitude * (np.ones(len(Ks)) if k_test_diag is None else k_test_diag)
        v = cho_solve(self._c, Ks.T)
        var = np.clip(kd - np.sum(Ks * v.T, axis=1), 1e-12, None)
        return mean, np.sqrt(var + self.noise)

    def log_marginal_likelihood(self) -> float:
        L = self._c[0]
        return float(-0.5 * self._yc @ self._a - np.sum(np.log(np.diag(L))) - 0.5 * len(self._yc) * np.log(2 * np.pi))


@dataclass
class KSVR:
    C: float = 1.0
    epsilon: float = 0.05

    def fit(self, K, y):
        self._m = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon).fit(K, y)
        return self

    def predict(self, K_test_train):
        return self._m.predict(K_test_train)


ESTIMATORS = {"krr": KRR, "gp": GP, "svr": KSVR}


def select_and_fit(kernel_fn: KernelFn, kernel_grid: Sequence[dict], est_name: str, est_grid: Sequence[dict],
                   X: np.ndarray, y: np.ndarray, inner_folds: list, budget: int, seed: int = 0):
    """Random-subsample the (kernel x estimator) grid to exactly `budget` configs (equal-budget rule),
    pick by inner-CV MAE, refit on all of (X, y). Returns (model, kernel_params, est_params, cv_table)."""
    grid = [(kp, ep) for kp in kernel_grid for ep in est_grid]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(grid), size=min(budget, len(grid)), replace=False)
    table = []
    for gi in pick:
        kp, ep = grid[gi]
        K = kernel_fn(kp, X, None)
        errs = []
        for tr, va in inner_folds:
            try:
                m = ESTIMATORS[est_name](**ep).fit(K[np.ix_(tr, tr)], y[tr])
                errs.append(np.mean(np.abs(m.predict(K[np.ix_(va, tr)]) - y[va])))
            except np.linalg.LinAlgError:
                errs.append(np.inf)
        table.append({"kernel": kp, "est": ep, "cv_mae": float(np.mean(errs))})
    best = min(table, key=lambda r: r["cv_mae"])
    K = kernel_fn(best["kernel"], X, None)
    model = ESTIMATORS[est_name](**best["est"]).fit(K, y)
    return model, best["kernel"], best["est"], table
