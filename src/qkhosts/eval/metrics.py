"""Regression + screening metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def spearman(y, p):
    return float(spearmanr(y, p).statistic)


def topk_precision(y_true, y_score, frac: float) -> float:
    """|predicted top-k ∩ true top-k| / k, k = ceil(frac * n). Higher y = better host."""
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    t = set(np.argsort(-np.asarray(y_true))[:k])
    s = set(np.argsort(-np.asarray(y_score))[:k])
    return len(t & s) / k


def enrichment_factor(y_true, y_score, frac: float, threshold: float) -> float:
    """Hit rate (y_true >= threshold) in predicted top-frac divided by base rate."""
    y_true = np.asarray(y_true)
    k = max(1, int(np.ceil(frac * len(y_true))))
    top = np.argsort(-np.asarray(y_score))[:k]
    base = np.mean(y_true >= threshold)
    return float(np.mean(y_true[top] >= threshold) / base) if base > 0 else float("nan")


def gaussian_nll(y, mean, std):
    y, mean, std = map(np.asarray, (y, mean, std))
    return float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * ((y - mean) / std) ** 2))


def coverage(y, mean, std, z: float = 1.96):
    y, mean, std = map(np.asarray, (y, mean, std))
    return float(np.mean(np.abs(y - mean) <= z * std))


def all_regression(y, p) -> dict:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p), "spearman": spearman(y, p)}
