"""Leakage-aware splits for materials data.

All functions return a list of (train_idx, test_idx) numpy arrays and are deterministic
given `seed`. Group key default: chemical system = sorted unique elements joined by '-'.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold


def chemsys(elements: Iterable[str]) -> str:
    return "-".join(sorted(set(elements)))


def random_kfold(n: int, k: int = 5, seed: int = 0):
    return [(tr, te) for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.arange(n))]


def group_kfold(groups: Sequence[str], k: int = 5, seed: int = 0):
    """GroupKFold with a seeded permutation of group labels (sklearn's is deterministic)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    relabel = dict(zip(uniq, perm))
    g = np.array([relabel[x] for x in groups])
    return list(GroupKFold(k).split(np.zeros(len(g)), groups=g))


def leave_element_out(element_sets: Sequence[set[str]], elements: Sequence[str]):
    """One fold per element E: test = all materials containing E."""
    out = []
    for e in elements:
        te = np.array([i for i, s in enumerate(element_sets) if e in s])
        tr = np.setdiff1d(np.arange(len(element_sets)), te)
        if len(te) and len(tr):
            out.append((tr, te))
    return out


def cluster_out(X: np.ndarray, k: int = 5, seed: int = 0):
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    return [(np.where(lab != c)[0], np.where(lab == c)[0]) for c in range(k)]


def tail_split(y: np.ndarray, quantile: float = 0.9):
    thr = np.quantile(y, quantile)
    return [(np.where(y < thr)[0], np.where(y >= thr)[0])]


def subsample(train_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(train_idx):
        return np.asarray(train_idx)
    return np.sort(np.random.default_rng(seed).choice(train_idx, n, replace=False))


def assert_disjoint_groups(folds, groups) -> None:
    groups = np.asarray(groups)
    for tr, te in folds:
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("group leakage between train and test")
