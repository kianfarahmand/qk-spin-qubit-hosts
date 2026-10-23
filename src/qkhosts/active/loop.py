"""Pool-based retrospective active learning with a GP on a precomputed pool kernel."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from qkhosts.models.kernel_models import GP


def acquire(mean, std, best, kind: str, beta: float = 2.0, rng=None):
    if kind == "ucb":
        return mean + beta * std
    if kind == "ei":
        z = (mean - best) / std
        return (mean - best) * norm.cdf(z) + std * norm.pdf(z)
    if kind == "greedy":
        return mean
    if kind == "random":
        return rng.random(len(mean))
    raise ValueError(kind)


def run_al(K_pool: np.ndarray, y: np.ndarray, n_init: int, batch: int, n_rounds: int, kind: str,
           top_frac: float = 0.05, gp_kwargs: dict | None = None, seed: int = 0, prior_score=None) -> dict:
    """Returns recall-of-true-top-k after each round. `prior_score` (e.g. scaling-law T2) enables
    the 'sl_greedy' arm: acquire by prior score only, no model."""
    rng = np.random.default_rng(seed)
    n = len(y)
    k = max(1, int(np.ceil(top_frac * n)))
    top = set(np.argsort(-y)[:k])
    labelled = list(rng.choice(n, n_init, replace=False))
    recall = [len(top & set(labelled)) / k]
    for _ in range(n_rounds):
        pool = np.setdiff1d(np.arange(n), labelled)
        if len(pool) == 0:
            break
        if kind == "sl_greedy":
            score = np.asarray(prior_score)[pool]
        else:
            L = np.array(labelled)
            gp = GP(**(gp_kwargs or {})).fit(K_pool[np.ix_(L, L)], y[L])
            mu, sd = gp.predict(K_pool[np.ix_(pool, L)], np.diag(K_pool)[pool], return_std=True)
            score = acquire(mu, sd, y[L].max(), kind, rng=rng)
        labelled += list(pool[np.argsort(-score)[:batch]])
        recall.append(len(top & set(labelled)) / k)
    return {"recall": recall, "labelled": [int(i) for i in labelled], "k": k}


def labels_to_recall(recall, n_init: int, batch: int, target: float = 0.8):
    for r, v in enumerate(recall):
        if v >= target:
            return n_init + r * batch
    return None
