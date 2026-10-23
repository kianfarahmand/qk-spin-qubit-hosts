"""Statistics for paired model comparison (pre-registered in prereg/preregistration.md §5)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(diffs, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected resampled t-test on J paired differences.
    var_corr = (1/J + n_test/n_train) * s^2 ; t = mean / sqrt(var_corr) ; df = J - 1. Two-sided p."""
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    if J < 2:
        raise ValueError("need >= 2 differences")
    s2 = d.var(ddof=1)
    if s2 == 0:
        return (np.inf if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0)
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * s2)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    b = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def h1_decision(per_n: dict, ratio_thr: float = 0.90, alpha: float = 0.05, min_hits: int = 2) -> dict:
    """per_n: {N: {"mae_q": [...J], "mae_c": [...J], "n_train": int, "n_test": int}} -> decision dict.
    Implements prereg §5 exactly; returns per-N details + overall verdict."""
    Ns = sorted(per_n)
    rows, raw_p = [], []
    for N in Ns:
        r = per_n[N]
        q, c = np.asarray(r["mae_q"]), np.asarray(r["mae_c"])
        d = q - c
        t, p = corrected_resampled_ttest(d, r["n_train"], r["n_test"])
        rel = d / c
        rows.append({"N": N, "ratio_median": float(np.median(q / c)), "t": t, "p": p,
                     "ci_diff": bootstrap_ci(d), "ci_rel": bootstrap_ci(rel)})
        raw_p.append(p)
    for row, pa in zip(rows, holm(raw_p)):
        row["p_holm"] = float(pa)
        lo, hi = row["ci_diff"]
        row["advantage"] = row["ratio_median"] <= ratio_thr and pa < alpha and hi < 0
        row["classical_better"] = row["ratio_median"] >= 1 / ratio_thr and pa < alpha and lo > 0
        rl, rh = row["ci_rel"]
        row["parity"] = rl >= -0.05 and rh <= 0.05
    hits = lambda k: sum(r[k] for r in rows) >= min_hits
    verdict = ("advantage" if hits("advantage") else "classical_better" if hits("classical_better")
               else "parity" if hits("parity") else "inconclusive")
    return {"verdict": verdict, "per_n": rows}
