"""Configuration-level chunks for long CCE directions (T-015, S019; experiments/E2_notes.md §2).

A host direction whose 20 configurations exceed the 6 h Actions job limit (I = 9/2 baths) runs as several shards that
differ only in `seed` and `n_conf`: configuration k uses isotope seed `seed + 1 + k` (`pycce_runner._site_step`), so
chunks with seed = 0, m, 2m, ... and n_conf = m draw disjoint configurations, the same ones a single 20-configuration
run would draw. Each chunk also picks its own time window from its first configuration, so `merge_chunks`
interpolates onto the shortest window (linear; every window resolves the decay by construction) before averaging and
fitting, as `_site_step` does for one run.

S020: the merge needs only each chunk's mean curve and its number of configurations. Linear interpolation commutes
with averaging, so the n_conf_ok-weighted mean of the interpolated chunk means equals the mean of the interpolated
configuration curves to rounding (`tests/test_chunks.py`). `store_conf_curves` is therefore optional; it is not used
in E2, which keeps the records small.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from qkhosts.labels.pycce_runner import fit_t2, fit_t2_nls, guard_fit, n_fit_points


def chunk_plan(n_conf: int, per_chunk: int, seed: int = 0) -> list[tuple[int, int]]:
    """(seed, n_conf) per chunk covering configurations 0..n_conf-1 of a run with `seed`."""
    if per_chunk < 1 or n_conf < 1:
        raise ValueError("n_conf and per_chunk must be >= 1")
    return [(seed + s, min(per_chunk, n_conf - s)) for s in range(0, n_conf, per_chunk)]


def merge_chunks(finals: Sequence[dict], min_fit_points: int = 15, fit_n_range: tuple = (0.5, 6.0),
                 max_conf_fail_frac: float = 0.1, n_conf_requested: int | None = None) -> dict:
    """Merge the `final` step dicts of the chunks of one host direction (each with `curve_t_ms`, `per_conf_t2_ms`,
    `conf_errors`, and either `conf_curves` or `curve_mean` + `n_conf_ok`). Returns the merged step: t2_ms and
    stretch_n of the averaged curve (guarded), the NLS fit, mean-of-fits ± SEM over all configurations, the common
    grid and the counts. Times in ms."""
    if not finals:
        raise ValueError("no chunks")
    for f in finals:
        if "conf_curves" not in f and not ("curve_mean" in f and "n_conf_ok" in f):
            raise ValueError("chunk lacks conf_curves and curve_mean/n_conf_ok")
    ts = np.asarray(min((f["curve_t_ms"] for f in finals), key=lambda t: t[-1]), float)
    acc, n_ok = np.zeros(len(ts)), 0
    for f in finals:
        t = np.asarray(f["curve_t_ms"], float)
        if "conf_curves" in f:
            for c in f["conf_curves"]:
                acc += np.interp(ts, t, np.asarray(c, float))
                n_ok += 1
        else:
            acc += int(f["n_conf_ok"]) * np.interp(ts, t, np.asarray(f["curve_mean"], float))
            n_ok += int(f["n_conf_ok"])
    if n_ok < 1:
        raise ValueError("no configurations in the chunks")
    mean_curve = acc / n_ok
    npts = n_fit_points(mean_curve)
    t2, n, flag = guard_fit(*fit_t2(ts, mean_curve), ts, npts, min_fit_points, fit_n_range)
    t2_nls, n_nls, _ = guard_fit(*fit_t2_nls(ts, mean_curve), ts, npts, min_fit_points, fit_n_range)
    errors = [e for f in finals for e in f.get("conf_errors", [])]
    n_req = n_conf_requested if n_conf_requested is not None else n_ok + len(errors)
    if len(errors) > max_conf_fail_frac * n_req:
        t2, n, t2_nls, n_nls, flag = (float("nan"),) * 4 + ("conf_errors",)
    per = np.concatenate([np.asarray(f["per_conf_t2_ms"], float) for f in finals])
    ok = per[np.isfinite(per)]
    return {"t2_ms": t2, "stretch_n": n, "fit_flag": flag, "n_fit_points": npts, "t2_ms_nls": t2_nls,
            "stretch_n_nls": n_nls, "per_conf_t2_ms": per.tolist(),
            "t2_ms_mean_of_fits": float(ok.mean()) if len(ok) else float("nan"),
            "t2_sem_ms": float(ok.std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else float("nan"),
            "n_fit_failed": int(len(per) - len(ok)), "n_conf_ok": int(n_ok), "conf_errors": errors,
            "n_chunks": len(finals), "curve_t_ms": ts.tolist(), "curve_mean": mean_curve.tolist()}
