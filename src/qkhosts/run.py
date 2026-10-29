"""Experiment runner entry point (T-030):

    python -m qkhosts.run --config configs/experiments/E#.yaml [--set key.sub=value ...] [--out-root DIR]

The resolved config is `configs/base.yaml`, deep-merged with the experiment config, then with each
`--set` (the value is parsed as YAML). Every run writes `results/<id>/<UTC>_<gitsha>[-dirty]/` with
`config.yaml` (resolved, including overrides), `metrics.json`, `predictions.parquet` (or `.csv.gz` without
pyarrow) and `env.txt` (versions, git state, command line, times).

Experiment kinds live in `KINDS`. Implemented now: `smoke` and `e3` (classical learning curves on grouped folds).
The prereg guard (CLAUDE.md §2) refuses outer-test metrics for E3/E4 on non-synthetic data until `prereg/LOCK` exists.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from qkhosts.data.splits import assert_disjoint_groups, group_kfold, random_kfold, subsample
from qkhosts.eval.metrics import all_regression
from qkhosts.kernels.classical import linear, matern52, rbf
from qkhosts.kernels.feature_maps import FeatureMap
from qkhosts.kernels.quantum import fidelity_kernel
from qkhosts.models.kernel_models import select_and_fit

ROOT = Path(__file__).resolve().parents[2]
GUARDED_KINDS = {"e3", "e4"}  # outer-test metrics need prereg/LOCK unless data are synthetic


# ---------------------------------------------------------------- config

def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins. Lists and scalars are replaced, not merged. Inputs are not modified."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def apply_set(cfg: dict, assignment: str) -> dict:
    """Apply one `dotted.key=value` override; the value is parsed as YAML (so `5`, `1e-3`, `[1,2]`, `true` work)."""
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = key.strip().split(".")
    patch: dict = {}
    node = patch
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = _yaml_value(raw)
    return deep_merge(cfg, patch)


def _yaml_value(raw: str):
    """yaml.safe_load, except that YAML 1.1 strings such as '1e-3' (no dot) become floats."""
    v = yaml.safe_load(raw)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    if isinstance(v, list):
        return [_yaml_value(str(x)) if isinstance(x, str) else x for x in v]
    return v


def load_config(path: str | Path, sets: list[str] | None = None, base_path: str | Path | None = None) -> dict:
    """base.yaml <- experiment yaml <- --set overrides. Records the sources under `_provenance`."""
    base_path = Path(base_path) if base_path else ROOT / "configs" / "base.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    exp = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = deep_merge(base or {}, exp)
    for s in sets or []:
        cfg = apply_set(cfg, s)
    for req in ("id", "kind"):
        if req not in cfg:
            raise KeyError(f"config {path} lacks required key {req!r}")
    cfg["_provenance"] = {"config": str(path), "base": str(base_path), "set": list(sets or [])}
    return cfg


# ---------------------------------------------------------------- provenance

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    except OSError:
        return ""


def git_state() -> tuple[str, bool]:
    """(short sha or 'nogit', dirty flag). Dirty = tracked files modified."""
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return sha, dirty


def make_run_dir(out_root: Path, eid: str, sha: str, dirty: bool, now: dt.datetime) -> Path:
    """results/<eid>/<YYYYmmddTHHMMSSZ>_<sha>[-dirty]; suffix _2, _3... if the name exists."""
    name = f"{now:%Y%m%dT%H%M%SZ}_{sha}{'-dirty' if dirty else ''}"
    d = out_root / eid / name
    i = 2
    while d.exists():
        d = out_root / eid / f"{name}_{i}"
        i += 1
    d.mkdir(parents=True)
    return d


def env_text(sha: str, dirty: bool, argv: list[str], started: dt.datetime, ended: dt.datetime) -> str:
    import scipy
    import sklearn
    lines = [f"python {platform.python_version()} ({platform.platform()})",
             f"numpy {np.__version__}", f"scipy {scipy.__version__}", f"scikit-learn {sklearn.__version__}",
             f"pandas {pd.__version__}", f"pyyaml {yaml.__version__}",
             f"git {sha}{' (dirty: tracked files modified)' if dirty else ''}",
             f"argv {' '.join(argv)}", f"started {started.isoformat()}", f"ended {ended.isoformat()}"]
    return "\n".join(lines) + "\n"


def write_predictions(df: pd.DataFrame, out: Path) -> str:
    """predictions.parquet if a parquet engine is installed, else predictions.csv.gz. Returns the file name."""
    try:
        df.to_parquet(out / "predictions.parquet", index=False)
        return "predictions.parquet"
    except ImportError:
        df.to_csv(out / "predictions.csv.gz", index=False)
        return "predictions.csv.gz"


# ---------------------------------------------------------------- prereg guard

def check_prereg_guard(cfg: dict, root: Path = ROOT) -> None:
    """CLAUDE.md §2: no outer-test-fold metrics for E3/E4 before prereg/LOCK (synthetic data exempt)."""
    if cfg["kind"] not in GUARDED_KINDS:
        return
    synthetic = cfg.get("data", {}).get("source") == "synthetic"
    outer = cfg.get("eval", {}).get("outer", True)
    if outer and not synthetic and not (root / "prereg" / "LOCK").exists():
        raise PermissionError(f"{cfg['id']}: outer-test metrics on real data need prereg/LOCK (T-040). "
                              "Run with eval.outer=false (inner CV only) until the lock.")


# ---------------------------------------------------------------- data

def synthetic_dataset(n: int, d: int, n_groups: int, noise: float, seed: int) -> dict:
    """Grouped synthetic regression set standing in for (features, Δ = log10 T2_CCE − log10 T2_SL).
    Targets share a per-group offset so that grouped and random splits differ, as with chemical systems."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    groups = np.array([f"g{i:03d}" for i in rng.integers(0, n_groups, n)])
    offset = {g: rng.normal(0, 0.2) for g in np.unique(groups)}
    y = (0.6 * np.sin(2 * X[:, 0]) * X[:, 1] + 0.3 * X[:, 2] ** 2 + np.array([offset[g] for g in groups])
         + noise * rng.normal(size=n))
    return {"X": X, "y": y, "groups": groups, "ids": np.array([f"syn-{i:04d}" for i in range(n)])}


def load_dataset(cfg: dict) -> dict:
    src = cfg.get("data", {}).get("source", "synthetic")
    if src == "synthetic":
        s = cfg["data"].get("synthetic", {})
        return synthetic_dataset(s.get("n", 150), s.get("d", 6), s.get("n_groups", 30), s.get("noise", 0.05),
                                 s.get("seed", 0))
    raise NotImplementedError(f"data.source {src!r}: real D-CCE loading arrives with T-016/T-031")


def minmax_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = X.min(0), X.max(0)
    return lo, np.where(hi > lo, hi - lo, 1.0)


def minmax_apply(X: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] with train-fold statistics (test values may fall slightly outside)."""
    return 2 * (X - lo) / span - 1


# ---------------------------------------------------------------- model families (E3, classical)

def _family_table() -> dict[str, tuple[Callable, Callable[[dict], list[dict]]]]:
    """name -> (kernel_fn(kp, A, B), grid builder from the family's config block)."""
    def ls_grid(c):
        return [{"ls": float(v)} for v in np.logspace(*c.get("ls_logspace", [-1, 1, 7]))]
    return {
        "rbf": (lambda kp, A, B: rbf(A, A if B is None else B, kp["ls"]), ls_grid),
        "matern52": (lambda kp, A, B: matern52(A, A if B is None else B, kp["ls"]), ls_grid),
        "linear": (lambda kp, A, B: linear(A, A if B is None else B), lambda c: [{}]),
    }


def _est_grid(c: dict) -> list[dict]:
    return [{"alpha": float(a)} for a in np.logspace(*c.get("alpha_logspace", [-6, 0, 5]))]


def e3(cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Learning curves of classical kernel families + B0 (Δ = 0, i.e. the scaling law itself) on grouped folds.
    Equal budget: every family gets cfg.budget.hparam_evals_per_family evaluations, the same folds and seeds."""
    data = load_dataset(cfg)
    X, y, groups = data["X"], data["y"], data["groups"]
    ecfg = cfg.get("e3", {})
    families = ecfg.get("families", ["rbf", "matern52", "linear"])
    table = _family_table()
    budget = int(cfg["budget"]["hparam_evals_per_family"])
    k_outer, k_inner = cfg["splits"]["primary"]["k"], cfg["splits"]["inner"]["k"]
    repeats = cfg["seeds"][: cfg["splits"]["primary"].get("repeats", len(cfg["seeds"]))]
    n_grid = cfg["splits"]["learning_curve_N"]
    est = ecfg.get("estimator", "krr")
    outer_eval = cfg.get("eval", {}).get("outer", True)

    runs, rows = [], []
    for rep in repeats:
        folds = group_kfold(groups, k_outer, seed=rep)
        assert_disjoint_groups(folds, groups)
        for fi, (tr_all, te) in enumerate(folds):
            for n_req in n_grid:
                n = len(tr_all) if n_req == "max" else int(n_req)
                if n > len(tr_all):
                    continue
                tr = subsample(tr_all, n, seed=rep)
                inner = group_kfold(groups[tr], k_inner, seed=rep)
                lo, span = minmax_fit(X[tr])
                Xtr, Xte = minmax_apply(X[tr], lo, span), minmax_apply(X[te], lo, span)
                base = {"repeat": rep, "fold": fi, "N_label": str(n_req), "N": n, "n_test": len(te)}
                if outer_eval:
                    runs.append({**base, "model": "B0_zero", "status": "ok", **all_regression(y[te], np.zeros(len(te)))})
                    rows += [{**base, "model": "B0_zero", "id": data["ids"][j], "y": y[j], "pred": 0.0} for j in te]
                for fam in families:
                    kfn, gfn = table[fam]
                    rec = {**base, "model": fam}
                    try:
                        m, kp, ep, cvt = select_and_fit(kfn, gfn(ecfg.get(fam, {})), est, _est_grid(ecfg),
                                                        Xtr, y[tr], inner, budget=budget, seed=rep)
                        rec.update(status="ok", kernel=kp, est=ep, inner_cv_mae=min(r["cv_mae"] for r in cvt),
                                   n_evals=len(cvt))
                        if outer_eval:
                            p = m.predict(kfn(kp, Xte, Xtr))
                            rec.update(all_regression(y[te], p))
                            rows += [{**base, "model": fam, "id": data["ids"][j], "y": y[j], "pred": float(pj)}
                                     for j, pj in zip(te, p)]
                    except Exception as e:  # report failures, never drop them (CLAUDE.md §2)
                        rec.update(status="error", error=f"{type(e).__name__}: {e}")
                    runs.append(rec)
    return {"summary": summarise(runs), "runs": runs, "outer_eval": outer_eval,
            "data_source": cfg.get("data", {}).get("source")}, pd.DataFrame(rows)


def summarise(runs: list[dict]) -> list[dict]:
    """Mean / SD over (repeat, fold) per (model, requested N); 'max' keeps one row although N varies by fold.
    Counts failures."""
    df = pd.DataFrame(runs)
    metric = "mae" if "mae" in df else "inner_cv_mae"
    out = []
    for (model, label), g in df.groupby(["model", "N_label"], sort=False):
        ok = g[g["status"] == "ok"]
        out.append({"model": model, "N": label, "N_mean": float(g["N"].mean()), "metric": metric,
                    "J": int(len(ok)), "n_failed": int(len(g) - len(ok)),
                    "mean": float(ok[metric].mean()) if len(ok) else float("nan"),
                    "sd": float(ok[metric].std(ddof=1)) if len(ok) > 1 else float("nan")})
    return out


# ---------------------------------------------------------------- smoke (S001)

def smoke(cfg: dict) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(cfg.get("seed", 0))
    n, q = cfg.get("n", 80), cfg.get("q", 4)
    X = rng.uniform(-1, 1, (n, q))
    y = np.sin(2 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=n)
    tr, te = random_kfold(n, 5, 0)[0]
    inner = random_kfold(len(tr), 3, 1)
    qk = lambda kp, A, B: fidelity_kernel(FeatureMap("iqp", kp["L"], kp["bw"]), A, B)
    ck = lambda kp, A, B: rbf(A, A if B is None else B, kp["ls"])
    res, rows = {}, []
    for name, kfn, kgrid in [("qk_iqp", qk, [{"L": 1, "bw": b} for b in (0.1, 0.3, 1.0)]),
                             ("rbf", ck, [{"ls": l} for l in (0.3, 1.0, 3.0)])]:
        m, kp, ep, _ = select_and_fit(kfn, kgrid, "krr", [{"alpha": a} for a in (1e-3, 1e-1)],
                                      X[tr], y[tr], inner, budget=6)
        p = m.predict(kfn(kp, X[te], X[tr]))
        res[name] = {"kernel": kp, "est": ep, **all_regression(y[te], p)}
        rows += [{"model": name, "idx": int(j), "y": float(y[j]), "pred": float(pj)} for j, pj in zip(te, p)]
    return res, pd.DataFrame(rows)


KINDS: dict[str, Callable[[dict], tuple[dict, pd.DataFrame]]] = {"smoke": smoke, "e3": e3}


# ---------------------------------------------------------------- entry point

def run(cfg: dict, out_root: Path | None = None, argv: list[str] | None = None) -> Path:
    """Execute one resolved config and write the results folder. Returns its path."""
    kind = cfg["kind"]
    if kind not in KINDS:
        raise NotImplementedError(f"experiment kind {kind!r} is not implemented (known: {sorted(KINDS)})")
    check_prereg_guard(cfg)
    started = dt.datetime.now(dt.timezone.utc)
    sha, dirty = git_state()
    metrics, preds = KINDS[kind](cfg)
    ended = dt.datetime.now(dt.timezone.utc)
    out = make_run_dir(Path(out_root) if out_root else ROOT / "results", cfg["id"], sha, dirty, started)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    metrics = {"id": cfg["id"], "kind": kind, "git": sha, "dirty": dirty, "predictions": write_predictions(preds, out),
               **(metrics if kind != "smoke" else {"results": metrics})}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    (out / "env.txt").write_text(env_text(sha, dirty, argv or sys.argv, started, ended), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qkhosts.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override (repeatable)")
    ap.add_argument("--out-root", default=None, help="results root (default: <repo>/results)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out = run(cfg, a.out_root, ["python", "-m", "qkhosts.run", *(argv if argv is not None else sys.argv[1:])])
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for r in m.get("summary", []):
        print(f"{r['model']:>10s}  N={r['N']:<4s} {r['metric']}={r['mean']:.4f} +/- {r['sd']:.4f}  (J={r['J']}, failed={r['n_failed']})")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
