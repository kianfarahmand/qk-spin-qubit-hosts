"""Experiment runner entry point:  python -m qkhosts.run --config configs/experiments/E#.yaml

Implemented now: `smoke` (synthetic end-to-end check of kernels -> selection -> metrics -> stats).
Real experiments are wired up in T-030.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import yaml

from qkhosts.data.splits import random_kfold
from qkhosts.eval.metrics import all_regression
from qkhosts.kernels.classical import rbf
from qkhosts.kernels.feature_maps import FeatureMap
from qkhosts.kernels.quantum import fidelity_kernel
from qkhosts.models.kernel_models import select_and_fit

ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def smoke(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg.get("seed", 0))
    n, q = cfg.get("n", 80), cfg.get("q", 4)
    X = rng.uniform(-1, 1, (n, q))
    y = np.sin(2 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=n)
    tr, te = random_kfold(n, 5, 0)[0]
    inner = random_kfold(len(tr), 3, 1)
    qk = lambda kp, A, B: fidelity_kernel(FeatureMap("iqp", kp["L"], kp["bw"]), A, B)
    ck = lambda kp, A, B: rbf(A, A if B is None else B, kp["ls"])
    res = {}
    for name, kfn, kgrid in [("qk_iqp", qk, [{"L": 1, "bw": b} for b in (0.1, 0.3, 1.0)]),
                             ("rbf", ck, [{"ls": l} for l in (0.3, 1.0, 3.0)])]:
        m, kp, ep, _ = select_and_fit(kfn, kgrid, "krr", [{"alpha": a} for a in (1e-3, 1e-1)],
                                      X[tr], y[tr], inner, budget=6)
        p = m.predict(kfn(kp, X[te], X[tr]))
        res[name] = {"kernel": kp, "est": ep, **all_regression(y[te], p)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    eid, kind = cfg["id"], cfg["kind"]
    if kind != "smoke":
        raise NotImplementedError(f"experiment kind {kind!r}: implement in T-030")
    out = ROOT / "results" / eid / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{_git_sha()}"
    out.mkdir(parents=True, exist_ok=True)
    metrics = smoke(cfg)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    (out / "env.txt").write_text(f"python {platform.python_version()}\nnumpy {np.__version__}\n")
    print(json.dumps(metrics, indent=1))
    print("->", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
