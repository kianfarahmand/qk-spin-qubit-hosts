#!/usr/bin/env python3
"""Check a fetched E2 request against the pilot pass criteria of experiments/E2_notes.md §6 (S022, T-015).

    python scripts/check_e2_pilot.py <request-id> [--write]

Reads results/cce/<id>/{request.yaml, merged.jsonl} (scripts/fetch_cce_results.py) and reports, per item: ok,
label_ok, the lattice the runner used (`cell_A`: a, b, c in Å, angles in degrees, crystal family guess), the windows
tried (first window accepted = one window), measured wall hours against `pred_h`. Benchmarks: the host log-mean
over directions against earlier labels, recomputed from their records (010 CaS, 008 CeO2, 009 CaO; log-mean over the
three ADR-0004 directions). With --write the report goes to results/T-015/<UTC>_<sha>/pilot_check.json (+ env.txt).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# host -> (earlier request, item ids whose log-mean is the earlier label)
BENCH = {"mp-1672": ("010-e2-cost", ["cas-c", "cas-ab", "cas-abc"]),
         "mp-20194": ("008-e0b-dilute", ["ceo2-A"]),
         "mp-2605": ("009-e0b-cao", ["cao-c", "cao-ab", "cao-abc"])}
BENCH_TOL_DEX = 0.05
COST_FACTOR = 2.0


def records(rid: str) -> list[dict]:
    p = ROOT / "results" / "cce" / rid / "merged.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def lattice(cell_a: list) -> dict:
    """a, b, c (Å) and alpha, beta, gamma (deg) of a 3x3 cell given as columns (pycce BathCell.cell)."""
    m = np.asarray(cell_a, float).T  # rows = lattice vectors
    a, b, c = np.linalg.norm(m, axis=1)

    def ang(u, v):
        return float(np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))))

    out = {"a": float(a), "b": float(b), "c": float(c),
           "alpha": ang(m[1], m[2]), "beta": ang(m[0], m[2]), "gamma": ang(m[0], m[1])}
    eq = lambda x, y, tol: abs(x - y) <= tol  # noqa: E731
    right = all(eq(out[k], 90, 0.1) for k in ("alpha", "beta", "gamma"))
    if right and eq(a, b, 1e-3 * a) and eq(b, c, 1e-3 * b):
        fam = "cubic"
    elif right and eq(a, b, 1e-3 * a):
        fam = "tetragonal"
    elif eq(out["alpha"], 90, 0.1) and eq(out["beta"], 90, 0.1) and eq(out["gamma"], 120, 0.1) and eq(a, b, 1e-3 * a):
        fam = "hexagonal"
    elif right:
        fam = "orthorhombic"
    else:
        fam = "other"
    out["family"] = fam
    return out


def log_mean(x: list[float]) -> float:
    x = np.asarray(x, float)
    return float(10 ** np.mean(np.log10(x))) if len(x) and np.all(np.isfinite(x) & (x > 0)) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rid")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    started = dt.datetime.now(dt.timezone.utc)
    req = yaml.safe_load((ROOT / "results" / "cce" / a.rid / "request.yaml").read_text(encoding="utf-8"))
    recs = {r["id"]: r for r in records(a.rid)}
    items, hosts = [], {}
    for it in req["materials"]:
        r = recs.get(it["id"])
        row = {"id": it["id"], "host": it.get("host", it["id"]), "pred_h": it.get("pred_h"), "present": r is not None}
        if r is not None:
            tried = [s["final"].get("windows_tried_ms") for s in r.get("sites", [])]
            row.update(ok=bool(r.get("ok")), label_ok=bool(r.get("label_ok")), t2_ms=r.get("t2_ms"),
                       wall_h=r.get("wall_s", float("nan")) / 3600, n_conf_used=r.get("n_conf_used"),
                       windows_tried_ms=tried, first_window=all(t is not None and len(t) == 1 for t in tried),
                       t_max_ms_prior=it["protocol"].get("t_max_ms"), error=r.get("error"),
                       lattice=lattice(r["cell_A"]) if "cell_A" in r else None)
            row["cost_ratio"] = row["wall_h"] / it["pred_h"] if it.get("pred_h") else None
            if r.get("label_ok"):
                hosts.setdefault(row["host"], []).append(r["t2_ms"])
        items.append(row)
    bench = {}
    for h, (rid, ids) in BENCH.items():
        if h not in hosts:
            continue
        old = {r["id"]: r for r in records(rid)}
        prev = log_mean([old[i]["t2_ms"] for i in ids])
        new = log_mean(hosts[h])
        d = float(np.log10(new / prev))
        bench[h] = {"earlier": f"{rid}:{'+'.join(ids)}", "earlier_ms": prev, "pilot_ms": new,
                    "n_dirs_pilot": len(hosts[h]), "delta_dex": d, "pass": abs(d) <= BENCH_TOL_DEX}
    present = [x for x in items if x["present"]]
    ratios = [x["cost_ratio"] for x in present if x.get("cost_ratio")]
    summary = {
        "request": a.rid, "items": len(items), "records": len(present),
        "all_ok": all(x.get("ok") and x.get("label_ok") for x in present) and len(present) == len(items),
        "not_ok": [x["id"] for x in items if not (x.get("ok") and x.get("label_ok"))],
        "first_window_share": float(np.mean([x["first_window"] for x in present])) if present else None,
        "cost_ratio_median": float(np.median(ratios)) if ratios else None,
        "cost_ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
        "cost_outside_x2": [x["id"] for x in present if x.get("cost_ratio")
                            and not (1 / COST_FACTOR <= x["cost_ratio"] <= COST_FACTOR)],
        "runner_h_measured": float(sum(x["wall_h"] for x in present)),
        "runner_h_pred": float(sum(x["pred_h"] or 0 for x in items)),
        "families": {h: sorted({x["lattice"]["family"] for x in present if x["host"] == h and x.get("lattice")})
                     for h in sorted({x["host"] for x in items})},
        "bench": bench}
    print(json.dumps(summary, indent=1, default=str))
    for x in present:
        lt = x.get("lattice") or {}
        print(f"{x['id']:<18} ok={x.get('ok')!s:<5} T2={x.get('t2_ms') or float('nan'):9.4g} ms  "
              f"wall {x['wall_h']:6.3f} h / pred {x['pred_h'] or float('nan'):6.3f}  win {x['windows_tried_ms']}  "
              f"{lt.get('family', '?')} a={lt.get('a', 0):.3f} b={lt.get('b', 0):.3f} c={lt.get('c', 0):.3f} "
              f"γ={lt.get('gamma', 0):.1f}")
    if a.write:
        from qkhosts.run import env_text, git_state, make_run_dir
        sha, dirty = git_state()
        ended = dt.datetime.now(dt.timezone.utc)
        out = make_run_dir(ROOT / "results", "T-015", sha, dirty, ended)
        (out / "pilot_check.json").write_text(json.dumps({"summary": summary, "items": items}, indent=1,
                                                         default=str), encoding="utf-8")
        (out / "env.txt").write_text(env_text(sha, dirty, sys.argv, started, ended), encoding="utf-8")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
