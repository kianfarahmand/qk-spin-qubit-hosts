#!/usr/bin/env python3
"""Check a fetched E2 batch: failures, missing items, timeouts, window prior, cost against pred_h (S024, T-015).

    python scripts/check_e2_batch.py <request-id> [--jobs jobs.json] [--write]

Builds on scripts/check_e2_pilot.py (same per-item rows) and adds: failures grouped by error, items with no record
grouped by shard (a lost or timed-out job writes none), measured runner-h per shard (sum of record walls) and, with
--jobs (`gh run view <run> --json jobs`), each job's wall clock and the 6 h limit; cost ratio wall/pred_h of ok items
overall and per spin class (`cost_class` of data/processed/dsl.csv.gz); a least-squares fit wall = f * pred_h + c0
(c0 = per-item overhead, h); items whose window prior was not accepted at once. With --write the report goes to
results/T-015/<UTC>_<sha>/batch_check.json (+ env.txt).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import check_e2_pilot as cp  # noqa: E402

JOB_LIMIT_H = 6.0


def item_rows(rid: str) -> list[dict]:
    """Per-item rows as in check_e2_pilot (plus the shard index)."""
    req = yaml.safe_load((ROOT / "results" / "cce" / rid / "request.yaml").read_text(encoding="utf-8"))
    recs = {r["id"]: r for r in cp.records(rid)}
    items = []
    for it in req["materials"]:
        r = recs.get(it["id"])
        row = {"id": it["id"], "host": it.get("host", it["id"]), "shard": it.get("shard"), "pred_h": it.get("pred_h"),
               "present": r is not None}
        if r is not None:
            tried = [s["final"].get("windows_tried_ms") for s in r.get("sites", [])]
            row.update(ok=bool(r.get("ok")), label_ok=bool(r.get("label_ok")), t2_ms=r.get("t2_ms"),
                       wall_h=r.get("wall_s", float("nan")) / 3600, windows_tried_ms=tried,
                       first_window=all(t is not None and len(t) == 1 for t in tried),
                       t_max_ms_prior=it["protocol"].get("t_max_ms"), error=r.get("error"))
            row["cost_ratio"] = row["wall_h"] / it["pred_h"] if it.get("pred_h") else None
        items.append(row)
    return items


def job_walls(path: Path) -> dict:
    """shard index -> {conclusion, wall_h} from `gh run view --json jobs` (job names 'run (<i>)')."""
    jobs = json.loads(path.read_text(encoding="utf-8"))["jobs"]
    out = {}
    for j in jobs:
        name = j.get("name", "")
        if not name.startswith("run ("):
            continue
        i = int(name[5:-1])
        t = [dt.datetime.fromisoformat(j[k].replace("Z", "+00:00")) if j.get(k) else None
             for k in ("startedAt", "completedAt")]
        out[i] = {"conclusion": j.get("conclusion"),
                  "wall_h": (t[1] - t[0]).total_seconds() / 3600 if all(t) else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rid")
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    started = dt.datetime.now(dt.timezone.utc)
    items = item_rows(a.rid)
    present = [x for x in items if x["present"]]
    ok = [x for x in present if x.get("ok") and x.get("label_ok")]
    failed = [x for x in present if not (x.get("ok") and x.get("label_ok"))]
    missing = [x for x in items if not x["present"]]
    dsl = pd.read_csv(ROOT / "data" / "processed" / "dsl.csv.gz").set_index("mp_id")
    cls = {h: dsl.loc[h, "cost_class"] if h in dsl.index else "?" for h in {x["host"] for x in items}}

    by_err = defaultdict(set)
    for x in failed:
        by_err[(x.get("error") or "label not ok").split("\n")[0][:120]].add(x["host"])
    miss_by_shard = defaultdict(list)
    for x in missing:
        miss_by_shard[x["shard"]].append(x["id"])

    shard_pred, shard_meas = defaultdict(float), defaultdict(float)
    for x in items:
        shard_pred[x["shard"]] += x["pred_h"] or 0.0
        if x["present"]:
            shard_meas[x["shard"]] += x["wall_h"]
    jobs = job_walls(a.jobs) if a.jobs else {}

    r = np.array([x["cost_ratio"] for x in ok if x.get("cost_ratio")], float)
    p = np.array([x["pred_h"] for x in ok if x.get("cost_ratio")], float)
    w = np.array([x["wall_h"] for x in ok if x.get("cost_ratio")], float)
    f, c0 = np.linalg.lstsq(np.c_[p, np.ones_like(p)], w, rcond=None)[0]
    per_class = {}
    for c in sorted(set(cls.values())):
        sel = [x for x in ok if cls[x["host"]] == c and x.get("cost_ratio")]
        if sel:
            per_class[c] = {"items": len(sel), "pred_h": float(sum(x["pred_h"] for x in sel)),
                            "meas_h": float(sum(x["wall_h"] for x in sel)),
                            "ratio_median": float(np.median([x["cost_ratio"] for x in sel]))}
    big = p >= 0.5
    per_spin = {}  # items predicted >= 0.5 h, by the host's largest nuclear spin I_max (dsl)
    for s in sorted({float(dsl.loc[x["host"], "I_max"]) for x in ok if x["host"] in dsl.index}):
        sel = [x for x in ok if x.get("cost_ratio") and x["pred_h"] >= 0.5 and x["host"] in dsl.index
               and float(dsl.loc[x["host"], "I_max"]) == s]
        if sel:
            per_spin[str(s)] = {"items": len(sel), "pred_h": float(sum(x["pred_h"] for x in sel)),
                                "ratio_total": float(sum(x["wall_h"] for x in sel) / sum(x["pred_h"] for x in sel)),
                                "ratio_median": float(np.median([x["cost_ratio"] for x in sel])),
                                "ratio_max": float(max(x["cost_ratio"] for x in sel))}
    summary = {
        "request": a.rid, "items": len(items), "records": len(present), "ok_items": len(ok),
        "hosts": len({x["host"] for x in items}),
        "hosts_all_ok": len({x["host"] for x in items} - {x["host"] for x in failed + missing}),
        "failed_by_error": {k: sorted(v) for k, v in by_err.items()},
        "missing_by_shard": dict(miss_by_shard),
        "first_window_share": float(np.mean([x["first_window"] for x in ok])),
        "not_first_window": [{"id": x["id"], "prior_ms": x["t_max_ms_prior"], "tried_ms": x["windows_tried_ms"]}
                             for x in ok if not x["first_window"]],
        "runner_h_pred_all": float(sum(x["pred_h"] or 0 for x in items)),
        "runner_h_measured_records": float(sum(x["wall_h"] for x in present)),
        "runner_h_pred_ok": float(p.sum()), "runner_h_meas_ok": float(w.sum()),
        "cost_ratio_ok_total": float(w.sum() / p.sum()),
        "cost_ratio_median": float(np.median(r)), "cost_ratio_p10_p90": [float(np.percentile(r, 10)),
                                                                         float(np.percentile(r, 90))],
        "cost_ratio_items_pred_ge_0.5h": {"n": int(big.sum()), "total": float(w[big].sum() / p[big].sum()),
                                          "median": float(np.median(r[big])) if big.any() else None,
                                          "max": float(r[big].max()) if big.any() else None},
        "fit_wall_eq_f_pred_plus_c0": {"f": float(f), "c0_h": float(c0)},
        "per_class": per_class,
        "per_I_max_items_pred_ge_0.5h": per_spin,
        "shard_meas_h_max": float(max(shard_meas.values())),
        "shard_pred_h_max": float(max(shard_pred.values())),
        "shards_meas_over_5h": sorted(k for k, v in shard_meas.items() if v > 5.0),
    }
    if jobs:
        jw = [v["wall_h"] for v in jobs.values() if v["wall_h"] is not None]
        summary.update(jobs=len(jobs), jobs_conclusions=dict(Counter(v["conclusion"] for v in jobs.values())),
                       job_wall_h_max=float(max(jw)), job_wall_h_median=float(np.median(jw)),
                       jobs_over_limit=sorted(i for i, v in jobs.items() if (v["wall_h"] or 0) >= JOB_LIMIT_H - 0.05),
                       jobs_not_success={i: v for i, v in jobs.items() if v["conclusion"] != "success"})
    print(json.dumps(summary, indent=1, default=str))
    if a.write:
        from qkhosts.run import env_text, git_state, make_run_dir
        sha, dirty = git_state()
        ended = dt.datetime.now(dt.timezone.utc)
        out = make_run_dir(ROOT / "results", "T-015", sha, dirty, ended)
        shards = {k: {"pred_h": shard_pred[k], "meas_h": shard_meas.get(k, 0.0), **(jobs.get(k, {}))}
                  for k in sorted(shard_pred)}
        (out / "batch_check.json").write_text(json.dumps({"summary": summary, "shards": shards, "items": items},
                                                         indent=1, default=str), encoding="utf-8")
        (out / "env.txt").write_text(env_text(sha, dirty, sys.argv, started, ended), encoding="utf-8")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
