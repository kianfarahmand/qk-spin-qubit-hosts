#!/usr/bin/env python3
"""Descriptive breakdown of assembled E2 labels: Δ = log10(T2_CCE / T2_SL) by bath regime, spin class, stratum and
forced tail; direction spread; stretch exponent (S024, T-015). Descriptive only (no outer-test metrics).

    python scripts/e2_label_breakdown.py <results/E2/<run>/labels.csv> <request-id> [<request-id> ...]

Bath regime per host from the fetched records: "capped" if any ladder step of any direction hit the 1,500-spin cap,
"floored" if any step was floored, else "free". Strata and the forced tail come from the E2 draw
(results/T-014/20261031T184133Z_e1ef094/sample_adr0007.csv). Writes results/E2/<UTC>_<sha>/breakdown.json (+ env.txt).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qkhosts.labels.assemble import read_records  # noqa: E402

DRAW = ROOT / "results/T-014/20261031T184133Z_e1ef094/sample_adr0007.csv"


def regime(rec: dict) -> set[str]:
    out = set()
    for s in rec.get("sites", []):
        steps = s.get("ladder") or [st for d in s.get("directions", []) for st in d.get("ladder", [])]
        for st in steps:
            out.add("capped" if st.get("capped") else "floored" if st.get("floored") else "free")
    return out


def stats(x: pd.Series) -> dict:
    return {"n": int(len(x)), "median": float(x.median()), "p10": float(x.quantile(0.1)),
            "p90": float(x.quantile(0.9)), "min": float(x.min()), "max": float(x.max())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels")
    ap.add_argument("rids", nargs="+")
    a = ap.parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc)
    lab = pd.read_csv(a.labels)
    ok = lab[lab["status"] == "ok"].copy()
    reg: dict[str, set] = {}
    for rid in a.rids:
        for r in read_records(ROOT / "results" / "cce" / rid):
            if r.get("ok"):
                h = r["id"].split("-d")[0] if r["id"].startswith("mp-") else r["id"]
                reg.setdefault(h, set()).update(regime(r))
    ok["regime"] = ok["host"].map(lambda h: "capped" if "capped" in reg.get(h, ()) else
                                  "floored" if "floored" in reg.get(h, ()) else "free")
    draw = pd.read_csv(DRAW)[["mp_id", "sample_stratum", "forced_tail"]]
    ok = ok.merge(draw, left_on="host", right_on="mp_id", how="left")
    d = ok["delta_log10"]
    lg_cce, lg_sl = ok["log10_t2_ms"], np.log10(ok["t2_sl_s"] * 1e3)
    out = {
        "labels": a.labels, "requests": a.rids, "n_ok": int(len(ok)),
        "status": lab["status"].value_counts().to_dict(),
        "delta_all": stats(d), "share_delta_negative": float((d < 0).mean()),
        "by_regime": {k: stats(g["delta_log10"]) for k, g in ok.groupby("regime")},
        "by_cost_class": {k: stats(g["delta_log10"]) for k, g in ok.groupby("cost_class")},
        "by_stratum": {k: stats(g["delta_log10"]) for k, g in ok.groupby("sample_stratum")},
        "by_forced_tail": {str(k): stats(g["delta_log10"]) for k, g in ok.groupby("forced_tail")},
        "direction_spread_log10": stats(ok["direction_spread_log10"]),
        "stretch_n": stats(ok["stretch_n"]),
        "t2_ms_range": [float(ok["t2_ms"].min()), float(ok["t2_ms"].max())],
        "pearson_log10_cce_vs_sl": float(np.corrcoef(lg_cce, lg_sl)[0, 1]),
        "spearman_cce_vs_sl": float(pd.Series(lg_cce.values).corr(pd.Series(lg_sl.values), method="spearman")),
        "largest_direction_spread": ok.nlargest(5, "direction_spread_log10")[
            ["host", "material", "direction_spread_log10"]].to_dict("records"),
        "lowest_delta": ok.nsmallest(5, "delta_log10")[["host", "material", "delta_log10", "regime"]].to_dict("records"),
        "highest_delta": ok.nlargest(5, "delta_log10")[["host", "material", "delta_log10", "regime"]].to_dict("records"),
    }
    print(json.dumps(out, indent=1, default=str))
    from qkhosts.run import env_text, git_state, make_run_dir
    sha, dirty = git_state()
    ended = dt.datetime.now(dt.timezone.utc)
    run = make_run_dir(ROOT / "results", "E2", sha, dirty, ended)
    (run / "breakdown.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (run / "env.txt").write_text(env_text(sha, dirty, sys.argv, started, ended), encoding="utf-8")
    print(f"-> {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
