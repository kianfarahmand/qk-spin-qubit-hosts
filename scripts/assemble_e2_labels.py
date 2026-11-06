#!/usr/bin/env python3
"""Assemble D-CCE host labels from fetched E2 results (T-015, S020).

    python scripts/assemble_e2_labels.py <request-id> [<request-id> ...] [--out-root results]

Reads results/cce/<id>/request.yaml and merged.jsonl for every id (fetched with scripts/fetch_cce_results.py),
merges configuration chunks and directions per host (`qkhosts.labels.assemble`) and joins D-SL's T2_SL, so each
row also carries Δ = log10(T2_CCE / T2_SL). Writes results/E2/<UTC>_<sha>[-dirty]/: labels.csv (one row per host),
metrics.json (status counts, request ids, record counts), env.txt. Hosts whose items are spread over requests that
have not all been fetched come out as "missing" until they are.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkhosts.labels.assemble import assemble_labels, read_records  # noqa: E402
from qkhosts.run import env_text, git_state, make_run_dir  # noqa: E402


def load(rid: str) -> tuple[dict, list[dict]]:
    d = ROOT / "results" / "cce" / rid
    req = yaml.safe_load((d / "request.yaml").read_text(encoding="utf-8"))
    recs = read_records(d)
    return req, recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rids", nargs="+")
    ap.add_argument("--out-root", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc)
    reqs, recs = [], []
    for rid in a.rids:
        q, r = load(rid)
        reqs.append(q)
        recs += r
    rows = assemble_labels(reqs, recs)
    df = pd.DataFrame(rows)
    dsl = pd.read_csv(ROOT / "data/processed/dsl.csv.gz")[["mp_id", "material", "t2_sl_s", "cost_class"]]
    df = df.merge(dsl, left_on="host", right_on="mp_id", how="left").drop(columns="mp_id")
    df["delta_log10"] = df["log10_t2_ms"] - np.log10(df["t2_sl_s"] * 1e3)
    for c in ("direction_t2_ms", "n_conf_ok", "n_chunks", "problems", "superseded"):
        df[c] = df[c].map(json.dumps)
    ok = df[df["label_ok"]]
    m = {"requests": a.rids, "n_records": len(recs), "n_hosts": int(len(df)),
         "status": df["status"].value_counts().to_dict(),
         "delta_log10": ({"median": float(ok["delta_log10"].median()), "min": float(ok["delta_log10"].min()),
                          "max": float(ok["delta_log10"].max())} if len(ok) else None)}
    ended = dt.datetime.now(dt.timezone.utc)
    sha, dirty = git_state()
    out = make_run_dir(Path(a.out_root), "E2", sha, dirty, ended)
    df.to_csv(out / "labels.csv", index=False, float_format="%.6g")
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    (out / "env.txt").write_text(env_text(sha, dirty, sys.argv, started, ended), encoding="utf-8")
    print(f"-> {out}\n{json.dumps(m, indent=1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
