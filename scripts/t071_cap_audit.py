#!/usr/bin/env python3
"""T-071 cap audit: paired cap 3,000 vs cap 1,500 at B || c on the audit hosts (S025; ADR-0004 Amendment 4a).

    python scripts/t071_cap_audit.py [--audit 016-e2-capaudit01] [--base 013-e2-b01 017-e2-repair01]
                                     [--labels results/E2/<run>/labels.csv]

Per audit host, direction c (lattice [0, 0, 1]):
  * curve level: offset = log10 T2(cap 3,000) - log10 T2(cap 1,500), each T2 from the record's `final` step, or
    from `merge_chunks` over its chunks exactly as `qkhosts.labels.assemble` does for labels;
  * configuration level: both runs use seed 0, so configuration k has the same isotope draw in both; the per-
    configuration fits (`per_conf_t2_ms`, chunks concatenated in seed order) give paired log10 ratios, whose
    mean and SD per host show how uniform the offset is inside one host.
Escalation rule (ADR-0004 Amendment 4a, Q24a): if the host offsets of a class have SD > 0.02 dex, E2 escalates for
that class (floor on r_dipole or a larger cap). Times are in ms, offsets in dex (log10).
Writes results/T-071/<UTC>_<sha>/{metrics.json, pairs.csv, env.txt}.
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

DIR_C = (0, 0, 1)
SD_ESCALATE_DEX = 0.02
DRAW = ROOT / "results/T-014/20261031T184133Z_e1ef094/sample_adr0007.csv"


def load(rid: str) -> tuple[dict, list[dict]]:
    """Parsed request YAML and batch records of a fetched request."""
    d = ROOT / "results" / "cce" / rid
    return yaml.safe_load((d / "request.yaml").read_text(encoding="utf-8")), read_records(d)


def direction_items(req: dict, host: str, direction: tuple = DIR_C) -> list[dict]:
    """The items of one host and one field direction in a request."""
    return [m for m in req["materials"] if m["host"] == host
            and tuple(m.get("protocol", {}).get("b_directions_lattice", [[None]])[0]) == direction]


def conf_t2(items: list[dict], recs: dict, base_seed: int = 0) -> list[float]:
    """Per-configuration T2 (ms) of one host direction, chunks concatenated in configuration-seed order, so that
    entry k belongs to configuration seed base_seed + 1 + k in every chunking."""
    parts = []
    for it in items:
        r = recs[it["id"]]
        seed = int(it.get("protocol", {}).get("seed", base_seed))
        parts.append((seed, list(r["sites"][0]["final"]["per_conf_t2_ms"])))
    return [t for _, ts in sorted(parts) for t in ts]


def paired_conf_stats(t2_hi: list[float], t2_lo: list[float]) -> dict:
    """Paired per-configuration log10 ratios hi / lo (dex): n, mean, SD (ddof 1). Pairs with a non-finite or
    non-positive T2 on either side are dropped."""
    a, b = np.asarray(t2_hi, float), np.asarray(t2_lo, float)
    if a.shape != b.shape:
        raise ValueError(f"configuration counts differ: {a.size} vs {b.size}")
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    d = np.log10(a[ok] / b[ok])
    return {"n": int(d.size), "mean": float(d.mean()) if d.size else float("nan"),
            "sd": float(d.std(ddof=1)) if d.size > 1 else float("nan")}


def class_stats(x: pd.Series, sd_limit: float = SD_ESCALATE_DEX) -> dict:
    """Summary of host offsets (dex) for one class, with the Amendment 4a escalation flag (SD > sd_limit)."""
    x = x.dropna()
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": sd, "median": float(x.median()),
            "min": float(x.min()), "max": float(x.max()),
            "sem": sd / np.sqrt(len(x)) if len(x) > 1 else float("nan"),
            "escalate": bool(np.isfinite(sd) and sd > sd_limit)}


def final_geom(items: list[dict], recs: dict) -> dict:
    """Bath geometry of the first chunk's final step (identical across chunks: same cell, cap and radii)."""
    f = recs[items[0]["id"]]["sites"][0]["final"]
    return {k: f[k] for k in ("n_bath_in_rbath", "n_bath_in_rbath_nominal", "capped", "r_bath_used_A",
                               "r_dipole_used_A")}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", default="016-e2-capaudit01")
    ap.add_argument("--base", nargs="+", default=["013-e2-b01", "017-e2-repair01"])
    ap.add_argument("--labels", default=None, help="labels.csv for delta_log10 / cost_class (optional)")
    a = ap.parse_args(argv)
    started = dt.datetime.now(dt.timezone.utc)

    aq, arecs = load(a.audit)
    base = [load(r) for r in a.base]
    hosts = list(dict.fromkeys(m["host"] for m in aq["materials"]))
    hi_rows = {r["host"]: r for r in assemble_labels([aq], arecs, directions=(DIR_C,))}
    lo_rows = {r["host"]: r for r in assemble_labels([q for q, _ in base], [x for _, rs in base for x in rs],
                                                     directions=(DIR_C,)) if r["host"] in hosts}
    arec = {r["id"]: r for r in arecs}
    # later base requests supersede earlier ones for the same item id (repair rule)
    brec, bitems = {}, {}
    for q, rs in base:
        for r in rs:
            brec[r["id"]] = r
        for m in q["materials"]:
            if m["host"] in hosts:
                bitems[m["id"]] = m
    cap_hi = int(aq["protocol"]["max_bath_spins"])
    caps_lo = {int(q["protocol"]["max_bath_spins"]) for q, _ in base}
    if len(caps_lo) != 1:
        raise ValueError(f"base requests mix caps {caps_lo}")
    cap_lo = caps_lo.pop()

    rows = []
    for h in hosts:
        hi, lo = hi_rows[h], lo_rows.get(h)
        if lo is None or hi["status"] != "ok" or lo["status"] != "ok":
            rows.append({"host": h, "status": f"hi {hi['status']} / lo {lo['status'] if lo else 'absent'}"})
            continue
        it_hi = direction_items(aq, h)
        it_lo = [m for m in bitems.values() if m["host"] == h
                 and tuple(m["protocol"]["b_directions_lattice"][0]) == DIR_C]
        pc = paired_conf_stats(conf_t2(it_hi, arec), conf_t2(it_lo, brec))
        g_hi, g_lo = final_geom(it_hi, arec), final_geom(it_lo, brec)
        t_hi, t_lo = hi["direction_t2_ms"][0], lo["direction_t2_ms"][0]
        rows.append({
            "host": h, "status": "ok", "t2_ms_cap_hi": t_hi, "t2_ms_cap_lo": t_lo,
            "offset_dex": float(np.log10(t_hi / t_lo)),
            "offset_mof_dex": float(np.log10(hi["t2_ms_mean_of_fits"] / lo["t2_ms_mean_of_fits"])),
            "conf_n": pc["n"], "conf_mean_dex": pc["mean"], "conf_sd_dex": pc["sd"],
            "stretch_n_hi": hi["stretch_n"], "stretch_n_lo": lo["stretch_n"],
            "n_bath_nominal_50A": g_lo["n_bath_in_rbath_nominal"],
            "n_bath_hi": g_hi["n_bath_in_rbath"], "n_bath_lo": g_lo["n_bath_in_rbath"],
            "capped_hi": g_hi["capped"], "capped_lo": g_lo["capped"],
            "r_bath_hi_A": g_hi["r_bath_used_A"], "r_bath_lo_A": g_lo["r_bath_used_A"],
            "r_dipole_hi_A": g_hi["r_dipole_used_A"], "r_dipole_lo_A": g_lo["r_dipole_used_A"],
            "n_chunks_hi": len(it_hi), "n_chunks_lo": len(it_lo),
            "wall_h_hi": hi["wall_s"] / 3600, "wall_h_lo": lo["wall_s"] / 3600,
        })
    df = pd.DataFrame(rows)
    draw = pd.read_csv(DRAW)[["mp_id", "material", "sample_stratum", "cost_class", "forced_tail"]]
    df = df.merge(draw, left_on="host", right_on="mp_id", how="left").drop(columns="mp_id")
    if a.labels:
        lab = pd.read_csv(a.labels)[["host", "delta_log10", "direction_spread_log10"]]
        df = df.merge(lab, on="host", how="left")
    ok = df[df["status"] == "ok"].copy()
    ok["cost_ratio_hi_lo"] = ok["wall_h_hi"] / ok["wall_h_lo"]

    out = {
        "audit": a.audit, "base": a.base, "cap_hi": cap_hi, "cap_lo": cap_lo, "direction_lattice": list(DIR_C),
        "sd_escalate_dex": SD_ESCALATE_DEX, "n_hosts": int(len(df)), "n_ok": int(len(ok)),
        "not_ok": df.loc[df["status"] != "ok", ["host", "status"]].to_dict("records"),
        "all": class_stats(ok["offset_dex"]),
        "all_mean_of_fits": class_stats(ok["offset_mof_dex"]),
        "by_stratum": {k: class_stats(g["offset_dex"]) for k, g in ok.groupby("sample_stratum")},
        "by_cost_class": {k: class_stats(g["offset_dex"]) for k, g in ok.groupby("cost_class")},
        "conf_level": {"median_within_host_sd_dex": float(ok["conf_sd_dex"].median()),
                       "max_within_host_sd_dex": float(ok["conf_sd_dex"].max()),
                       "max_abs_conf_mean_minus_curve_dex":
                           float((ok["conf_mean_dex"] - ok["offset_dex"]).abs().max())},
        "share_offset_negative": float((ok["offset_dex"] < 0).mean()),
        "share_hi_still_capped": float(ok["capped_hi"].mean()),
        "spearman_offset_vs_nominal_bath": float(ok["offset_dex"].corr(np.log10(ok["n_bath_nominal_50A"]),
                                                                        method="spearman")),
        "spearman_offset_vs_r_dipole_lo": float(ok["offset_dex"].corr(ok["r_dipole_lo_A"], method="spearman")),
        "stretch_n_change": {"median": float((ok["stretch_n_hi"] - ok["stretch_n_lo"]).median()),
                             "max_abs": float((ok["stretch_n_hi"] - ok["stretch_n_lo"]).abs().max())},
        "cost_ratio_hi_lo": {"median": float(ok["cost_ratio_hi_lo"].median()),
                             "min": float(ok["cost_ratio_hi_lo"].min()), "max": float(ok["cost_ratio_hi_lo"].max()),
                             "total": float(ok["wall_h_hi"].sum() / ok["wall_h_lo"].sum())},
        "most_negative": ok.nsmallest(5, "offset_dex")[["host", "material", "offset_dex", "conf_sd_dex",
                                                        "r_dipole_lo_A", "n_bath_nominal_50A"]].to_dict("records"),
        "most_positive": ok.nlargest(3, "offset_dex")[["host", "material", "offset_dex", "conf_sd_dex",
                                                       "r_dipole_lo_A", "n_bath_nominal_50A"]].to_dict("records"),
    }
    if "delta_log10" in ok:
        out["delta_sl_of_audit_hosts"] = {"median": float(ok["delta_log10"].median()),
                                          "median_after_offset": float((ok["delta_log10"] + ok["offset_dex"])
                                                                       .median())}
    out["escalate_any"] = bool(out["all"]["escalate"] or any(
        v["escalate"] for grp in ("by_stratum", "by_cost_class") for v in out[grp].values()))
    print(json.dumps(out, indent=1, default=str))

    from qkhosts.run import env_text, git_state, make_run_dir
    sha, dirty = git_state()
    ended = dt.datetime.now(dt.timezone.utc)
    run = make_run_dir(ROOT / "results", "T-071", sha, dirty, ended)
    (run / "metrics.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    df.to_csv(run / "pairs.csv", index=False)
    (run / "env.txt").write_text(env_text(sha, dirty, sys.argv, started, ended), encoding="utf-8")
    print(f"-> {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
