#!/usr/bin/env python3
"""Write the E2 (T-015) compute requests for a host sample, packed by the CCE cost model (S020).

    python scripts/make_e2_request.py --cost results/T-070/<run>/metrics.json [--sample ...] [--n-conf 20]
        [--cap 1500] [--min-r-dipole A] [--target-h 2.5] [--max-jobs 256] [--first 12] [--write]

Without --write it prints the plan only. With --write it writes compute/drafts/<NNN>-e2-bKK.yaml (drafts: a launch is
`git mv` to compute/requests/ with Kian's OK, one request per push) and the plan to results/T-015/<UTC>_<sha>/
(plan.csv: one row per item with its predicted hours; metrics.json; env.txt).

Plan:
1. Items: one per host x ADR-0004 direction (c, a+b, a+b+c), protocol = ADR-0004 + `cell_setting:
   conventional_standard` (CIFs are MP cells, `scripts/make_e2_structures.py`) + the chosen n_conf / cap / r_dipole
   floor. Predicted hours per direction = (configurations + extra windows) x t_eval from the cost model
   (`qkhosts.labels.cost.host_eval_seconds`, parameters from a T-070 run), 1 configuration for ~100 %-abundant baths.
2. Chunks: a direction predicted above --target-h is split into configuration chunks (`chunk_plan`; seeds 0, m, 2m,
   ...), each paying its own window search; `qkhosts.labels.assemble` merges them. A chunk of one configuration
   predicted above the job limit is reported as infeasible (the host stays in the plan, flagged).
3. Order: hosts round-robin over the strata (S1, S2, S3, S4), cheapest first within each stratum, so every finished
   request adds a balanced set of labels.
4. Requests: consecutive hosts fill a request until its first-fit-decreasing packing into shards of --target-h would
   exceed --max-jobs shards (Actions matrix limit 256). Each item gets an explicit `shard` (batch `shard_items`);
   heavier shards come first.
--target-h 2.5 leaves a factor of about 2 (the model's uncertainty) below the 5 h 55 min job limit.

--window-prior K (S020, proposed in ADR-0004 Amendment 4): every item starts its window search at t_max_ms =
K x T2_SL instead of 50 ms. The search itself is unchanged (it still shrinks or widens when the first window does
not resolve the decay), so a wrong prior costs what the 50 ms start costs; a right one saves the 1-2 extra window
evaluations of every item and every chunk. K = 7 accepts the first window (L(t_max) <= 0.05 and >= 15 points with
0.05 < L < 0.95) for T2 / T2_SL in [0.41, 4.51] at n = 2.5 ([0.24, 3.19] at n = 1.4, [0.45, 4.72] at n = 2.8);
the 21 directions of 010 (ratios 0.95-3.3, n 1.4-2.8) all pass at K = 6, 7 and 8 (S020, experiments/E2_notes.md
§4). The plan then assumes no extra window.

--n-conf-floored M (S021, ADR-0004 Amendment 4(e)): hosts whose bath inside 50 Å is below the floor (1,000 spins; the
floor grows r_bath) get M configurations instead of --n-conf; their ensembles are wider (010: CaS log10 SEM 0.027-0.034
dex at 10). Dense hosts keep 1. A host whose n_conf differs from the request's carries it in its item protocol.

--hosts / --exclude (S021, Q24e pilot): restrict the sample to the listed mp-ids, or drop them (comma-separated, or
@file with one id per line). --name sets the request id stem (default `e2-b`: <NNN>-e2-b01, ...). The pilot is
`--hosts <ids> --name e2-pilot`; the batches then run with `--exclude <same ids>`, so no host is labelled twice.

--directions / --audit N (S022, T-071, ADR-0004 Amendment 4(a)): --directions keeps only these ADR-0004 direction
indices (0 = c, 1 = a+b, 2 = a+b+c; item ids keep their d<j>). --audit N draws N hosts (split equally over the strata,
seeded by --audit-seed) from the capped non-dense hosts of the selection: nominal bath inside 50 Å above the E2 cap
(BATCH_CAP = 1,500) and dense share < 0.99. The cap audit is
`--hosts @<batch-1 hosts> --audit 24 --cap 3000 --directions 0 --n-conf 10 --window-prior 7 --name e2-capaudit`;
it keeps seed 0 and n_conf 10, so its configurations pair one to one with the batch items (the same supercell and
configuration seeds; only the cap differs).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkhosts.labels.chunks import chunk_plan  # noqa: E402
from qkhosts.labels.cost import bath_mix, extra_windows, host_eval_seconds, isotopes_from_table  # noqa: E402
from qkhosts.physics.isotopes import load_kanai_isotope_table  # noqa: E402

SAMPLE = "results/T-014/20261031T184133Z_e1ef094/sample_adr0007.csv"
DIRECTIONS = ((0, 0, 1), (1, 1, 0), (1, 1, 1))
STRATA = ("S1", "S2", "S3", "S4")
FLOOR = 1000
JOB_LIMIT_H = 5 + 55 / 60
DENSE_SHARE = 0.99
BATCH_CAP = 1500


def parse_ids(arg: str | None) -> list[str]:
    """mp-ids from 'a,b,c' or '@file' (one per line, # comments); [] for None."""
    if not arg:
        return []
    if arg.startswith("@"):
        lines = (ROOT / arg[1:]).read_text(encoding="utf-8").splitlines()
        return [x.split("#")[0].strip() for x in lines if x.split("#")[0].strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def select_hosts(smp: pd.DataFrame, hosts: list[str], exclude: list[str]) -> pd.DataFrame:
    """Rows of the sample for `hosts` (all if empty) minus `exclude`; unknown ids are an error."""
    known = set(smp["mp_id"])
    bad = [h for h in hosts + exclude if h not in known]
    if bad:
        raise SystemExit(f"not in the sample: {bad}")
    out = smp[smp["mp_id"].isin(hosts)] if hosts else smp
    return out[~out["mp_id"].isin(exclude)].reset_index(drop=True)


def parse_directions(arg: str | None) -> list[int]:
    """Direction indices into DIRECTIONS from '0,2'; all for None."""
    if not arg:
        return list(range(len(DIRECTIONS)))
    out = sorted({int(x) for x in arg.split(",") if x.strip()})
    if not out or any(j not in range(len(DIRECTIONS)) for j in out):
        raise SystemExit(f"--directions must be indices in 0..{len(DIRECTIONS) - 1}: {arg}")
    return out


def audit_draw(hosts: pd.DataFrame, n: int, seed: int = 0, cap: int = BATCH_CAP) -> list[str]:
    """mp-ids of an n-host audit subsample: capped (n_bath_50A > cap) non-dense hosts, n split as equally as
    possible over STRATA (remainder to the first strata), drawn without replacement with numpy seed `seed`;
    a stratum with too few capped hosts gives all it has. Sorted."""
    pool = hosts[(hosts["n_bath_50A"] > cap) & (hosts["dense_share"] < DENSE_SHARE)]
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for k, s in enumerate(STRATA):
        ids = sorted(pool.loc[pool["sample_stratum"] == s, "mp_id"])
        m = min(len(ids), n // len(STRATA) + (k < n % len(STRATA)))
        out += list(rng.choice(ids, size=m, replace=False)) if m else []
    return sorted(out)


def host_n_conf(dense: bool, n_bath_50A: float, n_conf: int, n_conf_floored: int | None = None) -> int:
    """Configurations for one host: 1 for ~100 %-abundant (dense) baths, `n_conf_floored` (if given) when the
    expected bath inside 50 Å is below the floor (FLOOR spins), else `n_conf`."""
    if dense:
        return 1
    if n_conf_floored is not None and n_bath_50A < FLOOR:
        return int(n_conf_floored)
    return int(n_conf)


def window_prior_ms(t2_sl_s: float, k: float) -> float:
    """Initial window (ms) = k x T2_SL, to 4 significant figures (T2_SL in s)."""
    return float(f"{k * t2_sl_s * 1e3:.4g}")


def plan_host(t_eval_s: float, n_conf: int, extra: int, target_h: float) -> list[tuple[int, int, float]]:
    """[(seed, n_conf, predicted h)] for one direction: one item if (n_conf + extra) evaluations fit target_h,
    otherwise the fewest equal chunks that do (each chunk pays `extra` window evaluations); at least 1 per chunk."""
    h_eval = t_eval_s / 3600.0
    if (n_conf + extra) * h_eval <= target_h or n_conf == 1:
        return [(0, n_conf, (n_conf + extra) * h_eval)]
    per = max(1, int(math.floor(target_h / h_eval)) - extra)
    n_chunks = math.ceil(n_conf / per)
    per = math.ceil(n_conf / n_chunks)  # balance the chunks
    return [(s, n, (n + extra) * h_eval) for s, n in chunk_plan(n_conf, per)]


def round_robin(hosts: pd.DataFrame, key: str = "material_h") -> list[str]:
    """mp-ids interleaved over STRATA (S1, S2, S3, S4, S1, ...), ascending `key` within each stratum."""
    queues = {s: list(hosts[hosts["sample_stratum"] == s].sort_values([key, "mp_id"])["mp_id"]) for s in STRATA}
    other = sorted(set(hosts["mp_id"]) - {m for q in queues.values() for m in q})
    if other:
        raise ValueError(f"hosts outside {STRATA}: {other[:3]}")
    out = []
    while any(queues.values()):
        for s in STRATA:
            if queues[s]:
                out.append(queues[s].pop(0))
    return out


def pack(hours: list[float], cap_h: float) -> list[list[int]]:
    """First-fit decreasing: item indices per bin of capacity cap_h (an item above cap_h gets its own bin).
    Bins come out heaviest first."""
    bins: list[list[int]] = []
    load: list[float] = []
    for i in sorted(range(len(hours)), key=lambda i: -hours[i]):
        for b in range(len(bins)):
            if load[b] + hours[i] <= cap_h + 1e-12:
                bins[b].append(i)
                load[b] += hours[i]
                break
        else:
            bins.append([i])
            load.append(hours[i])
    order = sorted(range(len(bins)), key=lambda b: -load[b])
    return [bins[b] for b in order]


def makespan(loads: list[float], n_parallel: int = 20) -> float:
    """Wall hours for shards started in list order on n_parallel runners (each takes the next shard when free)."""
    free = [0.0] * n_parallel
    for x in loads:
        i = int(np.argmin(free))
        free[i] += x
    return max(free) if loads else 0.0


def build(hosts: pd.DataFrame, items_by_host: dict[str, list[dict]], order: list[str], target_h: float,
          max_jobs: int) -> list[dict]:
    """Requests as {hosts, items (with shard), loads}: consecutive hosts in `order` while the packing fits max_jobs."""
    reqs, cur = [], []

    def packed(hs):
        its = [it for h in hs for it in items_by_host[h]]
        return its, pack([it["pred_h"] for it in its], target_h)

    for h in order:
        its, bins = packed(cur + [h])
        if len(bins) > max_jobs and cur:
            reqs.append(cur)
            cur = [h]
        else:
            cur.append(h)
    if cur:
        reqs.append(cur)
    out = []
    for hs in reqs:
        its, bins = packed(hs)
        if len(bins) > max_jobs:
            raise ValueError(f"host {hs[0]} alone needs {len(bins)} shards > {max_jobs}")
        its = [dict(it) for it in its]
        for s, b in enumerate(bins):
            for i in b:
                its[i]["shard"] = s
        loads = [sum(its[i]["pred_h"] for i in b) for b in bins]
        out.append({"hosts": hs, "items": sorted(its, key=lambda it: (it["shard"], it["id"])), "loads": loads})
    return out


def request_yaml(rid: str, protocol: dict, req: dict, header: list[str]) -> str:
    lines = [f"# {x}" if x else "#" for x in header]
    lines += [f"id: {rid}", "mode: cce", f"nshards: {len(req['loads'])}", "protocol:"]
    lines += ["  " + ln for ln in yaml.safe_dump(protocol, sort_keys=False, default_flow_style=None).splitlines()]
    lines.append("materials:")
    for it in req["items"]:
        d = {k: it[k] for k in ("id", "host", "shard", "cif")}
        d["protocol"] = it["protocol"]
        d["pred_h"] = round(it["pred_h"], 3)
        lines.append("  - " + json.dumps(d))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cost", required=True, help="T-070 metrics.json with 'fit' and 'k_supercell_spins_per_nominal'")
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--structures", default="compute/structures/e2")
    ap.add_argument("--n-conf", type=int, default=20)
    ap.add_argument("--n-conf-floored", type=int, default=None, help="configurations for floored hosts (default: --n-conf)")
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--min-r-dipole", type=float, default=None)
    ap.add_argument("--target-h", type=float, default=2.5)
    ap.add_argument("--max-jobs", type=int, default=256)
    ap.add_argument("--window-prior", type=float, default=None, help="t_max_ms = K x T2_SL (None: 50 ms start)")
    ap.add_argument("--first", type=int, default=None, help="first request number (default: next free)")
    ap.add_argument("--hosts", default=None, help="only these mp-ids ('a,b' or @file)")
    ap.add_argument("--exclude", default=None, help="drop these mp-ids ('a,b' or @file)")
    ap.add_argument("--directions", default=None, help="direction indices, e.g. '0' (default: all three)")
    ap.add_argument("--audit", type=int, default=None, help="draw this many capped non-dense hosts (T-071)")
    ap.add_argument("--audit-seed", type=int, default=0)
    ap.add_argument("--name", default="e2-b", help="request id stem: <NNN>-<name><KK>")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    started = dt.datetime.now(dt.timezone.utc)
    cost = json.loads((ROOT / a.cost).read_text(encoding="utf-8"))
    fit, k_sc = cost["fit"], float(cost["k_supercell_spins_per_nominal"])
    iso = load_kanai_isotope_table(ROOT / "data/external/kanai2022/isotopetable.csv")
    spin = dict(zip(iso["isotope"], iso["spin_I"]))
    smp = select_hosts(pd.read_csv(ROOT / a.sample), parse_ids(a.hosts), parse_ids(a.exclude))
    dsl = pd.read_csv(ROOT / "data/processed/dsl.csv.gz")
    hosts = smp[["mp_id", "sample_stratum", "forced_tail"]].merge(dsl, on="mp_id", how="left", validate="1:1")
    if a.audit is not None:
        hosts = hosts[hosts["mp_id"].isin(audit_draw(hosts, a.audit, a.audit_seed))].reset_index(drop=True)
    dirs = parse_directions(a.directions)

    items_by_host, rows = {}, []
    for _, r in hosts.iterrows():
        counts = json.loads(r["composition"])
        f, d = bath_mix(counts, isotopes_from_table(iso, counts), spin)
        t_eval = host_eval_seconds(fit, f, d, r["n_bath_50A"], FLOOR, a.cap, k_sc, a.min_r_dipole)
        dense = r["dense_share"] >= DENSE_SHARE
        n_conf = host_n_conf(dense, r["n_bath_50A"], a.n_conf, a.n_conf_floored)
        extra = extra_windows(r["t2_sl_s"] * 1e3) if a.window_prior is None else 0
        cif = f"{a.structures}/{r['mp_id']}.cif"
        if not (ROOT / cif).is_file():
            raise SystemExit(f"{cif} missing (scripts/make_e2_structures.py)")
        its = []
        for j in dirs:
            dvec = DIRECTIONS[j]
            chunks = plan_host(t_eval, n_conf, extra, a.target_h)
            for c, (seed, n, h) in enumerate(chunks):
                p = {"b_directions_lattice": [list(dvec)]}
                if a.window_prior is not None:
                    p["t_max_ms"] = window_prior_ms(r["t2_sl_s"], a.window_prior)
                if len(chunks) > 1:
                    p.update(seed=int(seed), n_conf=int(n))
                elif n_conf not in (1, a.n_conf):
                    p["n_conf"] = int(n_conf)
                its.append({"id": f"{r['mp_id']}-d{j}" + (f"-k{c}" if len(chunks) > 1 else ""),
                            "host": r["mp_id"], "cif": cif, "protocol": p, "pred_h": float(h),
                            "direction": j, "chunk": c, "n_chunks": len(chunks)})
        items_by_host[r["mp_id"]] = its
        rows.append({"mp_id": r["mp_id"], "material": r["material"], "stratum": r["sample_stratum"],
                     "dense": bool(dense), "t_eval_s": t_eval, "n_conf": n_conf, "extra_windows": extra,
                     "material_h": sum(it["pred_h"] for it in its), "n_items": len(its),
                     "max_item_h": max(it["pred_h"] for it in its),
                     "infeasible": bool((1 + extra) * t_eval / 3600 > JOB_LIMIT_H)})
    hdf = pd.DataFrame(rows)
    hosts = hosts.merge(hdf[["mp_id", "material_h"]], on="mp_id")
    order = round_robin(hosts)
    reqs = build(hosts, items_by_host, order, a.target_h, a.max_jobs)

    used = [int(p.stem[:3]) for d in ("requests", "drafts") for p in (ROOT / "compute" / d).glob("[0-9][0-9][0-9]-*.yaml")]
    first = a.first if a.first is not None else max(used) + 1
    if first in used:
        raise SystemExit(f"request number {first:03d} is taken")
    protocol = {"cell_setting": "conventional_standard", "n_conf": a.n_conf, "max_bath_spins": a.cap,
                "min_bath_spins": FLOOR, "seed": 0}
    if a.min_r_dipole is not None:
        protocol["min_r_dipole_A"] = a.min_r_dipole
    summary = []
    for k, req in enumerate(reqs):
        rid = f"{first + k:03d}-{a.name}{k + 1:02d}"
        req["id"] = rid
        summary.append({"id": rid, "hosts": len(req["hosts"]), "items": len(req["items"]), "shards": len(req["loads"]),
                        "runner_h": round(sum(req["loads"]), 1), "max_shard_h": round(max(req["loads"]), 2),
                        "wall_h_at_20": round(makespan(req["loads"]), 1),
                        "strata": pd.Series([hosts.set_index("mp_id").loc[h, "sample_stratum"] for h in req["hosts"]])
                        .value_counts().sort_index().to_dict()})
    tot = {"hosts": int(len(hdf)), "items": int(sum(len(v) for v in items_by_host.values())),
           "chunked_hosts": int((hdf["n_items"] > len(dirs)).sum()), "runner_h": round(float(hdf["material_h"].sum()), 1),
           "requests": len(reqs), "shards": int(sum(s["shards"] for s in summary)),
           "wall_h_at_20_sequential": round(sum(s["wall_h_at_20"] for s in summary), 1),
           "items_over_target": int(sum(it["pred_h"] > a.target_h for v in items_by_host.values() for it in v)),
           "infeasible_hosts": hdf.loc[hdf["infeasible"], "mp_id"].tolist()}
    settings = {"cost": a.cost, "sample": a.sample, "hosts": a.hosts, "exclude": a.exclude, "name": a.name, "directions": dirs, "audit": a.audit, "audit_seed": a.audit_seed, "audit_hosts": sorted(hdf["mp_id"]) if a.audit is not None else None, "n_conf": a.n_conf, "n_conf_floored": a.n_conf_floored, "cap": a.cap, "min_r_dipole_A": a.min_r_dipole,
                "window_prior": a.window_prior, "target_h": a.target_h, "max_jobs": a.max_jobs, "fit": fit, "k_sc": k_sc}
    print(json.dumps({"settings": settings, "total": tot, "requests": summary}, indent=1, default=str))
    if not a.write:
        return 0

    from qkhosts.run import env_text, git_state, make_run_dir
    sha, dirty = git_state()
    ended = dt.datetime.now(dt.timezone.utc)
    out = make_run_dir(ROOT / "results", "T-015", sha, dirty, ended)
    for k, req in enumerate(reqs):
        header = [f"DRAFT (T-015/T-071). E2 labels, directions {dirs}, {req['id']}, batch {k + 1} of {len(reqs)}: {len(req['hosts'])} hosts, "
                  f"{len(req['items'])} items, {len(req['loads'])} shards.",
                  f"Generated by scripts/make_e2_request.py ({started:%Y-%m-%d %H:%M} UTC); plan: {out.relative_to(ROOT).as_posix()}/",
                  f"Settings: n_conf {a.n_conf} (floored hosts {a.n_conf_floored or a.n_conf}), cap {a.cap}, floor {FLOOR}, r_dipole floor {a.min_r_dipole}, "
                  f"window prior {a.window_prior}, target {a.target_h} h per shard; cost model {a.cost}.",
                  f"Predicted: {sum(req['loads']):.1f} runner-h, max shard {max(req['loads']):.2f} h, about "
                  f"{makespan(req['loads']):.1f} h wall at 20 jobs (model x/÷ 2).",
                  "Items: one host x direction (x configuration chunk); `host` and `pred_h` are metadata; labels are",
                  "assembled with qkhosts.labels.assemble (scripts/assemble_e2_labels.py). Launch = git mv to",
                  "compute/requests/ with Kian's OK, one request per push."]
        (ROOT / "compute" / "drafts" / f"{req['id']}.yaml").write_text(
            request_yaml(req["id"], protocol, req, header), encoding="utf-8", newline="\n")
    plan = pd.DataFrame([{**{k: v for k, v in it.items() if k != "protocol"}, "request": req["id"]}
                         for req in reqs for it in req["items"]])
    plan.to_csv(out / "plan.csv", index=False, float_format="%.4g")
    hdf.to_csv(out / "hosts.csv", index=False, float_format="%.4g")
    (out / "metrics.json").write_text(json.dumps({"settings": settings, "total": tot, "requests": summary,
                                                  "protocol": protocol}, indent=1, default=str), encoding="utf-8")
    (out / "env.txt").write_text(env_text(sha, dirty, sys.argv, started, ended), encoding="utf-8")
    print(f"-> {out} and {len(reqs)} drafts in compute/drafts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
