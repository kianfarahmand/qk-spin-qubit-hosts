"""Assemble D-CCE host labels from E2 batch records (T-015, S020; experiments/E2_notes.md §4).

E2 requests (`scripts/make_e2_request.py`) hold one item per host x field direction x configuration chunk. Each
item names its host (`host`, the mp-id) and carries a one-direction protocol (`b_directions_lattice: [[...]]`);
a chunk differs from its siblings only in `seed` and `n_conf` (`qkhosts.labels.chunks`). This module turns the
items of one or more launched requests and their batch records (`merged.jsonl`) into one label per host:

1. per host and direction: the record's `final` step if the direction ran in one item, otherwise `merge_chunks`
   over its chunks (shortest window, weighted mean curve, one fit), after checking that the chunks share every
   protocol setting except seed and n_conf and that their configuration seeds do not overlap;
2. per host: label T2 = log-mean over the directions (strict: one NaN direction or a missing direction gives
   no label), as `run_material` does for a 3-direction item (ADR-0004 item 2).

Times are in ms. Nothing here needs PyCCE.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Sequence

import numpy as np

from qkhosts.labels.batch import _protocol
from qkhosts.labels.chunks import merge_chunks
from qkhosts.labels.pycce_runner import CCEProtocol, _log_mean, protocol_hash

ADR0004_DIRECTIONS = ((0, 0, 1), (1, 1, 0), (1, 1, 1))
_CHUNK_KEYS = ("seed", "n_conf", "b_directions_lattice")
_STATUS_RANK = {"ok": 0, "no_label": 1, "missing": 2, "error": 3, "incomplete": 4}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def item_direction(proto: CCEProtocol) -> tuple:
    """The single field direction of an E2 item (lattice combination of the conventional cell)."""
    dirs = proto.b_directions_lattice
    if dirs is None or len(dirs) != 1:
        raise ValueError("an E2 item must have exactly one entry in b_directions_lattice")
    return tuple(int(x) for x in dirs[0])


def record_final(rec: dict) -> dict:
    """The `final` step of a one-site, one-direction batch record."""
    sites = rec.get("sites") or []
    if len(sites) != 1 or "directions" in sites[0]:
        raise ValueError(f"{rec.get('id')}: expected one site and one direction")
    return sites[0]["final"]


def check_chunks(protos: Sequence[CCEProtocol]) -> None:
    """Chunks of one host direction must agree on every protocol field except seed and n_conf, and their
    configuration seeds (seed + 1 + k, k < n_conf) must not overlap."""
    ref = {k: v for k, v in asdict(protos[0]).items() if k not in _CHUNK_KEYS}
    used: set[int] = set()
    for p in protos:
        if {k: v for k, v in asdict(p).items() if k not in _CHUNK_KEYS} != ref:
            raise ValueError("chunks differ in a protocol setting other than seed / n_conf")
        seeds = set(range(p.seed + 1, p.seed + 1 + p.n_conf))
        if seeds & used:
            raise ValueError(f"chunks overlap in configuration seeds {sorted(seeds & used)[:3]}")
        used |= seeds


def label_protocol(protos: Sequence[CCEProtocol], directions: Iterable[tuple]) -> CCEProtocol:
    """The single-item protocol equivalent to a host's items: lowest seed, summed n_conf of one direction's
    chunks, all directions. Its hash identifies the label protocol across different chunkings."""
    p0 = min(protos, key=lambda p: p.seed)
    d0 = item_direction(p0)
    n = sum(p.n_conf for p in protos if item_direction(p) == d0)
    return CCEProtocol(**{**asdict(p0), "n_conf": n,
                          "b_directions_lattice": tuple(tuple(d) for d in directions)})


def assemble_labels(requests: Sequence[dict], records: Iterable[dict],
                    directions: Sequence[tuple] = ADR0004_DIRECTIONS) -> list[dict]:
    """One row per host named in the requests' items (first-seen order). requests: parsed request YAMLs;
    records: batch records of those requests (keyed by request id and item id; later duplicates win).
    status: "ok" (label present), "no_label" (all items ran, but a direction fit is NaN), "missing" (an item has no
    record, e.g. a timed-out shard), "error" (an item's record is not ok), "incomplete" (the requests lack a
    direction)."""
    recs = {(r.get("request"), r.get("id")): r for r in records}
    hosts: dict[str, list] = {}
    for req in requests:
        for it in req.get("materials", []):
            if "host" not in it:
                raise ValueError(f"{req['id']}/{it['id']}: E2 items need a 'host' key")
            p = _protocol(req, it)
            hosts.setdefault(it["host"], []).append((req["id"], it, p))
    rows = []
    want = [tuple(d) for d in directions]
    for host, items in hosts.items():
        by_dir: dict[tuple, list] = {}
        for rid, it, p in items:
            by_dir.setdefault(item_direction(p), []).append((rid, it, p))
        row = {"host": host, "n_items": len(items), "status": "ok", "problems": []}
        dir_t2, dir_n, dir_mof, dir_conf, wall = [], [], [], [], 0.0
        for d in want:
            grp = by_dir.get(d)
            if not grp:
                row["status"] = _worse(row["status"], "incomplete")
                row["problems"].append(f"direction {list(d)} not requested")
                dir_t2.append(float("nan")), dir_n.append(float("nan")), dir_mof.append(float("nan"))
                dir_conf.append(0)
                continue
            protos = [p for _, _, p in grp]
            check_chunks(protos)
            finals, bad = [], False
            for rid, it, p in grp:
                r = recs.get((rid, it["id"]))
                if r is None:
                    row["status"] = _worse(row["status"], "missing")
                    row["problems"].append(f"{rid}/{it['id']}: no record")
                    bad = True
                elif not r.get("ok"):
                    row["status"] = _worse(row["status"], "error")
                    row["problems"].append(f"{rid}/{it['id']}: {r.get('error', 'not ok')}")
                    bad = True
                else:
                    finals.append(record_final(r))
                    wall += float(r.get("wall_s", 0.0))
            if bad:
                dir_t2.append(float("nan")), dir_n.append(float("nan")), dir_mof.append(float("nan"))
                dir_conf.append(0)
                continue
            if len(finals) == 1:
                f = finals[0]
            else:
                p0 = protos[0]
                f = merge_chunks(finals, p0.min_fit_points, tuple(p0.fit_n_range), p0.max_conf_fail_frac,
                                 n_conf_requested=sum(_n_conf_used(grp, recs)))
            dir_t2.append(float(f["t2_ms"]))
            dir_n.append(float(f["stretch_n"]))
            dir_mof.append(float(f["t2_ms_mean_of_fits"]))
            dir_conf.append(int(f["n_conf_ok"]))
        t2 = _log_mean(np.array(dir_t2, float), strict=True)
        if row["status"] == "ok" and not np.isfinite(t2):
            row["status"] = "no_label"
        good = np.isfinite(dir_t2) & (np.array(dir_t2) > 0)
        lg = np.log10(np.array(dir_t2)[good])
        row.update(
            t2_ms=t2, log10_t2_ms=float(np.log10(t2)) if np.isfinite(t2) else float("nan"),
            label_ok=bool(np.isfinite(t2)), direction_t2_ms=dir_t2,
            direction_spread_log10=float(np.ptp(lg)) if len(lg) > 1 else float("nan"),
            stretch_n=float(np.nanmean(dir_n)) if np.isfinite(dir_n).any() else float("nan"),
            t2_ms_mean_of_fits=_log_mean(np.array(dir_mof, float), strict=True),
            n_conf_ok=dir_conf, n_chunks=[len(by_dir.get(d, [])) for d in want], wall_s=round(wall, 2),
            label_protocol_hash=protocol_hash(label_protocol([p for _, _, p in items], want)))
        rows.append(row)
    return rows


def _n_conf_used(grp, recs) -> list[int]:
    """Configurations each chunk ran (n_conf_used: 1 for dense baths), for the merged failure rule."""
    out = []
    for rid, it, p in grp:
        r = recs[(rid, it["id"])]
        out.append(int(r.get("n_conf_used", p.n_conf)))
    return out
