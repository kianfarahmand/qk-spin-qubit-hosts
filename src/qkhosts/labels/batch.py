"""Batch entry point for CCE compute jobs (run on GitHub Actions; see compute/README.md).

    python -m qkhosts.labels.batch --request compute/requests/<id>.yaml --shard I --nshards N --out OUT
    python -m qkhosts.labels.batch --request ... --merge OUT_DIR   # merge shard outputs

Request file (YAML):
    id: 000-envcheck            # unique, used as results folder name
    mode: envcheck | cce        # envcheck = install/API probe + toy diamond run
    nshards: 1                  # parallel jobs (<= 20 run concurrently on GitHub Free)
    protocol: {...}             # CCEProtocol overrides (mode cce)
    materials:                  # mode cce: list of {id, cif} (cif path relative to repo root)
      - {id: mp-149, cif: compute/structures/mp-149.cif}

Every job writes <OUT>/shard_<I>.jsonl, one JSON record per material, with errors captured rather than
raised, so one bad material never kills a shard.
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

from qkhosts.labels.pycce_runner import CCEProtocol, fit_t2, protocol_hash, run_material


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "scipy", "pycce", "ase", "pymatgen"):
        try:
            out[m] = getattr(__import__(m), "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            out[m] = f"MISSING ({type(e).__name__})"
    return out


def envcheck() -> dict:
    """Probe the installed PyCCE API and run a tiny diamond 13C-bath toy calculation.
    The output (signatures, timings, toy T2) is what E0 needs to finish run_material (T-010).
    The toy result is a smoke test, NOT a benchmark number."""
    rep: dict = {"versions": _versions(), "steps": {}}
    try:
        import pycce as pc
        from ase.build import bulk

        rep["pycce_api"] = {
            "has_read_ase": hasattr(pc, "read_ase"),
            "has_BathCell_from_ase": hasattr(getattr(pc, "BathCell", None), "from_ase"),
            "has_CenterArray": hasattr(pc, "CenterArray"),
            "Simulator_init": str(inspect.signature(pc.Simulator.__init__)),
            "Simulator_compute": str(inspect.signature(pc.Simulator.compute)),
        }
        atoms_ase = bulk("C", "diamond", cubic=True)
        cell = pc.read_ase(atoms_ase) if hasattr(pc, "read_ase") else pc.BathCell.from_ase(atoms_ase)
        cell.add_isotopes(("13C", 0.011))
        t0 = time.time()
        bath = cell.gen_supercell(100, remove=[("C", [0.0, 0.0, 0.0])], seed=1)
        rep["steps"]["supercell"] = {"ok": True, "n_bath": int(len(bath)), "s": round(time.time() - t0, 2)}
        ts = np.linspace(0, 3, 61)  # ms (PyCCE default time unit — confirm in E0)
        kwargs = dict(bath=bath, r_bath=30, r_dipole=6, order=2, magnetic_field=[0, 0, 500], pulses=1)
        t0 = time.time()
        try:
            calc = pc.Simulator(spin=0.5, position=[0, 0, 0], **kwargs)
            rep["steps"]["simulator_style"] = "spin=0.5, position=..."
        except Exception:  # noqa: BLE001
            center = pc.CenterArray(spin=0.5, position=[0, 0, 0])
            calc = pc.Simulator(spin=center, **kwargs)
            rep["steps"]["simulator_style"] = "spin=CenterArray(...)"
        L = np.abs(np.asarray(calc.compute(ts, method="cce", quantity="coherence")))
        t2, n = fit_t2(ts, L)
        rep["steps"]["toy_cce"] = {"ok": True, "s": round(time.time() - t0, 2), "toy_t2_ms": t2, "stretch_n": n,
                                   "L_first_last": [float(L[0]), float(L[-1])]}
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()[-3000:]
    return rep


def _protocol(req: dict) -> CCEProtocol:
    names = {f.name for f in fields(CCEProtocol)}
    over = req.get("protocol", {}) or {}
    bad = set(over) - names
    if bad:
        raise ValueError(f"unknown protocol keys {sorted(bad)}")
    return CCEProtocol(**over)


def run_shard(req: dict, shard: int, nshards: int, out: Path, root: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{shard}.jsonl"
    with path.open("w") as f:
        if req["mode"] == "envcheck":
            rec = {"request": req["id"], "shard": shard, **envcheck()}
            f.write(json.dumps(rec, default=str) + "\n")
            print(json.dumps(rec, indent=1, default=str))
            return 0
        proto = _protocol(req)
        items = req.get("materials", [])[shard::nshards]
        for it in items:
            t0 = time.time()
            rec = {"request": req["id"], "shard": shard, "id": it["id"], "protocol_hash": protocol_hash(proto),
                   "protocol": asdict(proto)}
            try:
                rec.update(run_material(root / it["cif"], proto, out / "per_material"))
                rec["ok"] = True
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
            rec["wall_s"] = round(time.time() - t0, 2)
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(rec["id"], "ok" if rec["ok"] else rec["error"])
    return 0


def merge(out: Path) -> Path:
    recs = [json.loads(line) for p in sorted(out.rglob("shard_*.jsonl")) for line in p.read_text().splitlines() if line]
    merged = out / "merged.jsonl"
    merged.write_text("".join(json.dumps(r, default=str) + "\n" for r in recs))
    summary = {"n_records": len(recs), "n_ok": sum(bool(r.get("ok", "error" not in r)) for r in recs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=None)
    ap.add_argument("--out", default="cce_out")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    req = yaml.safe_load(Path(a.request).read_text())
    for k in ("id", "mode"):
        if k not in req:
            raise SystemExit(f"request missing '{k}'")
    if a.merge:
        merge(Path(a.out))
        return 0
    n = a.nshards or int(req.get("nshards", 1))
    if not 0 <= a.shard < n:
        raise SystemExit("shard out of range")
    return run_shard(req, a.shard, n, Path(a.out), Path(a.root))


if __name__ == "__main__":
    sys.exit(main())
