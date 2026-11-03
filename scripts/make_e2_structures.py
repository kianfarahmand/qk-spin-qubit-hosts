#!/usr/bin/env python3
"""Write the E2 (T-015) CIFs for a host sample from the archived E1 OPTIMADE snapshot (S020).

    python scripts/make_e2_structures.py [--sample results/T-014/20261031T184133Z_e1ef094/sample_adr0007.csv]
                                         [--out compute/structures/e2] [--dry-run]

Reads data/external/mp_optimade_e1/entries.jsonl.gz (the S016 snapshot behind D-SL, Q18b), so every CIF is the
structure D-SL's T2_SL was computed on; no network. Unlike scripts/make_e2cost_structures.py, the cell is MP's own
(no conversion here): E2 items run with `cell_setting: conventional_standard`, so the runner converts each structure
to pymatgen's conventional standard cell on Actions (ADR-0004 item 2 directions c, a+b, a+b+c), for every crystal
system. Each entry is checked against D-SL: reduced formula, site count, composition (atoms per cell) and ordered
sites. CIFs go to <out>/<mp-id>.cif with SOURCES.json (sha256, last_modified, sample row); an existing CIF is never
overwritten (same bytes or the run stops).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qkhosts.data import optimade as om  # noqa: E402

SNAP = ROOT / "data" / "external" / "mp_optimade_e1" / "entries.jsonl.gz"
DSL = ROOT / "data" / "processed" / "dsl.csv.gz"
SAMPLE = "results/T-014/20261031T184133Z_e1ef094/sample_adr0007.csv"


def check_entry(entry: dict, dsl_row: dict) -> tuple:
    """Lattice, Cartesian sites and species of one snapshot entry after the D-SL checks (raises ValueError)."""
    attr = entry["attributes"]
    if attr["chemical_formula_reduced"] != dsl_row["formula_now"]:
        raise ValueError(f"formula {attr['chemical_formula_reduced']} != D-SL {dsl_row['formula_now']}")
    if int(attr["nsites"]) != int(dsl_row["nsites"]):
        raise ValueError(f"nsites {attr['nsites']} != D-SL {dsl_row['nsites']}")
    L, cart, spc = om.entry_arrays(entry)  # raises on disorder
    comp = json.loads(dsl_row["composition"])
    if dict(Counter(spc)) != {k: int(v) for k, v in comp.items()}:
        raise ValueError(f"composition {dict(Counter(spc))} != D-SL {comp}")
    return L, cart, spc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--out", default=str(ROOT / "compute" / "structures" / "e2"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    sample = pd.read_csv(ROOT / args.sample)
    dsl = pd.read_csv(DSL).set_index("mp_id")
    want = list(sample["mp_id"])
    entries = {}
    with gzip.open(SNAP, "rt", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["id"] in set(want):
                entries[e["id"]] = e
    missing = [m for m in want if m not in entries]
    if missing:
        raise SystemExit(f"{len(missing)} sample hosts not in the E1 snapshot: {missing[:5]}")
    out = Path(args.out)
    sources, states = {}, Counter()
    for _, row in sample.iterrows():
        mp_id = row["mp_id"]
        e = entries[mp_id]
        try:
            L, cart, spc = check_entry(e, {**dsl.loc[mp_id].to_dict(), "mp_id": mp_id})
        except ValueError as err:
            raise SystemExit(f"{mp_id}: {err}")
        attr = e["attributes"]
        cif = om.to_cif(L, cart, spc, mp_id.replace("-", "_"), comments=[
            f"{mp_id} ({attr['chemical_formula_reduced']}) from the E1 OPTIMADE snapshot "
            f"data/external/mp_optimade_e1/entries.jsonl.gz (last_modified {attr['last_modified']})",
            "MP cell as served; the E2 runner converts it to the conventional standard cell (cell_setting)",
            f"written by scripts/make_e2_structures.py (T-015) for {args.sample}"]).encode("utf-8")
        path = out / f"{mp_id}.cif"
        if path.exists():
            if path.read_bytes() != cif:
                raise SystemExit(f"{path} exists with different content; not overwriting")
            states["unchanged"] += 1
        elif args.dry_run:
            states["dry-run"] += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(cif)
            states["written"] += 1
        sources[f"{mp_id}.cif"] = {
            "mp_id": mp_id, "formula": attr["chemical_formula_reduced"], "nsites": int(attr["nsites"]),
            "last_modified": attr["last_modified"], "cif_sha256": hashlib.sha256(cif).hexdigest(),
            "cell_family_mp": om.cell_family(L), "sample_stratum": row["sample_stratum"],
            "forced_tail": bool(row["forced_tail"]), "cost_class": row["cost_class"]}
    if not args.dry_run:
        (out / "SOURCES.json").write_text(json.dumps({"sample": args.sample,
                                                       "snapshot": "data/external/mp_optimade_e1/entries.jsonl.gz",
                                                       "files": sources}, indent=1) + "\n",
                                          encoding="utf-8", newline="\n")
    fam = Counter(v["cell_family_mp"] for v in sources.values())
    print(f"{len(sources)} hosts: {dict(states)}; MP cell families {dict(fam)}; "
          f"sites max {max(v['nsites'] for v in sources.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
