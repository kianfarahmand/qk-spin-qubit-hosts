"""Kanai et al. (PNAS 2022) host universe: sheet TableS1 of `extended data.xlsx` (T-011, S016).

Source: data/external/kanai2022_zip/extended_data.xlsx (from the arXiv source zip of Zenodo record 6323098,
CC-BY-4.0; `scripts/fetch_kanai2022_zip.py`). One row per host: reduced formula, Materials Project id, predicted
T2 (s, scaling law), band gap (eV), magnetic moment (mu_B), Debye temperature (K), densest spinful isotope and its
number density (cm^-3), crystal system, point group, bulk modulus (GPa). 12,859 rows, all with band gap > 1 eV.

Kanai's script used the MP `unit_cell_formula` and cell volume; the sheet gives the reduced formula and the
densest isotope's density instead. Both are intensive, so the volume per reduced formula unit follows from
V = count(el) * abundance(iso) / n(iso) (`volume_per_formula_A3`). Needs the standard library and pandas only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from qkhosts.data.xlsx import read_xlsx

COLUMNS = ["material", "mp_id", "t2_kanai_s", "band_gap_eV", "magmom_muB", "debye_K", "densest_isotope",
           "densest_n_cm3", "crystal_system", "point_group", "bulk_modulus_GPa"]
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\))(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """Element counts of a formula with nested parentheses, e.g. 'Ce(SO4)2' -> {Ce: 1, S: 2, O: 8}.
    Raises ValueError on characters outside element symbols, digits and parentheses."""
    stack: list[dict[str, float]] = [{}]
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"cannot parse formula {formula!r} at {pos}")
        pos = m.end()
        el, num, opn, cls, mult = m.groups()
        if el:
            stack[-1][el] = stack[-1].get(el, 0.0) + (float(num) if num else 1.0)
        elif opn:
            stack.append({})
        else:
            if len(stack) == 1:
                raise ValueError(f"unbalanced ')' in {formula!r}")
            inner = stack.pop()
            k = float(mult) if mult else 1.0
            for e, n in inner.items():
                stack[-1][e] = stack[-1].get(e, 0.0) + n * k
    if pos != len(formula) or len(stack) != 1:
        raise ValueError(f"cannot parse formula {formula!r}")
    return stack[0]


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) else float("nan")  # 'N/A' -> NaN


def load_table_s1(path: str | Path) -> pd.DataFrame:
    """The TableS1 sheet as a DataFrame with `COLUMNS`. Numeric 'N/A' cells become NaN."""
    rows = read_xlsx(path)["TableS1"]
    vals = [[getattr(c, "value", c) for c in r] for r in rows]
    header, body = vals[0], [r for r in vals[1:] if r and r[0]]
    if len(header) < len(COLUMNS) or not str(header[1]).startswith("Materials Project"):
        raise ValueError(f"unexpected TableS1 header: {header}")
    out = pd.DataFrame([r[:len(COLUMNS)] + [None] * (len(COLUMNS) - len(r)) for r in body], columns=COLUMNS)
    for c in ("t2_kanai_s", "band_gap_eV", "magmom_muB", "debye_K", "densest_n_cm3", "bulk_modulus_GPa"):
        out[c] = out[c].map(_num)
    return out


def split_isotope(label: str) -> tuple[int, str]:
    """'17O' -> (17, 'O')."""
    m = re.fullmatch(r"(\d+)([A-Z][a-z]?)", str(label))
    if not m:
        raise ValueError(f"not an isotope label: {label!r}")
    return int(m.group(1)), m.group(2)


def volume_per_formula_A3(counts: dict[str, float], densest_isotope: str, densest_n_cm3: float,
                          iso: pd.DataFrame) -> float:
    """Volume (Å^3) per reduced formula unit from the densest isotope: V = count * abundance / n.
    iso: table from `load_kanai_isotope_table` (abundance as a fraction)."""
    _, el = split_isotope(densest_isotope)
    row = iso[iso["isotope"] == densest_isotope]
    if len(row) != 1 or el not in counts or not densest_n_cm3 > 0:
        return float("nan")
    return float(counts[el] * row["abundance"].iloc[0] / densest_n_cm3 / 1e-24)
