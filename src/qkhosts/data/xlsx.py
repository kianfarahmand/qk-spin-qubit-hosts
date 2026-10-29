"""Minimal stdlib reader for .xlsx sheets (cached cell values only; no formulas evaluated).

Used for small external tables (Toriyama 2025 Zenodo sheets, T-017) so that no extra
dependency (openpyxl) is needed. Reads shared strings, inline strings, booleans and numbers.
Dates are returned as Excel serial numbers (floats); the sheets we read have none.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
_REF = re.compile(r"([A-Z]+)(\d+)")

Cell = str | float | bool | None


def col_index(letters: str) -> int:
    """Zero-based column index of an Excel column name ('A' -> 0, 'AA' -> 26)."""
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _text(el: ET.Element) -> str:
    return "".join(t.text or "" for t in el.iter(f"{{{_NS['m']}}}t"))


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return [_text(si) for si in root.findall("m:si", _NS)]


def _sheet_paths(z: zipfile.ZipFile) -> dict[str, str]:
    """Map sheet name -> zip path of its XML, in workbook order."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", _NS)}
    out = {}
    for s in wb.find("m:sheets", _NS):
        t = target[s.get(f"{{{_NS['r']}}}id")].lstrip("/")
        out[s.get("name")] = t if t.startswith("xl/") else f"xl/{t}"
    return out


def read_xlsx(path: str | Path) -> dict[str, list[list[Cell]]]:
    """Return {sheet name: rows}, each row a list of cell values padded to the sheet width."""
    with zipfile.ZipFile(path) as z:
        sst = _shared_strings(z)
        book = {}
        for name, sp in _sheet_paths(z).items():
            root = ET.fromstring(z.read(sp))
            cells: dict[tuple[int, int], Cell] = {}
            for c in root.iter(f"{{{_NS['m']}}}c"):
                m = _REF.fullmatch(c.get("r", ""))
                if m is None:
                    continue
                key = (int(m.group(2)) - 1, col_index(m.group(1)))
                t, v = c.get("t"), c.find("m:v", _NS)
                if t == "inlineStr":
                    is_ = c.find("m:is", _NS)
                    cells[key] = _text(is_) if is_ is not None else None
                elif v is None or v.text is None:
                    cells[key] = None
                elif t == "s":
                    cells[key] = sst[int(v.text)]
                elif t in ("str", "e"):
                    cells[key] = v.text
                elif t == "b":
                    cells[key] = v.text == "1"
                else:
                    cells[key] = float(v.text)
            if not cells:
                book[name] = []
                continue
            nr = max(r for r, _ in cells) + 1
            nc = max(c for _, c in cells) + 1
            book[name] = [[cells.get((r, c)) for c in range(nc)] for r in range(nr)]
        return book
