"""Schema checks for data/external/experimental_T2.csv (T-053): no value without a source."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "external" / "experimental_T2.csv"
BIB = ROOT / "paper" / "refs.bib"
pytestmark = pytest.mark.skipif(not (CSV.exists() and BIB.exists()), reason="private repo only (anchors + bib)")


def _rows() -> list[dict[str, str]]:
    with open(CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_every_row_has_value_doi_and_location():
    rows = _rows()
    assert rows
    for r in rows:
        assert re.fullmatch(r"A\d{3}", r["anchor_id"])
        assert float(r["T2_ms"]) > 0
        assert r["value_type"] in {"value", "lower_bound", "upper_bound", "saturation_value", "fit_component"}
        assert r["doi"].startswith("10.") and r["source_location"].strip()


def test_ids_unique_and_dois_in_bib():
    rows = _rows()
    assert len({r["anchor_id"] for r in rows}) == len(rows)
    bib = BIB.read_text(encoding="utf-8").lower()
    for r in rows:
        assert f"doi={{{r['doi'].lower()}}}" in bib, r["doi"]
