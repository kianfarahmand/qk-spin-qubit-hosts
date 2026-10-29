"""Tests for the stdlib xlsx reader (qkhosts.data.xlsx) and the Toriyama sheets it reads (T-017)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from qkhosts.data.xlsx import col_index, read_xlsx

ROOT = Path(__file__).resolve().parents[1]
M = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _write_xlsx(path: Path) -> None:
    sheet = (f'<worksheet xmlns="{M}"><sheetData>'
             '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="inlineStr"><is><t>n</t></is></c></row>'
             '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>1.5</v></c><c r="C2" t="b"><v>1</v></c></row>'
             '</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{M}" xmlns:r="{R}"><sheets>'
                   f'<sheet name="S1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/></Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{M}"><si><t>Compound</t></si>'
                   '<si><r><t>Si</t></r><r><t>O2</t></r></si></sst>')
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def test_col_index():
    assert [col_index(c) for c in ("A", "Z", "AA", "AZ", "BA")] == [0, 25, 26, 51, 52]


def test_read_xlsx_types_and_padding(tmp_path):
    p = tmp_path / "t.xlsx"
    _write_xlsx(p)
    assert read_xlsx(p) == {"S1": [["Compound", None, "n"], ["SiO2", 1.5, True]]}


@pytest.mark.skipif(not (ROOT / "data/external/toriyama2025/SHA256SUMS").exists(), reason="T-017 data not present")
def test_toriyama_sheets_match_paper_counts():
    """Counts and values quoted in toriyama2025 (Results; Fig. 3f) and our notes."""
    d = ROOT / "data/external/toriyama2025"
    (sub,) = read_xlsx(d / "Substrates_T2_Data.xlsx").values()
    t2 = {r[0]: r[1] for r in sub[1:]}
    assert round(t2["SiO2_Quartz"], 2) == 3.33 and round(t2["CeO2"], 1) == 45.7 and round(t2["CaO"], 1) == 31.9
    (mono,) = read_xlsx(d / "Data_2D-Materials.xlsx").values()
    assert len(mono) - 1 == 1172 and sum(r[1] > 1 for r in mono[1:]) == 189
    assert len(sub) - 1 == 7
