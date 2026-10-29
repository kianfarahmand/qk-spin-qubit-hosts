"""Bibliography guards (T-060): every entry has a DOI or arXiv id and a verification note; nothing cited is UNVERIFIED."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
SECTIONS = ROOT / "paper" / "sections"
pytestmark = pytest.mark.skipif(not BIB.exists(), reason="private repo only (paper/)")


def _entries() -> dict[str, str]:
    """Map bib key -> entry body (one entry per line, as refs.bib is written)."""
    text = BIB.read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"^@\w+\{(\w+),(.*)$", text, re.M)}


def _unverified_keys() -> set[str]:
    """Keys on the line right after a '% UNVERIFIED' comment, or whose note says UNVERIFIED."""
    lines = BIB.read_text(encoding="utf-8").splitlines()
    keys = set()
    for i, line in enumerate(lines):
        m = re.match(r"^@\w+\{(\w+),", line)
        if m and (("note={UNVERIFIED" in line) or (i > 0 and lines[i - 1].startswith("% UNVERIFIED"))):
            keys.add(m.group(1))
    return keys


def test_every_entry_has_identifier_and_note():
    entries = _entries()
    assert entries
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", BIB.read_text(encoding="utf-8")), "control char (broken TeX escape?)"
    for key, body in entries.items():
        assert re.search(r"\bdoi=\{10\.", body) or re.search(r"\beprint=\{\d{4}\.\d{4,5}\}", body), key
        assert re.search(r"\bnote=\{[^}]*verified", body, re.I), key


def test_no_cited_key_is_missing_or_unverified():
    entries, bad = _entries(), _unverified_keys()
    cited = set()
    for tex in [ROOT / "paper" / "main.tex", *SECTIONS.glob("*.tex")]:
        if tex.exists():
            for group in re.findall(r"\\cite[pt]?\*?\{([^}]*)\}", tex.read_text(encoding="utf-8")):
                cited.update(k.strip() for k in group.split(","))
    assert cited <= set(entries), cited - set(entries)
    assert not (cited & bad), cited & bad
