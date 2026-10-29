"""Draft compute requests (compute/drafts/) must parse into a valid protocol and never reuse a request id."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qkhosts.labels.batch import _protocol
from qkhosts.labels.pycce_runner import protocol_hash

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = sorted((ROOT / "compute" / "drafts").glob("*.yaml"))


@pytest.mark.parametrize("path", DRAFTS, ids=[p.name for p in DRAFTS])
def test_draft_request_is_valid(path):
    req = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert req["id"] == path.stem and req["mode"] == "cce"
    assert not (ROOT / "compute" / "requests" / path.name).exists(), "already launched; delete the draft"
    used = {yaml.safe_load(p.read_text(encoding="utf-8"))["id"] for p in (ROOT / "compute" / "requests").glob("*.yaml")}
    assert req["id"] not in used
    proto = _protocol(req)  # raises on unknown keys
    assert proto.as_delay is False and proto.b_field_T == 5.0
    assert isinstance(protocol_hash(proto), str)
    assert 1 <= req["nshards"] <= len(req["materials"])
    assert len({m["id"] for m in req["materials"]}) == len(req["materials"]), "item ids name the outputs"
    for m in req["materials"]:
        assert _protocol(req, m).as_delay is False  # per-item overrides must be valid too
        assert m["cif"].startswith("compute/structures/")
        assert (ROOT / m["cif"]).is_file(), f"{m['cif']} missing (scripts/fetch_structures_optimade.py)"


@pytest.mark.skipif(not (ROOT / "public" / "MANIFEST").exists(), reason="private repo only")
def test_drafts_not_synced():
    manifest = (ROOT / "public" / "MANIFEST").read_text(encoding="utf-8")
    assert "compute/drafts" not in manifest
