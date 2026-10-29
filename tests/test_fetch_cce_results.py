"""Q9b (S010): local `gh` artifact path for CCE results (scripts/fetch_cce_results.py), offline parts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_cce_results.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="private-repo script (not in public/MANIFEST)")


@pytest.fixture(scope="module")
def fc():
    spec = importlib.util.spec_from_file_location("fetch_cce_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_artifact_newest_unexpired(fc):
    arts = [{"name": "cce-004-e0-bench-merged", "expired": False, "created_at": "2026-11-01T00:00:00Z", "id": 1},
            {"name": "cce-004-e0-bench-merged", "expired": False, "created_at": "2026-11-02T00:00:00Z", "id": 2},
            {"name": "cce-004-e0-bench-merged", "expired": True, "created_at": "2026-11-03T00:00:00Z", "id": 3},
            {"name": "cce-004-e0-bench-shard-0", "expired": False, "created_at": "2026-11-04T00:00:00Z", "id": 4}]
    assert fc.pick_artifact(arts, "004-e0-bench")["id"] == 2
    with pytest.raises(LookupError):
        fc.pick_artifact(arts[2:], "004-e0-bench")


def _artifact(d: Path, lines=('{"id": "a"}', '{"id": "b"}')):
    d.mkdir()
    (d / "merged.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "summary.json").write_text('{"n": 2}', encoding="utf-8")
    return d


def test_install_layout_and_write_once(fc, tmp_path):
    src = _artifact(tmp_path / "art")
    req = tmp_path / "r.yaml"
    req.write_text("id: x\n", encoding="utf-8")
    dest = tmp_path / "results" / "x"
    assert fc.install(src, dest, req, {"run_id": "1", "source": "gh-artifact"}) == 2
    assert sorted(p.name for p in dest.iterdir()) == ["merged.jsonl", "provenance.json", "request.yaml", "summary.json"]
    prov = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    assert prov == {"run_id": "1", "source": "gh-artifact", "n_records": 2}
    with pytest.raises(FileExistsError):
        fc.install(src, dest, req, {})


def test_install_rejects_bad_json(fc, tmp_path):
    src = _artifact(tmp_path / "art", lines=('{"id": "a"}', "not json"))
    req = tmp_path / "r.yaml"
    req.write_text("id: x\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        fc.install(src, tmp_path / "out", req, {})
    assert not (tmp_path / "out").exists()
