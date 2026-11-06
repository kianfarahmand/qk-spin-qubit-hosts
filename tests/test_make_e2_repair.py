"""S024: E2 repair drafts copy failed or lost items unchanged and repack only the shards."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_e2_repair import build_repair  # noqa: E402
from make_e2_request import request_yaml  # noqa: E402

P = {"cell_setting": "conventional_standard", "n_conf": 10}


def _req(rid, n):
    return {"id": rid, "mode": "cce", "nshards": n, "protocol": dict(P),
            "materials": [{"id": f"h{i}-d0", "host": f"h{i}", "shard": i, "cif": f"s/h{i}.cif",
                           "protocol": {"b_directions_lattice": [[0, 0, 1]], "t_max_ms": 0.1 * (i + 1)},
                           "pred_h": 1.0 + i} for i in range(n)]}


def test_repair_takes_missing_and_failed_items_unchanged():
    q = _req("013", 4)
    recs = [{"id": "h0-d0", "ok": True}, {"id": "h1-d0", "ok": False, "error": "ValueError: boom\ntrace"},
            {"id": "h3-d0", "ok": True}]
    rep = build_repair([(q, recs)], target_h=2.5)
    assert [it["id"] for it in rep["items"]] == ["h2-d0", "h1-d0"]  # heaviest shard first
    assert rep["why"] == {"013/h1-d0": "ValueError: boom", "013/h2-d0": "no record"}
    for it in rep["items"]:
        src = next(m for m in q["materials"] if m["id"] == it["id"])
        assert {k: v for k, v in it.items() if k != "shard"} == {k: v for k, v in src.items() if k != "shard"}
    assert rep["loads"] == [3.0, 2.0] and rep["protocol"] == P
    text = request_yaml("017-e2-repair01", rep["protocol"], rep, ["DRAFT"])
    back = yaml.safe_load(text)
    assert back["nshards"] == 2 and back["protocol"] == P and len(back["materials"]) == 2


def test_repair_refuses_mixed_protocols_and_handles_nothing_to_do():
    a, b = _req("a", 1), _req("b", 1)
    b["protocol"] = {**P, "max_bath_spins": 3000}
    with pytest.raises(ValueError, match="protocol"):
        build_repair([(a, []), (b, [])], 2.5)
    assert build_repair([(a, [{"id": "h0-d0", "ok": True}])], 2.5)["items"] == []
