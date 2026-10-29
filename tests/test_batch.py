import json

import yaml

from qkhosts.labels.batch import main


def test_envcheck_and_merge_without_pycce(tmp_path):
    req = tmp_path / "r.yaml"
    req.write_text(yaml.safe_dump({"id": "t-env", "mode": "envcheck", "nshards": 1}))
    out = tmp_path / "out"
    assert main(["--request", str(req), "--shard", "0", "--out", str(out)]) == 0
    rec = json.loads((out / "shard_0.jsonl").read_text().splitlines()[0])
    assert rec["request"] == "t-env" and "versions" in rec  # error captured if pycce missing, not raised
    assert main(["--request", str(req), "--merge", "--out", str(out)]) == 0
    assert json.loads((out / "summary.json").read_text())["n_records"] == 1


def test_cce_mode_captures_errors(tmp_path):
    req = tmp_path / "r.yaml"
    req.write_text(yaml.safe_dump({"id": "t-cce", "mode": "cce", "nshards": 2,
                                   "materials": [{"id": "a", "cif": "missing_a.cif"}, {"id": "b", "cif": "missing_b.cif"}]}))
    out = tmp_path / "out"
    for s in (0, 1):
        assert main(["--request", str(req), "--shard", str(s), "--out", str(out), "--root", str(tmp_path)]) == 0
    recs = [json.loads(l) for p in sorted(out.glob("shard_*.jsonl")) for l in p.read_text().splitlines()]
    assert {r["id"] for r in recs} == {"a", "b"} and not any(r["ok"] for r in recs)


def test_item_protocol_overrides_request():
    import pytest
    from qkhosts.labels.batch import _protocol

    req = {"protocol": {"order": 2, "n_conf": 5}}
    p = _protocol(req, {"id": "x", "protocol": {"method": "gcce", "order": 3}})
    assert (p.method, p.order, p.n_conf) == ("gcce", 3, 5)
    assert _protocol(req).method == "cce"
    with pytest.raises(ValueError):
        _protocol(req, {"protocol": {"nope": 1}})
