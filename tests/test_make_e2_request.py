"""E2 request generator (S020): chunking, window prior, round-robin order, packing, and the YAML it writes."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qkhosts.labels.batch import _protocol, shard_items

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "scripts" / "make_e2_request.py").is_file():  # private-only script (not in public/MANIFEST)
    pytest.skip("scripts/make_e2_request.py not present", allow_module_level=True)
spec = importlib.util.spec_from_file_location("make_e2_request", ROOT / "scripts" / "make_e2_request.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_plan_host_chunks_only_above_target():
    assert g.plan_host(360.0, 20, 1, 2.5) == [(0, 20, pytest.approx(2.1))]  # 21 evals x 0.1 h
    ch = g.plan_host(1800.0, 20, 0, 2.5)  # 0.5 h per eval: 5 per chunk
    assert [(s, n) for s, n, _ in ch] == [(0, 5), (5, 5), (10, 5), (15, 5)]
    assert all(h <= 2.5 for _, _, h in ch)
    ch = g.plan_host(1800.0, 20, 1, 2.5)  # each chunk pays one window: 4 configurations per chunk
    assert sum(n for _, n, _ in ch) == 20 and all(h <= 2.5 for _, _, h in ch)
    assert g.plan_host(36000.0, 1, 0, 2.5) == [(0, 1, pytest.approx(10.0))]  # dense: never chunked
    assert g.window_prior_ms(2.17e-5, 7) == pytest.approx(0.1519)


def test_round_robin_and_pack():
    h = pd.DataFrame({"mp_id": list("abcdefg"), "sample_stratum": ["S1", "S1", "S2", "S3", "S4", "S4", "S2"],
                      "material_h": [2, 1, 1, 1, 3, 1, 0.5]})
    assert g.round_robin(h) == ["b", "g", "d", "f", "a", "c", "e"]
    with pytest.raises(ValueError):
        g.round_robin(pd.DataFrame({"mp_id": ["x"], "sample_stratum": ["S9"], "material_h": [1]}))
    bins = g.pack([2.0, 1.0, 1.5, 0.5, 3.0], 2.5)
    assert sorted(i for b in bins for i in b) == [0, 1, 2, 3, 4]
    loads = [sum([2.0, 1.0, 1.5, 0.5, 3.0][i] for i in b) for b in bins]
    assert loads == sorted(loads, reverse=True) and loads[0] == 3.0 and all(x <= 2.5 for x in loads[1:])
    assert g.makespan([3, 2, 2, 1], n_parallel=2) == 4


def test_request_yaml_round_trips_through_batch(tmp_path):
    items = [{"id": "mp-1-d0-k0", "host": "mp-1", "cif": "compute/structures/e2/mp-1.cif", "pred_h": 1.2,
              "protocol": {"b_directions_lattice": [[0, 0, 1]], "seed": 0, "n_conf": 10, "t_max_ms": 0.15}, "shard": 1},
             {"id": "mp-2-d1", "host": "mp-2", "cif": "compute/structures/e2/mp-2.cif", "pred_h": 0.3,
              "protocol": {"b_directions_lattice": [[1, 1, 0]]}, "shard": 0}]
    text = g.request_yaml("099-e2-b01", {"cell_setting": "conventional_standard", "n_conf": 20, "seed": 0},
                          {"items": items, "loads": [0.3, 1.2]}, ["DRAFT", "", "x"])
    req = yaml.safe_load(text)
    assert req["id"] == "099-e2-b01" and req["nshards"] == 2 and text.startswith("# DRAFT\n#\n# x\n")
    p = _protocol(req, req["materials"][0])
    assert (p.cell_setting, p.n_conf, p.seed, p.t_max_ms, p.b_directions_lattice) == \
        ("conventional_standard", 10, 0, 0.15, [[0, 0, 1]])
    assert _protocol(req, req["materials"][1]).n_conf == 20
    assert [m["id"] for m in shard_items(req["materials"], 1, 2)] == ["mp-1-d0-k0"]


def test_host_n_conf_floored_and_dense():
    assert g.host_n_conf(True, 10.0, 10, 20) == 1            # dense bath: 1 configuration
    assert g.host_n_conf(False, 500.0, 10, 20) == 20         # floored (< 1,000 spins in 50 Å)
    assert g.host_n_conf(False, 5000.0, 10, 20) == 10        # capped
    assert g.host_n_conf(False, 500.0, 10, None) == 10       # no floored override


def test_select_hosts_only_and_exclude():
    smp = pd.DataFrame({"mp_id": ["mp-1", "mp-2", "mp-3"]})
    assert list(g.select_hosts(smp, ["mp-1", "mp-3"], [])["mp_id"]) == ["mp-1", "mp-3"]
    assert list(g.select_hosts(smp, [], ["mp-2"])["mp_id"]) == ["mp-1", "mp-3"]
    with pytest.raises(SystemExit):
        g.select_hosts(smp, ["mp-9"], [])
    assert g.parse_ids(" mp-1, mp-2 ,") == ["mp-1", "mp-2"] and g.parse_ids(None) == []


def test_audit_draw_and_directions():
    """T-071 (S022): the audit draw takes only capped non-dense hosts, balanced over strata, reproducibly."""
    rows = []
    for s in g.STRATA:
        for i in range(8):
            rows.append({"mp_id": f"{s}-{i}", "sample_stratum": s, "n_bath_50A": 3000.0 if i < 6 else 900.0,
                         "dense_share": 1.0 if i == 0 else 0.5})
    hosts = pd.DataFrame(rows)
    a = g.audit_draw(hosts, 10, seed=0)
    assert a == g.audit_draw(hosts, 10, seed=0) and len(a) == 10
    assert all(int(m.split("-")[1]) in range(1, 6) for m in a)  # capped, non-dense only
    assert sorted(pd.Series([m[:2] for m in a]).value_counts()) == [2, 2, 3, 3]
    assert len(g.audit_draw(hosts, 40)) == 20  # 5 eligible per stratum
    assert g.parse_directions(None) == [0, 1, 2] and g.parse_directions("0") == [0]
    with pytest.raises(SystemExit):
        g.parse_directions("3")
