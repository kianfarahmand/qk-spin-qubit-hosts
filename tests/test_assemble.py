"""E2 label assembly (S020): split items (one direction, configuration chunks) must reproduce the label of one
3-direction item on a fake PyCCE; status rules; chunk checks; explicit shard packing in batch; cell_setting."""
import sys
import types

import numpy as np
import pytest

from qkhosts.labels import pycce_runner as pr
from qkhosts.labels.assemble import assemble_labels, check_chunks, label_protocol
from qkhosts.labels.batch import shard_items
from qkhosts.labels.chunks import chunk_plan, merge_chunks
from qkhosts.labels.pycce_runner import CCEProtocol, run_material

DIRS = ((0, 0, 1), (1, 1, 0), (1, 1, 1))
BASE = dict(site_rule="host_atom_removed_avg_inequivalent", n_sites_max=1, ladder=((50.0, 15.0),),
            max_bath_spins=None, min_bath_spins=None, n_conf_dense=None)


class _Bath:
    def __init__(self, seed):
        self.seed, self.xyz = seed, np.array([[1.0, 0, 0], [0, 2.0, 0]])

    def __len__(self):
        return 1000


class _Cell:
    cell = np.eye(3) * 4.0

    def gen_supercell(self, size, remove=None, seed=None):
        return _Bath(seed)


def _fake():
    """T2 depends on the field direction (1, 2, 4 ms along x, y, z) and, by +-5 %, on the isotope seed."""
    class Sim:
        def __init__(self, **kw):
            self.kw = kw

        def compute(self, ts, method, quantity):
            b = np.asarray(self.kw["magnetic_field"], float)
            b = b / np.linalg.norm(b)
            t2 = float(np.dot([1.0, 2.0, 4.0], b ** 2)) * (1 + 0.05 * ((self.kw["bath"].seed % 5) - 2) / 2)
            return np.exp(-(np.asarray(ts) / t2) ** 2.5)

    return types.SimpleNamespace(read_ase=lambda atoms: _Cell(), Simulator=Sim)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycce", _fake())
    monkeypatch.setattr(pr, "_load_sites", lambda path, n, **kw: (None, [("Si", np.zeros(3), 1)]))


def _record(rid, item_id, res):
    return {"request": rid, "id": item_id, "ok": True, **res, "wall_s": 1.0}


def test_split_items_reproduce_one_three_direction_item(fake, tmp_path):
    whole = run_material(tmp_path / "x.cif", CCEProtocol(**BASE, n_conf=6, b_directions_lattice=DIRS),
                         tmp_path / "whole")
    items, recs = [], []
    for j, d in enumerate(DIRS):
        for c, (seed, n) in enumerate(chunk_plan(6, 3)):
            it = {"id": f"mp-1-d{j}-k{c}", "host": "mp-1", "cif": "x.cif",
                  "protocol": {"seed": seed, "n_conf": n, "b_directions_lattice": [list(d)]}}
            res = run_material(tmp_path / "x.cif", CCEProtocol(**BASE, seed=seed, n_conf=n, b_directions_lattice=(d,)),
                               tmp_path / "split", name=it["id"])
            items.append(it)
            recs.append(_record("e2-t", it["id"], res))
    req = {"id": "e2-t", "mode": "cce", "protocol": dict(BASE), "materials": items}
    (row,) = assemble_labels([req], recs)
    assert row["status"] == "ok" and row["label_ok"] and row["n_chunks"] == [2, 2, 2]
    assert row["n_conf_ok"] == [6, 6, 6]
    # chunks pick their own time window, so the merge interpolates (linear; ~1e-5 here); c needs no interpolation
    assert row["direction_t2_ms"] == pytest.approx(whole["direction_t2_ms"], rel=1e-4)
    assert row["direction_t2_ms"][0] == pytest.approx(whole["direction_t2_ms"][0], rel=1e-12)
    assert row["t2_ms"] == pytest.approx(whole["t2_ms"], rel=1e-4)
    # the label protocol hash equals that of the equivalent single item
    assert row["label_protocol_hash"] == pr.protocol_hash(CCEProtocol(**BASE, n_conf=6, b_directions_lattice=DIRS))


def test_status_rules(fake, tmp_path):
    items, recs = [], []
    for j, d in enumerate(DIRS):
        it = {"id": f"h-d{j}", "host": "h", "cif": "x.cif", "protocol": {"b_directions_lattice": [list(d)]}}
        items.append(it)
        res = run_material(tmp_path / "x.cif", CCEProtocol(**BASE, n_conf=2, b_directions_lattice=(d,)),
                           tmp_path / "o", name=it["id"])
        recs.append(_record("r", it["id"], res))
    req = {"id": "r", "mode": "cce", "protocol": dict(BASE), "materials": items}
    assert assemble_labels([req], recs)[0]["status"] == "ok"
    missing = assemble_labels([req], recs[:2])[0]
    assert missing["status"] == "missing" and not missing["label_ok"] and np.isnan(missing["t2_ms"])
    err = [dict(recs[0], ok=False, error="TimeoutError")] + recs[1:]
    assert assemble_labels([req], err)[0]["status"] == "error"
    nan = [recs[0], recs[1], {**recs[2], "sites": [{**recs[2]["sites"][0],
                                                   "final": {**recs[2]["sites"][0]["final"], "t2_ms": float("nan")}}]}]
    assert assemble_labels([req], nan)[0]["status"] == "no_label"
    short = {**req, "materials": items[:2]}
    assert assemble_labels([short], recs)[0]["status"] == "incomplete"
    with pytest.raises(ValueError, match="host"):
        assemble_labels([{**req, "materials": [{"id": "x", "cif": "x.cif"}]}], [])
    # S024 repair: a later request re-running item h-d0 under the same id supersedes the failed/missing one
    rep = {"id": "rep", "mode": "cce", "protocol": dict(BASE), "materials": [items[0]]}
    fixed = assemble_labels([req, rep], err[1:] + [dict(recs[0], request="rep")])[0]
    assert fixed["status"] == "ok" and fixed["n_items"] == 3 and fixed["superseded"] == ["r/h-d0"]
    assert fixed["t2_ms"] == pytest.approx(assemble_labels([req], recs)[0]["t2_ms"])
    lost = assemble_labels([req, rep], recs[1:])[0]  # repair record not there yet: still missing
    assert lost["status"] == "missing" and lost["problems"] == ["rep/h-d0: no record"]


def test_check_chunks_rejects_overlap_and_mismatch():
    a = CCEProtocol(seed=0, n_conf=5, b_directions_lattice=((0, 0, 1),))
    check_chunks([a, CCEProtocol(seed=5, n_conf=5, b_directions_lattice=((0, 0, 1),))])
    with pytest.raises(ValueError, match="overlap"):
        check_chunks([a, CCEProtocol(seed=3, n_conf=5, b_directions_lattice=((0, 0, 1),))])
    with pytest.raises(ValueError, match="other than"):
        check_chunks([a, CCEProtocol(seed=5, n_conf=5, max_bath_spins=3000, b_directions_lattice=((0, 0, 1),))])
    p = label_protocol([a, CCEProtocol(seed=5, n_conf=5, b_directions_lattice=((0, 0, 1),))], DIRS)
    assert (p.seed, p.n_conf, p.b_directions_lattice) == (0, 10, DIRS)


def test_merge_from_mean_curves_equals_merge_from_configuration_curves():
    rng = np.random.default_rng(0)
    finals = []
    for t_max, k in ((3.0, 4), (4.5, 3), (3.6, 5)):
        ts = np.linspace(0, t_max, 201)
        cs = [np.exp(-(ts / t2) ** 2.4) for t2 in rng.uniform(0.8, 1.2, k)]
        finals.append({"curve_t_ms": ts.tolist(), "conf_curves": [c.tolist() for c in cs],
                       "curve_mean": np.mean(cs, axis=0).tolist(), "n_conf_ok": k,
                       "per_conf_t2_ms": [pr.fit_t2(ts, c)[0] for c in cs], "conf_errors": []})
    full = merge_chunks(finals)
    lean = merge_chunks([{k: v for k, v in f.items() if k != "conf_curves"} for f in finals])
    assert lean["n_conf_ok"] == full["n_conf_ok"] == 12
    assert lean["t2_ms"] == pytest.approx(full["t2_ms"], rel=1e-12)
    assert np.allclose(lean["curve_mean"], full["curve_mean"], rtol=0, atol=1e-14)


def test_shard_items_explicit_and_round_robin():
    mats = [{"id": str(i)} for i in range(5)]
    assert [m["id"] for m in shard_items(mats, 1, 2)] == ["1", "3"]  # legacy: materials[shard::nshards]
    packed = [{"id": "a", "shard": 1}, {"id": "b", "shard": 1}, {"id": "c", "shard": 0}]
    assert [m["id"] for m in shard_items(packed, 1, 2)] == ["a", "b"]
    with pytest.raises(ValueError):
        shard_items([{"id": "z", "shard": 2}], 0, 2)


def test_cell_setting_hash_neutral_and_validated(monkeypatch, tmp_path):
    base = pr.protocol_hash(CCEProtocol())
    assert pr.protocol_hash(CCEProtocol(cell_setting=None)) == base
    assert pr.protocol_hash(CCEProtocol(cell_setting="conventional_standard")) != base
    with pytest.raises(ValueError):
        pr.check_cell_setting("primitive")
    seen = {}
    monkeypatch.setitem(sys.modules, "pycce", _fake())
    monkeypatch.setattr(pr, "_load_sites", lambda path, n, **kw: (seen.update(kw), [("Si", np.zeros(3), 1)])[1:] and
                        (None, [("Si", np.zeros(3), 1)]))
    res = run_material(tmp_path / "x.cif", CCEProtocol(**{**BASE, "cell_setting": "conventional_standard"}, n_conf=1,
                                                       b_directions_lattice=((0, 0, 1),)), tmp_path / "o")
    assert seen == {"setting": "conventional_standard"} and np.allclose(res["cell_A"], np.eye(3) * 4.0)
    with pytest.raises(ValueError, match="origin_kanai"):
        run_material(tmp_path / "y.cif", CCEProtocol(site_rule="origin_kanai", cell_setting="conventional_standard"),
                     tmp_path / "o")


def test_read_records_plain_or_gzip(tmp_path):
    """S024: large batches are committed as merged.jsonl.gz; both forms read the same records."""
    import gzip
    import json
    from qkhosts.labels.assemble import read_records
    recs = [{"id": "a", "ok": True}, {"id": "b", "ok": False}]
    text = "\n".join(json.dumps(r) for r in recs) + "\n"
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "merged.jsonl").write_text(text, encoding="utf-8")
    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "merged.jsonl.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
    assert read_records(tmp_path / "p") == read_records(tmp_path / "z") == recs
