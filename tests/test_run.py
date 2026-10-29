"""Experiment runner (T-030): config merge, overrides, prereg guard, results layout, synthetic E3 end to end."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from qkhosts import run as R

ROOT = Path(__file__).resolve().parents[1]
E3_CFG = ROOT / "configs" / "experiments" / "E3_synthetic.yaml"
FAST = ["splits.primary.repeats=1", "splits.learning_curve_N=[25, max]", "budget.hparam_evals_per_family=3",
        "data.synthetic.n=80"]


def test_deep_merge_overrides_and_keeps_inputs():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 0}
    over = {"a": {"c": [9]}, "e": 1}
    out = R.deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": [9]}, "d": 0, "e": 1}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 0}


def test_apply_set_parses_yaml_values():
    cfg = R.apply_set({"x": {"y": 1}}, "x.z=1e-3")
    cfg = R.apply_set(cfg, "x.w=[1, max]")
    assert cfg == {"x": {"y": 1, "z": 1e-3, "w": [1, "max"]}}
    with pytest.raises(ValueError):
        R.apply_set(cfg, "nokey")


def test_load_config_merges_base_and_records_overrides():
    cfg = R.load_config(E3_CFG, ["budget.hparam_evals_per_family=4"])
    assert cfg["scaling_law"]["alpha"] == -1.65          # from base.yaml
    assert cfg["splits"]["primary"]["repeats"] == 2      # experiment overrides base (5)
    assert cfg["splits"]["primary"]["kind"] == "group_kfold"  # base key kept inside a merged block
    assert cfg["budget"]["hparam_evals_per_family"] == 4
    assert cfg["_provenance"]["set"] == ["budget.hparam_evals_per_family=4"]


def test_prereg_guard_blocks_outer_metrics_on_real_data(tmp_path):
    real = {"id": "E3", "kind": "e3", "data": {"source": "dcce"}, "eval": {"outer": True}}
    with pytest.raises(PermissionError):
        R.check_prereg_guard(real, root=tmp_path)
    R.check_prereg_guard({**real, "eval": {"outer": False}}, root=tmp_path)            # inner CV only: allowed
    R.check_prereg_guard({**real, "data": {"source": "synthetic"}}, root=tmp_path)     # synthetic: allowed
    R.check_prereg_guard({**real, "kind": "smoke"}, root=tmp_path)                     # not E3/E4
    (tmp_path / "prereg").mkdir()
    (tmp_path / "prereg" / "LOCK").write_text("x", encoding="utf-8")
    R.check_prereg_guard(real, root=tmp_path)                                         # locked: allowed


def test_run_dir_names_are_unique(tmp_path):
    now = dt.datetime(2026, 10, 29, 12, 0, 0, tzinfo=dt.timezone.utc)
    a = R.make_run_dir(tmp_path, "E3", "abc1234", True, now)
    b = R.make_run_dir(tmp_path, "E3", "abc1234", True, now)
    assert a.name == "20261029T120000Z_abc1234-dirty" and b.name == a.name + "_2"


def test_minmax_uses_train_statistics():
    Xtr = np.array([[0.0, 5.0], [2.0, 5.0]])
    lo, span = R.minmax_fit(Xtr)
    assert np.allclose(R.minmax_apply(Xtr, lo, span), [[-1, -1], [1, -1]])  # constant column -> no div by 0
    assert np.allclose(R.minmax_apply(np.array([[4.0, 5.0]]), lo, span), [[3, -1]])


def test_synthetic_dataset_is_deterministic():
    a, b = R.synthetic_dataset(40, 4, 8, 0.05, 3), R.synthetic_dataset(40, 4, 8, 0.05, 3)
    assert np.array_equal(a["y"], b["y"]) and np.array_equal(a["groups"], b["groups"])


def test_e3_synthetic_end_to_end(tmp_path):
    cfg = R.load_config(E3_CFG, FAST)
    out = R.run(cfg, out_root=tmp_path, argv=["test"])
    assert out.parent == tmp_path / "E3_synthetic"
    for f in ("config.yaml", "metrics.json", "env.txt"):
        assert (out / f).exists()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    pred_file = out / m["predictions"]
    preds = pd.read_parquet(pred_file) if pred_file.suffix == ".parquet" else pd.read_csv(pred_file)

    models = {"B0_zero", "rbf", "matern52", "linear"}
    assert {r["model"] for r in m["summary"]} == models
    assert {r["N"] for r in m["summary"]} == {"25", "max"}
    assert all(r["J"] == 5 and r["n_failed"] == 0 for r in m["summary"])   # 1 repeat x 5 folds
    # equal budget: each fitted family used min(budget, grid size) evaluations
    fitted = [r for r in m["runs"] if r["model"] != "B0_zero"]
    assert all(r["n_evals"] == 3 for r in fitted)
    # every material appears once per (model, N) in the outer test folds of a repeat
    per = preds.groupby(["model", "N_label"])["id"].agg(["count", "nunique"])
    assert (per["count"] == 80).all() and (per["nunique"] == 80).all()
    # the saved config is the resolved one, overrides included
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["budget"]["hparam_evals_per_family"] == 3 and saved["_provenance"]["set"] == FAST
    assert "numpy" in (out / "env.txt").read_text(encoding="utf-8")


def test_e3_is_reproducible(tmp_path):
    cfg = R.load_config(E3_CFG, FAST)
    m1, _ = R.e3(cfg)
    m2, _ = R.e3(cfg)
    assert m1["summary"] == m2["summary"]


def test_e3_inner_only_writes_no_outer_metrics(tmp_path):
    cfg = R.load_config(E3_CFG, FAST + ["eval.outer=false"])
    m, preds = R.e3(cfg)
    assert preds.empty and m["outer_eval"] is False
    assert all("mae" not in r for r in m["runs"])
    assert {r["metric"] for r in m["summary"]} == {"inner_cv_mae"}


def test_e3_records_failures_instead_of_dropping(monkeypatch):
    cfg = R.load_config(E3_CFG, FAST)
    real = R._family_table()

    def broken(kp, A, B):
        raise FloatingPointError("boom")
    monkeypatch.setattr(R, "_family_table", lambda: {**real, "linear": (broken, real["linear"][1])})
    m, _ = R.e3(cfg)
    lin = [r for r in m["runs"] if r["model"] == "linear"]
    assert lin and all(r["status"] == "error" and "boom" in r["error"] for r in lin)
    assert all(r["n_failed"] == 5 for r in m["summary"] if r["model"] == "linear")


def test_unknown_kind_and_smoke(tmp_path):
    with pytest.raises(NotImplementedError):
        R.run({"id": "X", "kind": "nope"}, out_root=tmp_path)
    cfg = R.load_config(ROOT / "configs" / "experiments" / "E_smoke.yaml")
    out = R.run(cfg, out_root=tmp_path, argv=["test"])
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(m["results"]) == {"qk_iqp", "rbf"}


def test_cli_main(tmp_path, capsys):
    assert R.main(["--config", str(E3_CFG), "--out-root", str(tmp_path), *sum((["--set", s] for s in FAST), [])]) == 0
    assert "B0_zero" in capsys.readouterr().out
