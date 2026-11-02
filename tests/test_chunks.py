"""Configuration-level chunks (T-015, S019): plan, merge across different windows, failure rule, runner flag."""
import numpy as np
import pytest

from qkhosts.labels import pycce_runner as pr
from qkhosts.labels.chunks import chunk_plan, merge_chunks


def _chunk(t2s, t_max, n_t=201, errors=()):
    ts = np.linspace(0.0, t_max, n_t)
    curves = [np.exp(-(ts / t2) ** 2.5) for t2 in t2s]
    per = [pr.fit_t2(ts, c)[0] for c in curves]
    return {"curve_t_ms": ts.tolist(), "conf_curves": [c.tolist() for c in curves], "per_conf_t2_ms": per,
            "conf_errors": list(errors)}


def test_chunk_plan():
    assert chunk_plan(20, 5) == [(0, 5), (5, 5), (10, 5), (15, 5)]
    assert chunk_plan(7, 3, seed=2) == [(2, 3), (5, 3), (8, 1)]
    with pytest.raises(ValueError):
        chunk_plan(20, 0)


def test_merge_matches_single_run_on_one_grid():
    t2s = np.linspace(0.8, 1.2, 10)
    whole = _chunk(t2s, 3.0)
    ts = np.asarray(whole["curve_t_ms"])
    direct = pr.fit_t2(ts, np.mean(whole["conf_curves"], axis=0))[0]
    m = merge_chunks([_chunk(t2s[:5], 3.0), _chunk(t2s[5:], 3.0)])
    assert m["t2_ms"] == pytest.approx(direct, rel=1e-12) and m["n_conf_ok"] == 10 and m["n_chunks"] == 2
    assert m["t2_ms_mean_of_fits"] == pytest.approx(np.mean(whole["per_conf_t2_ms"]))


def test_merge_across_windows_uses_shortest_and_interpolates():
    t2s = np.linspace(0.8, 1.2, 10)
    whole = _chunk(t2s, 3.0)
    direct = pr.fit_t2(np.asarray(whole["curve_t_ms"]), np.mean(whole["conf_curves"], axis=0))[0]
    m = merge_chunks([_chunk(t2s[:5], 3.0), _chunk(t2s[5:], 4.5)])
    assert m["curve_t_ms"][-1] == pytest.approx(3.0)
    assert m["t2_ms"] == pytest.approx(direct, rel=2e-3)  # linear interpolation error only


def test_merge_failure_rule_and_missing_curves():
    a = _chunk([1.0] * 9, 3.0, errors=[{"conf": 9}])
    assert np.isfinite(merge_chunks([a], n_conf_requested=10)["t2_ms"])  # 1/10 failed: allowed (<= 10 %)
    b = _chunk([1.0] * 3, 3.0, errors=[{"conf": 3}, {"conf": 4}])
    m = merge_chunks([a, b], n_conf_requested=15)  # 3/15 failed
    assert np.isnan(m["t2_ms"]) and m["fit_flag"] == "conf_errors"
    with pytest.raises(ValueError, match="conf_curves"):
        merge_chunks([{"curve_t_ms": [0, 1], "per_conf_t2_ms": []}])


def test_store_conf_curves_flag_keeps_hash_when_off():
    base = pr.protocol_hash(pr.CCEProtocol())
    assert pr.protocol_hash(pr.CCEProtocol(store_conf_curves=False)) == base
    assert pr.protocol_hash(pr.CCEProtocol(store_conf_curves=True)) != base
