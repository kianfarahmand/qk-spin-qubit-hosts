"""S025: T-071 cap-audit statistics (paired configurations, class summary with the 0.02 dex escalation rule)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from t071_cap_audit import class_stats, conf_t2, direction_items, paired_conf_stats  # noqa: E402


def test_paired_conf_stats_uniform_ratio():
    lo = [0.1, 0.2, 0.4]
    hi = [x * 10 ** -0.02 for x in lo]
    s = paired_conf_stats(hi, lo)
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(-0.02, abs=1e-12)
    assert s["sd"] == pytest.approx(0.0, abs=1e-12)


def test_paired_conf_stats_drops_bad_pairs_and_checks_length():
    s = paired_conf_stats([1.0, float("nan"), 2.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(np.log10(2) / 2)
    with pytest.raises(ValueError):
        paired_conf_stats([1.0], [1.0, 2.0])


def test_class_stats_escalation_flag():
    calm = class_stats(pd.Series([-0.01, -0.012, -0.014, -0.011]))
    assert not calm["escalate"] and calm["n"] == 4
    assert calm["mean"] == pytest.approx(-0.01175)
    wild = class_stats(pd.Series([-0.06, 0.0, -0.03, 0.02]))
    assert wild["sd"] > 0.02 and wild["escalate"]
    one = class_stats(pd.Series([-0.01, float("nan")]))
    assert one["n"] == 1 and not one["escalate"]


def test_conf_t2_orders_chunks_by_seed_and_direction_filter():
    req = {"materials": [
        {"id": "h-d0-k1", "host": "h", "protocol": {"b_directions_lattice": [[0, 0, 1]], "seed": 5, "n_conf": 2}},
        {"id": "h-d0-k0", "host": "h", "protocol": {"b_directions_lattice": [[0, 0, 1]], "seed": 0, "n_conf": 5}},
        {"id": "h-d1", "host": "h", "protocol": {"b_directions_lattice": [[1, 1, 0]]}},
        {"id": "g-d0", "host": "g", "protocol": {"b_directions_lattice": [[0, 0, 1]]}}]}
    recs = {"h-d0-k0": {"sites": [{"final": {"per_conf_t2_ms": [1, 2, 3, 4, 5]}}]},
            "h-d0-k1": {"sites": [{"final": {"per_conf_t2_ms": [6, 7]}}]}}
    items = direction_items(req, "h")
    assert sorted(i["id"] for i in items) == ["h-d0-k0", "h-d0-k1"]
    assert conf_t2(items, recs) == [1, 2, 3, 4, 5, 6, 7]
