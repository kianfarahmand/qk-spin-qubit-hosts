"""Pilot check helpers (S022): lattice parameters from the runner's cell_A (columns) and the host log-mean."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "scripts" / "check_e2_pilot.py").is_file():  # private-only script
    pytest.skip("scripts/check_e2_pilot.py not present", allow_module_level=True)
spec = importlib.util.spec_from_file_location("check_e2_pilot", ROOT / "scripts" / "check_e2_pilot.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_lattice_families():
    assert g.lattice((5.69 * np.eye(3)).tolist())["family"] == "cubic"
    t = g.lattice(np.diag([5.24, 5.24, 11.38]).tolist())
    assert t["family"] == "tetragonal" and t["c"] == pytest.approx(11.38)
    a, c = 10.4, 37.2
    rows = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    h = g.lattice(rows.T.tolist())  # columns, as pycce BathCell.cell
    assert h["family"] == "hexagonal" and h["gamma"] == pytest.approx(120) and h["c"] == pytest.approx(c)
    # an fcc primitive cell is not reported as cubic
    p = g.lattice((0.5 * 5.69 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])).T.tolist())
    assert p["family"] == "other" and p["alpha"] == pytest.approx(60)


def test_log_mean():
    assert g.log_mean([10.0, 1000.0]) == pytest.approx(100.0)
    assert np.isnan(g.log_mean([10.0, float("nan")]))
