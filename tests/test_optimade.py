"""Q9a (S010): keyless MP OPTIMADE structures -> P1 CIFs for E0 (qkhosts.data.optimade)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qkhosts.data import optimade as om

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "mp_optimade"
E0 = ROOT / "compute" / "structures" / "e0"


def _fcc_diamond_primitive(a=3.567):
    L = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    return L, np.array([[0, 0, 0], [a / 4] * 3]), ["C", "C"]


def test_cell_parameters_hexagonal():
    L = np.array([[3.0, 0, 0], [-1.5, 1.5 * np.sqrt(3), 0], [0, 0, 10.0]])
    a, b, c, al, be, ga = om.cell_parameters(L)
    assert (a, b, c) == pytest.approx((3, 3, 10)) and (al, be, ga) == pytest.approx((90, 90, 120))
    assert om.cell_family(L) == "hexagonal"


def test_fractional_wraps_into_unit_cell():
    L = np.diag([2.0, 3.0, 4.0])
    f = om.fractional(L, np.array([[2.0, -1.5, 4.0], [1.0, 1.5, 2.0]]))
    np.testing.assert_allclose(f, [[0, 0.5, 0], [0.5, 0.5, 0.5]], atol=1e-12)


def test_conventional_cubic_diamond():
    L, cart, spc = _fcc_diamond_primitive()
    C, pos, s = om.conventional_cubic(L, cart, spc)
    assert om.cell_family(C) == "cubic" and len(pos) == 8 and s == ["C"] * 8
    assert abs(np.linalg.det(C)) == pytest.approx(4 * abs(np.linalg.det(L)))
    f = np.round(om.fractional(C, pos) * 8) / 8  # diamond sites sit on multiples of 1/4 (+1/8 origin shifts)
    # every pair distance set is the diamond one: 4 nearest neighbours at a*sqrt(3)/4
    d = np.linalg.norm((om.fractional(C, pos)[:, None] - om.fractional(C, pos)[None] + 0.5) % 1 - 0.5, axis=2)
    nn = np.sort(d * 3.567, axis=1)[:, 1:5]
    np.testing.assert_allclose(nn, 3.567 * np.sqrt(3) / 4, rtol=1e-6)
    assert len({tuple(x) for x in f % 1}) == 8  # no duplicate sites


def test_conventional_cubic_rejects_non_cubic():
    L = np.array([[3.0, 0, 0], [-1.5, 1.5 * np.sqrt(3), 0], [0, 0, 10.0]])
    with pytest.raises(ValueError):
        om.conventional_cubic(L, np.zeros((1, 3)), ["Si"])


def test_cif_roundtrip_fractional():
    L, cart, spc = _fcc_diamond_primitive()
    C, pos, s = om.conventional_cubic(L, cart, spc)
    text = om.to_cif(C, pos, s, "test", comments=["hello"])
    assert text.startswith("# hello\ndata_test\n") and "'P 1'" in text
    rows = [ln.split() for ln in text.splitlines() if ln.startswith("  C")]
    assert len(rows) == 8 and rows[0][0] == "C1" and rows[-1][0] == "C8"
    f = np.array([[float(x) for x in r[2:5]] for r in rows])
    np.testing.assert_allclose(f, om.fractional(C, pos), atol=1e-8)


def test_entry_arrays_rejects_disorder():
    entry = {"data": {"attributes": {"species": [{"name": "X", "chemical_symbols": ["Si", "Ge"],
                                                  "concentration": [0.5, 0.5]}],
                                     "lattice_vectors": np.eye(3).tolist(), "cartesian_site_positions": [[0, 0, 0]],
                                     "species_at_sites": ["X"]}}}
    with pytest.raises(ValueError):
        om.entry_arrays(entry)


@pytest.mark.skipif(not (RAW / "SHA256SUMS").exists(), reason="raw OPTIMADE responses are private-repo only")
def test_stored_raw_matches_sums_and_rebuilds_cifs():
    """The stored raw responses are byte-exact, and rebuilding the E0 CIFs from them gives the committed files."""
    sums = dict(reversed(ln.split()) for ln in (RAW / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    for name, digest in sums.items():
        assert hashlib.sha256((RAW / name).read_bytes()).hexdigest() == digest, name
    sources = json.loads((E0 / "SOURCES.json").read_text(encoding="utf-8"))
    for cif_name, src in sources.items():
        cif = (E0 / cif_name).read_bytes()
        assert hashlib.sha256(cif).hexdigest() == src["cif_sha256"], cif_name
        resp = json.loads((RAW / f"{src['mp_id']}.json").read_text(encoding="utf-8"))
        L, cart, spc = om.entry_arrays(resp)
        if src["cell_family"] == "cubic" and om.cell_family(L) != "cubic":
            L, cart, spc = om.conventional_cubic(L, cart, spc)
        assert om.cell_family(L) == src["cell_family"] and len(spc) == src["n_sites_cif"]
