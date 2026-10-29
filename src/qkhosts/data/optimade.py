"""Keyless structure fetch from the Materials Project OPTIMADE endpoint (Q9a, S010).

The MP OPTIMADE API (https://optimade.materialsproject.org/v1) serves the same mp-ids as the keyed
MP API, without an API key. Each entry gives `lattice_vectors` (rows, Å), `cartesian_site_positions`
(Å) and `species_at_sites`. Its space-group fields are null, so callers check the cell family and
site count instead; pymatgen's SpacegroupAnalyzer confirms symmetry when the CCE runner loads the CIF.

The CIFs written here are P1 (all sites listed, fractional coordinates). pymatgen rebuilds the
lattice from (a, b, c, α, β, γ) with c along Cartesian z, so `b_direction (0, 0, 1)` is the c axis,
and for a conventional cubic cell `(1, 1, 1)` is the crystal [111] (compute/drafts/*.yaml).
Needs numpy and the standard library only.
"""
from __future__ import annotations

import itertools
import json
import urllib.request

import numpy as np

BASE_URL = "https://optimade.materialsproject.org/v1"


def fetch_entry(mp_id: str, base_url: str = BASE_URL, timeout: float = 60.0) -> dict:  # pragma: no cover - network
    """Return the full OPTIMADE response (`data` + `meta`) for one structure id, e.g. 'mp-66'.
    Sends an explicit User-Agent: the endpoint's Cloudflare front returns 403 to urllib's default one."""
    req = urllib.request.Request(f"{base_url}/structures/{mp_id}", headers={
        "User-Agent": "qkhosts/0 (+https://github.com/kianfarahmand/qk-spin-qubit-hosts)",
        "Accept": "application/vnd.api+json, application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
        return json.loads(r.read().decode("utf-8"))


def cell_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """(a, b, c) in Å and (α, β, γ) in degrees from lattice vectors given as rows (Å)."""
    L = np.asarray(lattice, float)
    a, b, c = np.linalg.norm(L, axis=1)

    def ang(u, v):
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0))))

    return float(a), float(b), float(c), ang(L[1], L[2]), ang(L[0], L[2]), ang(L[0], L[1])


def fractional(lattice: np.ndarray, cart: np.ndarray) -> np.ndarray:
    """Fractional coordinates in [0, 1) of Cartesian positions (Å) for row lattice vectors (Å)."""
    f = np.asarray(cart, float) @ np.linalg.inv(np.asarray(lattice, float))
    f = f - np.floor(f)
    f[np.isclose(f, 1.0, atol=1e-8)] = 0.0
    return f


def cell_family(lattice: np.ndarray, tol_len: float = 1e-3, tol_deg: float = 0.05) -> str:
    """'cubic', 'hexagonal' (a = b, α = β = 90°, γ = 120°) or 'other'. tol_len is relative."""
    a, b, c, al, be, ga = cell_parameters(lattice)

    def eq(x, y):
        return abs(x - y) <= tol_len * max(x, y)

    if eq(a, b) and eq(b, c) and all(abs(x - 90) < tol_deg for x in (al, be, ga)):
        return "cubic"
    if eq(a, b) and abs(al - 90) < tol_deg and abs(be - 90) < tol_deg and abs(ga - 120) < tol_deg:
        return "hexagonal"
    return "other"


def conventional_cubic(lattice: np.ndarray, cart: np.ndarray, species: list[str],
                       max_index: int = 1) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert a primitive cell of a cubic lattice (fcc, bcc, sc) into a conventional cubic cell.

    Searches integer matrices M (entries in [-max_index, max_index], det > 0) for new rows M @ L that
    are mutually orthogonal with equal lengths, picks the smallest such cell, and fills it with the
    lattice-translated images of the input sites. Returns (lattice rows Å, Cartesian Å, species).
    Raises ValueError if no cubic supercell is found."""
    L = np.asarray(lattice, float)
    cart = np.asarray(cart, float)
    rng = range(-max_index, max_index + 1)
    best = None
    for row in itertools.product(itertools.product(rng, repeat=3), repeat=3):
        M = np.array(row, int)
        det = round(float(np.linalg.det(M)))
        if det <= 0 or (best is not None and det >= best[0]):
            continue
        if cell_family(M @ L) == "cubic":
            best = (det, M)
    if best is None:
        raise ValueError("no conventional cubic cell found")
    det, M = best
    C = M @ L
    # images: shifts n of the old lattice that land inside the new cell (|n| <= det suffices)
    pos, spc = [], []
    span = range(-det, det + 1)
    for n in itertools.product(span, span, span):
        shifted = cart + np.asarray(n, float) @ L
        f = shifted @ np.linalg.inv(C)
        keep = np.all((f > -1e-6) & (f < 1 - 1e-6), axis=1)
        for i in np.flatnonzero(keep):
            pos.append(shifted[i])
            spc.append(species[i])
    pos = np.asarray(pos)
    if len(pos) != det * len(cart):
        raise ValueError(f"expected {det * len(cart)} sites in the cubic cell, got {len(pos)}")
    return C, pos, spc


def to_cif(lattice: np.ndarray, cart: np.ndarray, species: list[str], data_name: str,
           comments: list[str] = ()) -> str:
    """P1 CIF text for row lattice vectors (Å) and Cartesian sites (Å). Labels are <element><index>."""
    a, b, c, al, be, ga = cell_parameters(lattice)
    f = fractional(lattice, cart)
    out = [f"# {line}" for line in comments]
    out += [f"data_{data_name}", "_symmetry_space_group_name_H-M   'P 1'", "_symmetry_Int_Tables_number   1",
            f"_cell_length_a   {a:.8f}", f"_cell_length_b   {b:.8f}", f"_cell_length_c   {c:.8f}",
            f"_cell_angle_alpha   {al:.8f}", f"_cell_angle_beta   {be:.8f}", f"_cell_angle_gamma   {ga:.8f}",
            "loop_", " _symmetry_equiv_pos_as_xyz", "  'x, y, z'",
            "loop_", " _atom_site_label", " _atom_site_type_symbol",
            " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy"]
    count: dict[str, int] = {}
    for el, (x, y, z) in zip(species, f):
        count[el] = count.get(el, 0) + 1
        out.append(f"  {el}{count[el]}  {el}  {x:.8f}  {y:.8f}  {z:.8f}  1")
    return "\n".join(out) + "\n"


def entry_arrays(entry: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(lattice rows Å, Cartesian sites Å, species) from an OPTIMADE structure response or its `data`.
    Raises ValueError for partial occupancy or mixed species (CCE needs ordered hosts)."""
    data = entry.get("data", entry)
    a = data["attributes"]
    for s in a["species"]:
        if s["chemical_symbols"] != [s["name"]] or s["concentration"] != [1.0]:
            raise ValueError(f"disordered species {s['name']!r} not supported")
    return (np.asarray(a["lattice_vectors"], float), np.asarray(a["cartesian_site_positions"], float),
            list(a["species_at_sites"]))
