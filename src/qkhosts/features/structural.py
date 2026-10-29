"""Structural descriptors (proposal v2, "Structural" feature group; task T-013).

Pure numpy, so they run without pymatgen/ase. A crystal is a `Crystal`: lattice rows a, b, c in Å,
fractional coordinates and element symbols. `Crystal.from_pymatgen` converts a pymatgen Structure.

Descriptors:
- lattice anisotropy: max/min lattice length of the cell given (dimensionless). It depends on the
  cell choice, so pass the conventional standard cell (the pymatgen adapter can do this).
- dimensionality: rank of the periodic bonding network (0 molecule, 1 chain, 2 layer, 3 bulk), from a
  covalent-radius bond graph. The rank method is the one of Larsen et al. / Mounet et al.: walk each
  connected component, and every cycle that closes through a lattice translation adds that translation;
  the component's dimensionality is the rank of those translations.
- nearest-neighbour spinful distances around the central-spin site (a vacancy at a host site, ADR-0004):
  distances (Å) to sites of elements that have a spinful stable isotope, the expected number of spinful
  nuclei within a radius, and a local dipolar second-moment proxy.

Covalent radii are never typed in here: pass them, or let `covalent_radii` read them from ase or
pymatgen (both ship the Cordero et al. 2008 table).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Crystal:
    lattice: np.ndarray        # (3, 3) Å, rows are the lattice vectors
    frac: np.ndarray           # (N, 3) fractional coordinates
    species: tuple[str, ...]   # element symbol per site

    def __post_init__(self):
        lat = np.asarray(self.lattice, float)
        frac = np.asarray(self.frac, float).reshape(-1, 3)
        if lat.shape != (3, 3) or abs(np.linalg.det(lat)) < 1e-9:
            raise ValueError("lattice must be a non-singular 3x3 matrix (Å)")
        if len(frac) != len(self.species):
            raise ValueError("frac and species lengths differ")
        object.__setattr__(self, "lattice", lat)
        object.__setattr__(self, "frac", frac)
        object.__setattr__(self, "species", tuple(self.species))

    @property
    def cart(self) -> np.ndarray:
        """Cartesian coordinates (Å)."""
        return self.frac @ self.lattice

    @classmethod
    def from_pymatgen(cls, structure, conventional: bool = True) -> "Crystal":  # pragma: no cover - needs pymatgen
        if conventional:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        return cls(structure.lattice.matrix, structure.frac_coords,
                   tuple(site.specie.symbol for site in structure))


def covalent_radii(elements: Sequence[str]) -> dict[str, float]:  # pragma: no cover - needs ase or pymatgen
    """Covalent radii (Å) from ase.data (Cordero 2008), falling back to pymatgen's CovalentRadius."""
    try:
        from ase.data import atomic_numbers
        from ase.data import covalent_radii as cr
        return {e: float(cr[atomic_numbers[e]]) for e in elements}
    except ImportError:
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        return {e: float(CovalentRadius.radius[e]) for e in elements}


def lattice_anisotropy(c: Crystal) -> float:
    """max(|a|,|b|,|c|) / min(|a|,|b|,|c|), dimensionless, >= 1."""
    L = np.linalg.norm(c.lattice, axis=1)
    return float(L.max() / L.min())


def _image_range(lattice: np.ndarray, r: float) -> np.ndarray:
    """All integer translations n with enough images to cover every point within r (Å) of the cell."""
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(lattice).T, axis=1)  # interplanar spacings, Å
    m = np.ceil(r / spacing).astype(int) + 1
    g = np.mgrid[-m[0]:m[0] + 1, -m[1]:m[1] + 1, -m[2]:m[2] + 1]
    return g.reshape(3, -1).T


def bond_graph(c: Crystal, radii: Mapping[str, float], scale: float = 1.2) -> list[tuple[int, int, tuple]]:
    """Periodic bonds (i, j, n): site j in cell n is bonded to site i in cell 0 when
    d < scale * (r_i + r_j). `scale` is a heuristic bond tolerance (dimensionless); the dimensionality
    of a host can depend on it, so T-014 checks the strata against it."""
    r = np.array([radii[s] for s in c.species])
    cutoff = scale * 2 * r.max()
    imgs = _image_range(c.lattice, cutoff)
    cart = c.cart
    shifts = imgs @ c.lattice
    edges = []
    for i in range(len(cart)):
        d = np.linalg.norm(cart[None, :, :] + shifts[:, None, :] - cart[i], axis=2)  # (n_img, N)
        ok = d < scale * (r[i] + r[None, :])
        for k, j in zip(*np.nonzero(ok)):
            if i == j and not imgs[k].any():
                continue
            edges.append((i, int(j), tuple(int(x) for x in imgs[k])))
    return edges


def component_dimensionalities(c: Crystal, radii: Mapping[str, float], scale: float = 1.2) -> list[int]:
    """Dimensionality (0-3) of each connected component of the periodic bond graph."""
    adj: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(len(c.species))}
    for i, j, n in bond_graph(c, radii, scale):
        adj[i].append((j, np.array(n)))
    pos: dict[int, np.ndarray] = {}
    dims = []
    for start in adj:
        if start in pos:
            continue
        pos[start] = np.zeros(3, int)
        cycles = []
        q = deque([start])
        while q:
            i = q.popleft()
            for j, n in adj[i]:
                target = pos[i] + n
                if j not in pos:
                    pos[j] = target
                    q.append(j)
                elif (v := target - pos[j]).any():
                    cycles.append(v)
        dims.append(int(np.linalg.matrix_rank(np.array(cycles))) if cycles else 0)
    return dims


def dimensionality(c: Crystal, radii: Mapping[str, float], scale: float = 1.2) -> dict[str, float]:
    """dim_max: highest component dimensionality (3 for a bulk network); dim_min: lowest
    (e.g. 0 for intercalated molecules); is_low_dim: 1.0 when no component spans 3D."""
    d = component_dimensionalities(c, radii, scale)
    return {"dim_max": float(max(d)), "dim_min": float(min(d)), "is_low_dim": float(max(d) < 3)}


def spinful_elements(iso: pd.DataFrame) -> pd.DataFrame:
    """Per element: total abundance of spinful isotopes (fraction) and Σ_isotopes p g² I(I+1)."""
    s = iso[(iso["spin_I"] > 0) & (iso["abundance"] > 0)].copy()
    s["m2"] = s["abundance"] * s["g_factor"] ** 2 * s["spin_I"] * (s["spin_I"] + 1)
    return s.groupby("element").agg(p_spinful=("abundance", "sum"), m2=("m2", "sum"))


def nn_spinful(c: Crystal, center: int, iso: pd.DataFrame, r_count: float = 5.0,
               r_max: float = 15.0, k: int = 3) -> dict[str, float]:
    """Spinful environment of a central spin at site `center`, which is removed (vacancy at a host site).

    Returns, over sites of elements with a spinful stable isotope (within r_max, Å):
    - nn_spinful_d1_A: nearest distance (Å); nn_spinful_dk_mean_A: mean of the k nearest (Å);
    - n_spinful_exp_r: expected number of spinful nuclei within r_count (Σ site abundance);
    - log10_m2_local: log10 Σ_j Σ_iso p g² I(I+1) / r_j⁶ with r in Å (Å⁻⁶, g dimensionless),
      a Van Vleck-like local second-moment proxy for the bath seen by the centre.
    Distances are inf (m2 -inf) when no spinful site lies within r_max.
    """
    el = spinful_elements(iso)
    sp = np.array([s in el.index for s in c.species])
    out = {"nn_spinful_d1_A": np.inf, "nn_spinful_dk_mean_A": np.inf, "n_spinful_exp_r": 0.0,
           "log10_m2_local": -np.inf}
    if not sp.any():
        return out
    imgs = _image_range(c.lattice, r_max)
    cart = c.cart
    d = np.linalg.norm(cart[None, :, :] + (imgs @ c.lattice)[:, None, :] - cart[center], axis=2)
    d[np.all(imgs == 0, axis=1), center] = np.inf  # the removed host atom
    d = d[:, sp]
    names = [s for s, f in zip(c.species, sp) if f]
    p = np.array([el.loc[s, "p_spinful"] for s in names])
    m2 = np.array([el.loc[s, "m2"] for s in names])
    m = d <= r_max
    if not m.any():
        return out
    ds = np.sort(d[m])
    m2_sum = float((np.broadcast_to(m2, d.shape)[m] / d[m] ** 6).sum())
    out.update(nn_spinful_d1_A=float(ds[0]), nn_spinful_dk_mean_A=float(ds[:k].mean()),
               n_spinful_exp_r=float(np.broadcast_to(p, d.shape)[d <= r_count].sum()),
               log10_m2_local=float(np.log10(m2_sum)) if m2_sum > 0 else -np.inf)
    return out


def structural_descriptors(c: Crystal, iso: pd.DataFrame, radii: Mapping[str, float],
                           centers: Sequence[int] | None = None, scale: float = 1.2,
                           r_count: float = 5.0, r_max: float = 15.0) -> dict[str, float]:
    """All structural descriptors for one host. nn_* are averaged over `centers` (default: every
    site, so inequivalent sites count by multiplicity)."""
    centers = range(len(c.species)) if centers is None else centers
    nn = pd.DataFrame([nn_spinful(c, i, iso, r_count, r_max) for i in centers])
    with np.errstate(invalid="ignore"):
        nn_mean = {k: float(nn[k].mean()) for k in nn.columns}
    return {"lattice_anisotropy": lattice_anisotropy(c), **dimensionality(c, radii, scale), **nn_mean}
