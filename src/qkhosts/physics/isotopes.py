"""Isotope table handling.

The table is NOT typed in from memory. It is loaded from a file (preferred source:
Kanai et al. Zenodo `isotopetable.csv`, task T-004) and normalised to columns:
    element, isotope, abundance (fraction 0-1), spin_I, g_factor (nuclear g, dimensionless)
`load_isotope_table` validates units/ranges so silent unit errors are caught.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("element", "isotope", "abundance", "spin_I", "g_factor")


def normalise(df: pd.DataFrame, colmap: dict[str, str] | None = None) -> pd.DataFrame:
    if colmap:
        df = df.rename(columns=colmap)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"isotope table missing columns {missing}; pass colmap")
    df = df[list(REQUIRED)].copy()
    if df["abundance"].max() > 1.0 + 1e-9:  # given in percent
        df["abundance"] = df["abundance"] / 100.0
    if (df["abundance"] < 0).any() or (df["spin_I"] < 0).any():
        raise ValueError("negative abundance or spin")
    sums = df.groupby("element")["abundance"].sum()
    bad = sums[(sums - 1.0).abs() > 0.02]
    if len(bad):
        raise ValueError(f"abundances do not sum to ~1 for: {list(bad.index)[:10]}")
    return df.reset_index(drop=True)


def load_isotope_table(path: str | Path, colmap: dict[str, str] | None = None) -> pd.DataFrame:
    return normalise(pd.read_csv(path), colmap)


def spinful(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["spin_I"] > 0) & (df["abundance"] > 0)]
