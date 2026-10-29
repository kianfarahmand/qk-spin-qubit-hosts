# Kanai et al. 2022 — Zenodo record (fetched in S004, T-004)

- **Record:** "Dataset: Generalized scaling of spin qubit coherence in over 12,000 host materials", doi:[10.5281/zenodo.6323098](https://doi.org/10.5281/zenodo.6323098) (concept DOI 10.5281/zenodo.6323097; the only version, published 2022-03-02).
- **Creators:** Kanai, Heremans, Seo, Wolfowicz, Anderson, Sullivan, Onizhuk, Galli, Awschalom, Ohno.
- **Licence:** CC-BY-4.0. Cite the record and the PNAS paper when using these files.
- **Fetched:** 2026-10-25 by `python scripts/fetch_kanai2022.py`. Each file's MD5 was checked against the Zenodo API, then its SHA-256 written to `SHA256SUMS`. Full file list with Zenodo MD5s and sizes: `zenodo_record.json`.
- **Write-once** (CLAUDE.md §2). Do not edit these files. Every transformation goes in a script under `src/` or `scripts/`. `.gitattributes` marks `data/external/**` as `-text`, so git never changes line endings and the checksums hold on every OS. Check with `sha256sum -c SHA256SUMS` from this folder.

## What is here (28 files)
| File(s) | Content | Notes from inspection (S004) |
|---|---|---|
| `isotopetable.csv` | Isotope table used by the scaling law | 119 isotope rows. Columns: `symbol, #protons, #nutrons [sic], #nucleons, s, g, natural abundance %, electric quadrupole moment (barn)`. `s` is nuclear spin I, `g` the nuclear g-factor. ASCII. |
| `table1.csv` | Paper Table 1 (rank, material, T₂ in ms) | 32 filled rows: ranks 1–27 plus the benchmark rows 138 SiO2, 298 ZnO, 709 SiC (4h), 936 C (diamond), 1125 MgO. The file also has 68 empty rows, 14 empty columns and a UTF-8 BOM; read it with `encoding="utf-8-sig"` and drop empty rows. |
| `T2_Listup_v1.0.py` | Authors' script that scores MP hosts with the scaling law | Uses the legacy `pymatgen.MPRester` API (no longer works). Holds the constants that T-003 checks. Never imported or run here. |
| `*.m` (7 files) | Authors' MATLAB CCE helpers (`load_crystal.m`, `set_crystal_v2.m`, …) | Reference for T-005 (CCE setup). |
| `figure*.csv` (18 files) | Data behind the paper and SI figures | |

**Not fetched** (listed in `zenodo_record.json` with `"fetched": false`): 7 PNG renderings of figures and tables, and `arxiv2102.02986.zip` (6.3 MB, arXiv source). Fetch them by hand if needed.
