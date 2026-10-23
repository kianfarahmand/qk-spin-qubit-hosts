# Data layout
- `raw/`: as downloaded, never edited (gitignored; record the source and hash in DATA_CARD)
- `external/`: third-party curated files that are small enough to commit (e.g. the Kanai isotope table), with `SHA256SUMS`
- `interim/`: intermediate files (gitignored)
- `processed/`: versioned datasets used by experiments (`dsl.parquet`, `dcce.parquet`), each with a DATA_CARD entry
