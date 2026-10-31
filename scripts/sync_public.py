#!/usr/bin/env python3
"""Copy the allow-listed files into a local clone of the public repo.

    python scripts/sync_public.py ../qk-spin-qubit-hosts [--dry-run]

Rules: only paths listed in public/MANIFEST (globs allowed); public/overlay/* is copied to the
public repo root (README, LICENSE, CITATION.cff). Anything matching DENY is refused even if listed.
Commit/push in the public clone is left to the caller (review the diff first).
"""
from __future__ import annotations

import fnmatch
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "public" / "MANIFEST"
OVERLAY = ROOT / "public" / "overlay"
DENY = ["sessions/*", "STATUS.md", "tasks/*", "proposal/*", "paper/*", "prereg/preregistration.md",
        "*.env", ".env*", "*secret*", "*token*", "decisions/*", "literature/*", "CLAUDE.md", ".claude/*"]
# Exact paths exempt from DENY, each approved by Kian (and still listed in MANIFEST to be copied).
ALLOW_EXACT = {"decisions/ADR-0004-cce-label-protocol.md"}  # Q15 + Q17 (2026-10-31): the CCE label protocol


def denied(rel: str) -> bool:
    return rel not in ALLOW_EXACT and any(fnmatch.fnmatch(rel, pat) for pat in DENY)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    dest = Path(sys.argv[1]).resolve()
    dry = "--dry-run" in sys.argv
    if not (dest / ".git").exists():
        print(f"{dest} is not a git clone", file=sys.stderr)
        return 1
    pats = [l.strip() for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    files: set[Path] = set()
    for pat in pats:
        files |= {p for p in ROOT.glob(pat) if p.is_file()}
    n = 0
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        if denied(rel) or "__pycache__" in rel:
            print("DENY ", rel)
            continue
        print("copy ", rel)
        n += 1
        if not dry:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / rel)
    for p in sorted(OVERLAY.rglob("*")):
        if p.is_file():
            rel = p.relative_to(OVERLAY).as_posix()
            print("overlay", rel)
            n += 1
            if not dry:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest / rel)
    print(f"{n} file(s) {'would be ' if dry else ''}synced to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
