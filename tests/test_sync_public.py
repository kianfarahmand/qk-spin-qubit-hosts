"""scripts/sync_public.py deny rules: private paths never reach the public repo; exact exemptions only (Q17)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_public", ROOT / "scripts" / "sync_public.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def test_private_paths_denied():
    for rel in ["sessions/2026-10-30_S015_e0b-final.md", "STATUS.md", "tasks/tasks.yaml", "proposal/proposal_v2.md",
                "paper/main.tex", "prereg/preregistration.md", ".env", "CLAUDE.md", ".claude/commands/x.md",
                "literature/notes/kanai2022_si.md", "decisions/ADR-0001-target-delta-learning.md",
                "decisions/ADR-0004-cce-label-protocol.md.bak", "decisions/README.md"]:
        assert sp.denied(rel), rel


def test_exact_exemption_only():
    assert sp.ALLOW_EXACT == {"decisions/ADR-0004-cce-label-protocol.md"}
    assert not sp.denied("decisions/ADR-0004-cce-label-protocol.md")
    assert not sp.denied("experiments/E0_notes.md") and not sp.denied("src/qkhosts/labels/pycce_runner.py")
