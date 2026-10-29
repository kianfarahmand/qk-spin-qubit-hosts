"""T-009: scripts must read/write text as UTF-8 regardless of the platform default (Windows: cp1252)."""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# -X warn_default_encoding (PEP 597) + -W error turns any implicit-encoding open() into a failure,
# so this test catches the Windows bug on Linux CI too.
STRICT = [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning"]


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    # PYTHONIOENCODING only fixes the console streams, not the default for files under test.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("PYTHONUTF8", None)
    return subprocess.run([*STRICT, str(root / "scripts" / "session.py"), *args],
                          cwd=root, capture_output=True, text=True, encoding="utf-8", env=env)


@pytest.mark.skipif(not (ROOT / "scripts" / "session.py").exists() or not (ROOT / "sessions" / "TEMPLATE.md").exists(),
                    reason="session system is private-repo only (not in public/MANIFEST)")
def test_session_start_end_roundtrip_utf8(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "sessions").mkdir()
    shutil.copy2(ROOT / "scripts" / "session.py", tmp_path / "scripts" / "session.py")
    template = (ROOT / "sessions" / "TEMPLATE.md").read_text(encoding="utf-8")
    assert any(ord(c) > 127 for c in template)  # the template has non-ASCII (em dash, middle dot)
    (tmp_path / "sessions" / "TEMPLATE.md").write_text(template, encoding="utf-8")
    (tmp_path / "sessions" / "INDEX.md").write_text("| ID | Date | Where | Slug | Summary |\n|---|---|---|---|---|\n",
                                                   encoding="utf-8")

    r = _run(tmp_path, "start", "ümlaut — test", "--where", "claude-code")
    assert r.returncode == 0, r.stderr
    (log,) = (tmp_path / "sessions").glob("*_S001_*.md")
    text = log.read_bytes().decode("utf-8")  # strict decode: fails on cp1252 bytes
    assert text.startswith("# Session S001 — ") and "ümlaut — test" in text and "  ·  " in text

    r = _run(tmp_path, "end", "--summary", "Δ done — ok")
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "sessions" / "ACTIVE").exists()
    assert "**Ended:** _" not in log.read_bytes().decode("utf-8")
    last = (tmp_path / "sessions" / "INDEX.md").read_bytes().decode("utf-8").splitlines()[-1]
    assert last.startswith("| [S001](") and last.endswith("| Δ done — ok |")


def _implicit_encoding_calls(path: Path) -> list[str]:
    """Return `file:line` for read_text/write_text/open calls that do not pass encoding=."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name not in {"read_text", "write_text", "open"}:
            continue
        kws = {k.arg for k in node.keywords}
        modes = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if "encoding" in kws or any("b" in m for m in modes if name == "open"):
            continue
        bad.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    return bad


def test_no_implicit_encoding_in_scripts_and_src():
    files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
    bad = [hit for f in files for hit in _implicit_encoding_calls(f)]
    assert not bad, f"pass encoding='utf-8': {bad}"
