"""Exercise browser event handlers without installing a frontend framework."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_inspector_regressions() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the dependency-free UI regression checks")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, "tests/ui/inspector.cjs"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
