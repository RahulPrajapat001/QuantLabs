"""Thin, failure-tolerant git helpers.

Two parts of the integrity layer need git: the research log records the commit a
run executed at, and pre-registration proves from history that a hypothesis was
committed *before* it was tested. Both need the same behaviour — never raise
because git is missing or the tree is not a repository, just return ``None`` and
let the caller decide whether that is fatal (it is, for a holdout run).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

#: Anchor git calls to the installed package rather than the process's working
#: directory, so a run launched from anywhere still resolves the source tree.
_ANCHOR = Path(__file__).resolve().parent


def run_git(args: Sequence[str], *, cwd: Path | None = None) -> str | None:
    """Run ``git <args>`` and return stripped stdout, or ``None`` on any failure."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd or _ANCHOR,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


@lru_cache(maxsize=1)
def repo_root() -> Path | None:
    """The git work-tree root, or ``None`` outside a repository."""

    output = run_git(["rev-parse", "--show-toplevel"])
    if not output:
        return None
    return Path(output)


def relative_to_repo(path: Path) -> str | None:
    """``path`` as a repo-relative POSIX path, for ``git show`` and ``git log``."""

    root = repo_root()
    if root is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
