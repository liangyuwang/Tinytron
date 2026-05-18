from __future__ import annotations

import subprocess
from pathlib import Path


class EvolutionSafetyError(RuntimeError):
    """Raised when an evolution operation would violate isolation rules."""


def ensure_git_repo(path: str | Path) -> Path:
    repo = Path(path)
    result = _git(repo, "rev-parse", "--is-inside-work-tree")
    if result != "true":
        raise EvolutionSafetyError(f"{repo} is not inside a git worktree")
    return repo


def require_clean_worktree(path: str | Path) -> None:
    repo = ensure_git_repo(path)
    status = _git(repo, "status", "--porcelain")
    if status:
        raise EvolutionSafetyError(f"git worktree is dirty: {repo}")


def validate_experiment_branch(branch: str) -> None:
    if not branch.startswith("exp/"):
        raise EvolutionSafetyError(
            f"experiment branch must use the 'exp/' namespace: {branch!r}"
        )


def validate_promotion_ref(ref: str) -> None:
    allowed = ref.startswith("promoted/") or ref.startswith("research/")
    if not allowed:
        raise EvolutionSafetyError(
            f"promotion target must use 'promoted/' or 'research/' namespace: {ref!r}"
        )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        command = " ".join(["git", "-C", str(repo), *args])
        raise EvolutionSafetyError(f"{command} failed: {stderr}")
    return completed.stdout.strip()
