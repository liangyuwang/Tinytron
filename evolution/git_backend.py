from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .safety import (
    EvolutionSafetyError,
    ensure_git_repo,
    require_clean_worktree,
    validate_experiment_branch,
)


class GitCommandError(RuntimeError):
    """Raised when a git command fails while preparing experiment state."""


@dataclass(frozen=True)
class GitExperimentState:
    spec_id: str
    repo_path: str
    base_commit: str
    candidate_commit: str
    infra_commit: str
    branch: str
    worktree_path: str
    diff_hash: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class GitBackend:
    """Git-backed isolation layer for auto-evolution experiments."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = ensure_git_repo(repo_path)

    def prepare_experiment(
        self,
        spec_id: str,
        *,
        base_ref: str = "HEAD",
        branch: str | None = None,
        worktree_root: str | Path | None = None,
        require_clean_repo: bool = True,
    ) -> GitExperimentState:
        if require_clean_repo:
            require_clean_worktree(self.repo_path)

        branch = branch or f"exp/{_safe_ref_part(spec_id)}"
        validate_experiment_branch(branch)

        base_commit = self.resolve_commit(base_ref)
        infra_commit = self.resolve_commit("HEAD")
        root = Path(worktree_root) if worktree_root is not None else (
            self.repo_path / ".evolution" / "worktrees"
        )
        worktree_path = root / _safe_path_part(spec_id)
        if worktree_path.exists():
            raise EvolutionSafetyError(f"experiment worktree already exists: {worktree_path}")

        root.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(worktree_path), base_commit)
        candidate_commit = self._git_at(worktree_path, "rev-parse", "HEAD")
        return GitExperimentState(
            spec_id=spec_id,
            repo_path=str(self.repo_path),
            base_commit=base_commit,
            candidate_commit=candidate_commit,
            infra_commit=infra_commit,
            branch=branch,
            worktree_path=str(worktree_path),
            diff_hash=self.diff_hash(worktree_path, base_commit),
        )

    def snapshot(
        self,
        spec_id: str,
        *,
        worktree_path: str | Path,
        base_commit: str,
        branch: str | None = None,
        infra_commit: str | None = None,
    ) -> GitExperimentState:
        worktree = ensure_git_repo(worktree_path)
        branch = branch or self._git_at(worktree, "branch", "--show-current")
        validate_experiment_branch(branch)
        return GitExperimentState(
            spec_id=spec_id,
            repo_path=str(self.repo_path),
            base_commit=self.resolve_commit(base_commit),
            candidate_commit=self._git_at(worktree, "rev-parse", "HEAD"),
            infra_commit=infra_commit or self.resolve_commit("HEAD"),
            branch=branch,
            worktree_path=str(worktree),
            diff_hash=self.diff_hash(worktree, base_commit),
        )

    def resolve_commit(self, ref: str) -> str:
        return self._git("rev-parse", "--verify", f"{ref}^{{commit}}")

    def diff_hash(self, worktree_path: str | Path, base_commit: str) -> str:
        worktree = Path(worktree_path)
        committed = self._git_at_bytes(worktree, "diff", "--binary", base_commit, "HEAD")
        uncommitted = self._git_at_bytes(worktree, "diff", "--binary")
        staged = self._git_at_bytes(worktree, "diff", "--binary", "--cached")
        digest = hashlib.sha256()
        digest.update(committed)
        digest.update(b"\0--staged--\0")
        digest.update(staged)
        digest.update(b"\0--worktree--\0")
        digest.update(uncommitted)
        return digest.hexdigest()

    def _git(self, *args: str) -> str:
        return self._git_at(self.repo_path, *args)

    def _git_at(self, cwd: str | Path, *args: str) -> str:
        return self._git_at_bytes(cwd, *args).decode().strip()

    def _git_at_bytes(self, cwd: str | Path, *args: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace").strip()
            command = " ".join(["git", "-C", str(cwd), *args])
            raise GitCommandError(f"{command} failed: {stderr}")
        return completed.stdout


def _safe_ref_part(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "unnamed"


def _safe_path_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("._") or "unnamed"
