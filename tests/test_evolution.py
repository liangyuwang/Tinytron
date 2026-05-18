from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


def sample_spec_dict():
    return {
        "id": "router-entropy-v1",
        "objective": "Improve small-model routing stability without general regression.",
        "parent_model": "model_n",
        "candidate_model": "model_n_candidate",
        "model": {
            "base": "tiny-gpt-moe-v1",
            "changes": [
                {
                    "id": "router_entropy_penalty_v1",
                    "target": "moe.router",
                    "type": "loss_term",
                    "parameters": {"weight": 0.002},
                }
            ],
        },
        "training": {
            "stage": "rl",
            "optimizer": {"name": "adamw", "parameters": {"lr": 3e-5}},
            "schedule": {"type": "cosine", "parameters": {"warmup_steps": 100}},
            "losses": [{"name": "grpo", "parameters": {"clip_range": 0.2}}],
            "budget": {"max_steps": 1000, "tokens": 2000000},
        },
        "data": {
            "sources": [
                {
                    "name": "agent_math_selfplay",
                    "type": "generated_trajectory",
                    "generator_model": "model_n",
                    "verifier": "symbolic_math_v1",
                }
            ],
            "mixture": [{"source": "agent_math_selfplay", "weight": 1.0}],
        },
        "eval": {
            "baseline_model": "model_n",
            "candidate_model": "model_n_candidate",
            "suites": [
                {"name": "math_holdout", "metric": "accuracy", "min_delta": 0.02},
                {"name": "general_regression", "metric": "win_rate", "min_delta": -0.005},
            ],
            "promotion": {
                "rule": "all_required_pass",
                "required": ["math_holdout", "general_regression"],
            },
        },
        "evidence": {
            "status": "promote_to_large_scale_trial",
            "confidence": "medium",
            "risks": ["needs large-scale validation"],
        },
    }


class EvolutionSpecTest(unittest.TestCase):
    def test_spec_round_trips_json(self) -> None:
        from evolution import EvolutionSpec

        spec = EvolutionSpec.from_dict(sample_spec_dict())
        loaded = EvolutionSpec.from_json(spec.to_json())

        self.assertEqual(loaded.id, spec.id)
        self.assertEqual(loaded.model.changes[0].id, "router_entropy_penalty_v1")
        self.assertEqual(loaded.training.optimizer.parameters["lr"], 3e-5)

    def test_validation_rejects_unknown_data_mixture_source(self) -> None:
        from evolution import EvolutionSpec, SpecValidationError

        data = sample_spec_dict()
        data["data"]["mixture"] = [{"source": "missing", "weight": 1.0}]

        with self.assertRaisesRegex(SpecValidationError, "unknown source"):
            EvolutionSpec.from_dict(data)

    def test_promotion_gate_requires_required_suites(self) -> None:
        from evolution import EvalResult, EvolutionSpec, decide_promotion

        spec = EvolutionSpec.from_dict(sample_spec_dict())
        decision = decide_promotion(
            spec.eval,
            [
                EvalResult("math_holdout", baseline=0.50, candidate=0.53),
                EvalResult("general_regression", baseline=0.60, candidate=0.598),
            ],
        )

        self.assertTrue(decision.promoted)

    def test_tinytron_translator_exposes_future_adapter_interface(self) -> None:
        from evolution import EvolutionSpec, TinytronTranslator

        spec = EvolutionSpec.from_dict(sample_spec_dict())
        artifact = TinytronTranslator().translate(spec)

        self.assertEqual(artifact.framework, "tinytron")
        self.assertEqual(artifact.config["training"]["stage"], "rl")
        self.assertIn("bash scripts/debug/rl.sh", artifact.commands)
        self.assertTrue(artifact.is_complete)

    def test_public_api_and_registry_support_agent_control_loop(self) -> None:
        from pathlib import Path

        from evolution import (
            EvalResult,
            EvolutionRegistry,
            EvolutionSpec,
            ExperimentRunner,
            evaluate_promotion,
            render_report,
            save_spec,
            load_spec,
            translate_spec,
        )

        spec = EvolutionSpec.from_dict(sample_spec_dict())
        decision = evaluate_promotion(
            spec,
            [
                EvalResult("math_holdout", baseline=0.50, candidate=0.53),
                EvalResult("general_regression", baseline=0.60, candidate=0.598),
            ],
        )
        artifact = translate_spec(spec, "tinytron")
        report = render_report(spec, decision)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / "spec.json"
            save_spec(spec, spec_path)
            self.assertEqual(load_spec(spec_path).id, spec.id)

            registry = EvolutionRegistry(root / "registry")
            self.assertTrue(registry.record_spec(spec).exists())
            self.assertTrue(registry.record_artifact(spec.id, artifact).exists())
            self.assertTrue(registry.record_decision(spec.id, decision).exists())
            self.assertTrue(registry.record_report(spec.id, report).exists())

            prepared = ExperimentRunner(root).prepare(spec, "tinytron")
            self.assertTrue(prepared.is_runnable)
            self.assertEqual(prepared.commands, artifact.commands)

    def test_git_backend_prepares_isolated_experiment_worktree(self) -> None:
        from evolution import EvolutionRegistry, GitBackend

        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_git_repo(Path(tmp) / "repo")
            backend = GitBackend(repo)

            state = backend.prepare_experiment(
                "router-entropy-v1",
                worktree_root=Path(tmp) / "worktrees",
            )

            self.assertEqual(state.branch, "exp/router-entropy-v1")
            self.assertEqual(state.base_commit, state.candidate_commit)
            self.assertEqual(len(state.diff_hash), 64)
            self.assertTrue(Path(state.worktree_path).exists())

            registry = EvolutionRegistry(Path(tmp) / "registry")
            path = registry.record_git_state(state.spec_id, state)
            self.assertTrue(path.exists())
            self.assertIn("base_commit", path.read_text(encoding="utf-8"))

    def test_git_backend_rejects_dirty_repo_and_unsafe_branch(self) -> None:
        from evolution import EvolutionSafetyError, GitBackend

        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_git_repo(Path(tmp) / "repo")
            backend = GitBackend(repo)

            with self.assertRaisesRegex(EvolutionSafetyError, "namespace"):
                backend.prepare_experiment(
                    "bad-branch",
                    branch="main",
                    require_clean_repo=False,
                    worktree_root=Path(tmp) / "worktrees",
                )

            (repo / "README.md").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(EvolutionSafetyError, "dirty"):
                backend.prepare_experiment(
                    "dirty",
                    worktree_root=Path(tmp) / "worktrees",
                )

def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _run_git(path, "init")
    _run_git(path, "checkout", "-b", "main")
    (path / "README.md").write_text("tinytron\n", encoding="utf-8")
    _run_git(path, "add", "README.md")
    _run_git(
        path,
        "-c",
        "user.name=Tinytron",
        "-c",
        "user.email=tinytron@example.com",
        "commit",
        "-m",
        "init",
    )
    return path


def _run_git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
