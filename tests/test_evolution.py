from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
