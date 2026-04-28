from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class RLPrimitivesTest(unittest.TestCase):
    def test_gather_log_probs_matches_log_softmax(self) -> None:
        import torch

        from tinytron.rl import gather_log_probs

        logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 0.0, 1.0]]])
        token_ids = torch.tensor([[2, 0]])

        got = gather_log_probs(logits, token_ids)
        expected = torch.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

        self.assertTrue(torch.allclose(got, expected))

    def test_response_mask_keeps_prompt_out_and_includes_first_eos(self) -> None:
        import torch

        from tinytron.rl import build_response_mask

        sequences = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 9, 8, 7],
            ]
        )
        mask = build_response_mask(sequences, prompt_len=2, eos_token_id=4)

        expected = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_sequence_log_probs_sums_only_masked_tokens(self) -> None:
        import torch

        from tinytron.rl import sequence_log_probs

        log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        got = sequence_log_probs(log_probs, mask)

        self.assertTrue(torch.equal(got, torch.tensor([-4.0, -11.0])))

    def test_make_rollout_batch_keeps_response_aligned_fields(self) -> None:
        import torch

        from tinytron.rl import make_rollout_batch

        prompts = torch.tensor([[1, 2], [3, 4]])
        sequences = torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]])
        old_log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])

        batch = make_rollout_batch(prompts, sequences, old_log_probs=old_log_probs)

        self.assertTrue(torch.equal(batch.responses, torch.tensor([[5, 6], [7, 8]])))
        self.assertEqual(tuple(batch.response_mask.shape), tuple(old_log_probs.shape))

    def test_dpo_loss_prefers_larger_policy_margin(self) -> None:
        import torch

        from tinytron.rl import dpo_loss

        out = dpo_loss(
            chosen_log_probs=torch.tensor([-1.0, -2.0]),
            rejected_log_probs=torch.tensor([-3.0, -4.0]),
            reference_chosen_log_probs=torch.tensor([-2.0, -2.0]),
            reference_rejected_log_probs=torch.tensor([-3.0, -3.0]),
            beta=1.0,
        )

        self.assertLess(float(out.loss), 0.7)
        self.assertGreater(float(out.metrics["reward_accuracy"]), 0.0)

    def test_ppo_accepts_sequence_level_advantages(self) -> None:
        import torch

        from tinytron.rl import ppo_policy_loss

        log_probs = torch.tensor([[-1.0, -2.0], [-1.5, -2.5]])
        old_log_probs = log_probs.clone()
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(log_probs)

        out = ppo_policy_loss(log_probs, old_log_probs, advantages, mask)

        self.assertTrue(torch.isfinite(out.loss))


if __name__ == "__main__":
    unittest.main()
