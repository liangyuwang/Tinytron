"""Config validation tests (no distributed runtime needed)."""

import unittest

from aimegatron.core.config import Config, ModelConfig, ParallelConfig, TrainConfig


def _config(**overrides) -> Config:
    config = Config(
        model=ModelConfig(vocab_size=64, hidden_size=16, num_attention_heads=2,
                          num_key_value_heads=2, intermediate_size=32, num_layer=2),
        parallel=ParallelConfig(tp_size=2),
        train=TrainConfig(total_batch_size=16, batch_size=2, seq_len=8),
    )
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(config, section), field, value)
    return config


class TestConfigValidation(unittest.TestCase):

    def test_valid_config_passes(self):
        _config().validate(world_size=2)

    def test_world_size_must_be_divisible_by_tp(self):
        with self.assertRaisesRegex(ValueError, "divisible by tp_size"):
            _config().validate(world_size=3)

    def test_sequence_parallel_requires_tp(self):
        with self.assertRaisesRegex(ValueError, "sequence_parallel requires tp_size"):
            _config(**{"parallel.tp_size": 1, "parallel.sequence_parallel": True}).validate(world_size=1)

    def test_heads_must_be_divisible_by_tp(self):
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            _config(**{"parallel.tp_size": 4}).validate(world_size=4)

    def test_kv_heads_must_be_divisible_by_tp(self):
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            _config(**{"model.num_key_value_heads": 1}).validate(world_size=2)

    def test_vocab_must_be_divisible_by_tp(self):
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            _config(**{"model.vocab_size": 63}).validate(world_size=2)

    def test_batch_must_divide_total_batch(self):
        with self.assertRaisesRegex(ValueError, "total_batch_size"):
            _config(**{"train.total_batch_size": 17}).validate(world_size=2)

    def test_seq_len_within_block_size(self):
        with self.assertRaisesRegex(ValueError, "block_size"):
            _config(**{"model.block_size": 4}).validate(world_size=2)

    def test_bad_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            _config(**{"train.dtype": "fp16"}).validate(world_size=2)


if __name__ == "__main__":
    unittest.main()
