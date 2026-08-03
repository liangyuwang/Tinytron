"""aimegatron: an AI-native Megatron-style distributed training framework.

Design principles:
- Parallelism is declarative: one ParallelConfig, one mesh builder.
- Sharding is annotation, not code: models are built from small TP primitives.
- Every module is small and has a single responsibility, so both humans and
  AI agents can read, mutate, and verify each piece independently.
"""

__version__ = "0.1.0"
