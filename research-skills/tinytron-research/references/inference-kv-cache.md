# Inference And KV Cache Research

Use this reference for generation, cache layout, prefill/decode, and inference memory experiments.

## Useful Entry Points

- `tinytron/inference/engine.py`: generation loop, prefill/decode timing, sampling synchronization under SEP.
- `tinytron/inference/checkpoint.py`: checkpoint loading and bridge-backed model resharding into the current inference layout.
- `tinytron/inference/cache.py`: paged KV cache allocation, append behavior, layer cache layout.
- `tinytron/inference/sampler.py`: temperature, top-k, top-p sampling.
- `tinytron/model/modules/attn.py`: `past_kv` handling, paged cache detection, cache causal mask, SP decode path, QKV sharding.
- `scripts/debug/inference.py` and `scripts/debug/inference.sh`: inference smoke-test entry points.

## Good Experiment Shapes

- Paged versus non-paged KV cache with the same prompt and decode length.
- Page-size sweep for memory footprint, append overhead, and decode tokens/sec.
- Cache layout alternatives, such as head-major versus token-major organization.
- Prefix reuse or prompt cache experiments.
- Sliding-window cache, sink-token patterns, or long-context decode.
- KV quantization experiments with explicit dtype/scale handling and quality checks.
- Speculative or assisted decoding prototypes, keeping baseline generation unchanged.

## Metrics

- Prefill tokens/sec.
- Decode tokens/sec.
- Peak memory if available.
- Generated length and stop behavior.
- Cache allocation size and fragmentation proxy if measured.
- Numerical or token parity for experiments intended to be behavior-preserving.

## Guardrails

- Treat prefill and decode separately; they stress different paths.
- Keep random-weight smoke tests distinct from checkpoint quality tests.
- For cache layout changes, verify append, retrieval, and causal masking together.
- Avoid changing sampling behavior when the experiment is only about cache or memory.
- Treat `tinytron/bridge` as load-time model-state infrastructure. KV-cache and decode experiments should usually stay in inference/model modules, not bridge.
