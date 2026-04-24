from __future__ import annotations

import os
import torch
import torch.distributed as dist

from tinytron.model.config import ModelConfig, build_model_config
from tinytron.training.config import ParallelConfig
from tinytron.inference import InferenceEngine
from tinytron.inference.arguments import build_parser, parse_prompt_token_ids
from tinytron.distributed import parallel_state


def is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def initialize_distributed_inference(args) -> tuple[str, bool]:
    if not is_distributed_launch():
        return args.device, True

    dist.init_process_group(backend=args.backend, init_method=args.init_method)
    local_rank = int(os.environ["LOCAL_RANK"])

    if args.device == "cuda":
        device = f"cuda:{local_rank}"
    elif args.device.startswith("cuda:"):
        expected_device = f"cuda:{local_rank}"
        if args.device != expected_device:
            raise ValueError(
                f"Distributed inference rank {local_rank} must use device {expected_device}; "
                f"got {args.device!r}. Use --device cuda under torchrun."
            )
        device = args.device
    else:
        raise ValueError("Distributed inference currently expects --device to target CUDA ranks managed by torchrun.")

    torch.cuda.set_device(local_rank)
    parallel_state.initialize_model_parallel(
        ParallelConfig(
            backend=args.backend,
            init_method=args.init_method,
            sep_size=args.sep_size,
        )
    )
    return device, dist.get_rank() == 0


def destroy_distributed_inference() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_model_config_from_checkpoint(args) -> ModelConfig:
    model_cfg_dict = None
    if args.checkpoint_path:
        meta_path = args.checkpoint_path.replace("_model.pt", "_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            model_cfg_dict = meta.get("config", {}).get("model")

    if model_cfg_dict is None:
        model_cfg = build_model_config(args, seed=args.seed)
    else:
        model_cfg = ModelConfig(**model_cfg_dict)

    return model_cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.init_from_scratch and not args.checkpoint_path:
        parser.error("Either provide --checkpoint_path, or set --init_from_scratch for random-weight smoke testing.")
    model_config = load_model_config_from_checkpoint(args)
    if is_distributed_launch() and args.sep_size <= 1:
        parser.error("Distributed inference requires --sep_size > 1 to initialize SEP process groups.")

    device = args.device
    master_process = True
    try:
        device, master_process = initialize_distributed_inference(args)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        prompt = torch.tensor([parse_prompt_token_ids(args.prompt_token_ids)], dtype=torch.long)
        engine = InferenceEngine(
            model_config=model_config,
            checkpoint_path=args.checkpoint_path,
            device=device,
            dtype=dtype,
        )
        output, stats = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=args.eos_token_id,
            return_stats=True,
        )
        if master_process:
            print(",".join(str(int(x)) for x in output[0].tolist()))
            print(f"prefill tok/s: {stats['prefill_tokens_per_sec']:.2f}")
            print(f"decode tok/s: {stats['decode_tokens_per_sec']:.2f}")
    finally:
        destroy_distributed_inference()


if __name__ == "__main__":
    main()
