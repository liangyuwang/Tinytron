"""aimegatron.train.checkpoint

Sharded checkpointing with layout-driven resharding.

On-disk layout (one directory per step):
    step_XXXXXXX/
        meta.pt                          # step + layout sizes (global rank 0)
        model_pp{stage}_tp{rank}.pt      # stage module state of each rank
        opt_global{rank}.pt              # optimizer state of each rank (opt.)

Loading rules:
- same pp/tp on disk as in memory   -> each rank loads its own file
- different tp                      -> the stage's source TP files are read
  and each parameter is resharded via aimegatron.train.layout
- different pp                      -> every key is routed to the source
  stage that owns it (blocks by index range, wte first, lnf/lm_head last),
  combined with TP resharding when tp also changed
- different ep                       -> unsupported in v1.1 (explicit error)
- optimizer state is restored only when world/tp/pp/ep/dp all match.
"""

from __future__ import annotations

import os

import torch

from aimegatron.core import mesh
from aimegatron.train.layout import reshard_tensor, source_stage_of_key

META_FILE = "meta.pt"


def checkpoint_path(log_dir: str, step: int) -> str:
    return os.path.join(log_dir, f"step_{step:07d}")


def model_file_name(pp_rank: int, tp_rank: int) -> str:
    return f"model_pp{pp_rank:03d}_tp{tp_rank:03d}.pt"


def find_latest_checkpoint(log_dir: str) -> str | None:
    if not os.path.isdir(log_dir):
        return None
    steps = []
    for name in os.listdir(log_dir):
        if name.startswith("step_") and os.path.isfile(os.path.join(log_dir, name, META_FILE)):
            try:
                steps.append(int(name[len("step_"):]))
            except ValueError:
                continue
    if not steps:
        return None
    return checkpoint_path(log_dir, max(steps))


def save_checkpoint(log_dir: str, step: int, model, optimizer=None, extra_meta: dict | None = None) -> str:
    path = checkpoint_path(log_dir, step)
    os.makedirs(path, exist_ok=True)
    if mesh.is_initialized():
        torch.distributed.barrier()

    torch.save(model.state_dict(),
               os.path.join(path, model_file_name(mesh.get_pp_rank(), mesh.get_tp_rank())))
    if optimizer is not None:
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        torch.save(optimizer.state_dict(), os.path.join(path, f"opt_global{global_rank:05d}.pt"))

    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if global_rank == 0:
        meta = {
            "step": step,
            "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            "tp_size": mesh.get_tp_world_size(),
            "pp_size": mesh.get_pp_world_size(),
            "dp_size": mesh.get_dp_world_size(),
            "ep_size": mesh.get_ep_world_size(),
        }
        if extra_meta:
            meta.update(extra_meta)
        torch.save(meta, os.path.join(path, META_FILE))

    if mesh.is_initialized():
        torch.distributed.barrier()
    return path


def _load_stage_states(path: str, stage: int, source_tp_size: int, map_location) -> list[dict]:
    sources = []
    for tp_rank in range(source_tp_size):
        file = os.path.join(path, model_file_name(stage, tp_rank))
        assert os.path.isfile(file), f"missing checkpoint shard: {file}"
        sources.append(torch.load(file, map_location=map_location, weights_only=True))
    return sources


def load_checkpoint(path: str, model, optimizer=None) -> dict:
    """Load a checkpoint directory into model (and optimizer when compatible).
    Returns the meta dict."""
    meta = torch.load(os.path.join(path, META_FILE), map_location="cpu", weights_only=True)
    source_pp_size, source_tp_size = meta["pp_size"], meta["tp_size"]
    target_pp_size, target_tp_size = mesh.get_pp_world_size(), mesh.get_tp_world_size()
    target_pp_rank, target_tp_rank = mesh.get_pp_rank(), mesh.get_tp_rank()
    device = next(model.parameters()).device

    if meta.get("ep_size", 1) != mesh.get_ep_world_size():
        raise NotImplementedError(
            f"checkpoint ep_size {meta.get('ep_size', 1)} != current ep_size "
            f"{mesh.get_ep_world_size()}; cross-EP resharding is not supported in v1.1")

    if source_pp_size == target_pp_size and source_tp_size == target_tp_size:
        state = torch.load(os.path.join(path, model_file_name(target_pp_rank, target_tp_rank)),
                           map_location=device, weights_only=True)
    else:
        num_layer = meta.get("num_layer")
        assert num_layer, "checkpoint meta lacks num_layer (saved by an old trainer)"
        stage_cache: dict[int, list[dict]] = {}
        target_keys = set(model.state_dict().keys())
        state = {}
        for name in target_keys:
            stage = source_stage_of_key(name, source_pp_size, num_layer) \
                if source_pp_size != target_pp_size else target_pp_rank
            if stage not in stage_cache:
                stage_cache[stage] = _load_stage_states(path, stage, source_tp_size,
                                                        map_location=device)
            shards = [src[name] for src in stage_cache[stage]]
            state[name] = reshard_tensor(shards, name, source_tp_size, target_tp_size,
                                         target_tp_rank)

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tied lm_head weights only exist once in memory; tolerate the key either way.
    assert all("lm_head" in k for k in missing + unexpected), \
        f"checkpoint mismatch: missing={missing} unexpected={unexpected}"

    if optimizer is not None:
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        layout_matches = (
            meta.get("world_size", 1) == world_size
            and source_tp_size == target_tp_size
            and source_pp_size == target_pp_size
            and meta.get("dp_size", 1) == mesh.get_dp_world_size()
            and meta.get("ep_size", 1) == mesh.get_ep_world_size()
        )
        if layout_matches:
            opt_file = os.path.join(path, f"opt_global{global_rank:05d}.pt")
            if os.path.isfile(opt_file):
                optimizer.load_state_dict(torch.load(opt_file, map_location=device, weights_only=True))
        else:
            if global_rank == 0:
                print(f"[aimegatron.checkpoint] layout changed "
                      f"(tp {source_tp_size}->{target_tp_size}, "
                      f"pp {source_pp_size}->{target_pp_size}); optimizer state not restored",
                      flush=True)
    return meta
