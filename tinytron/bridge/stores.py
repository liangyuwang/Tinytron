from __future__ import annotations

from pathlib import Path
from typing import Protocol
import base64

import torch

from .layout import Placement, ShardSpec, SliceSpec, to_torch_slices


class TensorStore(Protocol):
    def read(
        self,
        shard: ShardSpec,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> torch.Tensor:
        ...

    def write(
        self,
        shard: ShardSpec,
        tensor: torch.Tensor,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> None:
        ...


class StateDictTensorStore:
    """A placement-keyed state-dict store for local bridge tests and tooling."""

    def __init__(self, state_dicts: dict[Placement, dict[str, torch.Tensor]] | None = None):
        self.state_dicts: dict[Placement, dict[str, torch.Tensor]] = state_dicts or {}

    def read(
        self,
        shard: ShardSpec,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> torch.Tensor:
        try:
            tensor = self.state_dicts[shard.placement][shard.param_name]
        except KeyError as exc:
            raise KeyError(f"missing tensor {shard.param_name!r} at placement {shard.placement}") from exc
        if tuple(tensor.shape) != shard.local_shape:
            raise ValueError(
                f"stored tensor shape mismatch for {shard.param_name!r} at {shard.placement}: "
                f"got {tuple(tensor.shape)}, expected {shard.local_shape}"
            )
        if local_slices is None:
            return tensor
        return tensor[to_torch_slices(local_slices)]

    def write(
        self,
        shard: ShardSpec,
        tensor: torch.Tensor,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> None:
        placement_state = self.state_dicts.setdefault(shard.placement, {})
        if local_slices is None:
            if tuple(tensor.shape) != shard.local_shape:
                raise ValueError(
                    f"write shape mismatch for {shard.param_name!r}: "
                    f"got {tuple(tensor.shape)}, expected {shard.local_shape}"
                )
            placement_state[shard.param_name] = tensor.clone()
            return

        current = placement_state.get(shard.param_name)
        if current is None:
            current = torch.empty(shard.local_shape, dtype=tensor.dtype, device=tensor.device)
            placement_state[shard.param_name] = current
        if tuple(current.shape) != shard.local_shape:
            raise ValueError(
                f"stored tensor shape mismatch for {shard.param_name!r} at {shard.placement}: "
                f"got {tuple(current.shape)}, expected {shard.local_shape}"
            )
        current[to_torch_slices(local_slices)] = tensor

    def state_dict_for(self, placement: Placement) -> dict[str, torch.Tensor]:
        return self.state_dicts.setdefault(placement, {})


class FileTensorStore:
    """Simple per-shard tensor store.

    This is a materialization route building block, not the bridge's core
    abstraction. File naming is deterministic and placement-aware so unrelated
    training/inference worlds can exchange shards through a shared directory.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(
        self,
        shard: ShardSpec,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> torch.Tensor:
        path = self._path_for(shard)
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != shard.local_shape:
            raise ValueError(
                f"stored tensor shape mismatch for {shard.param_name!r} at {shard.placement}: "
                f"got {tuple(tensor.shape)}, expected {shard.local_shape}"
            )
        if local_slices is None:
            return tensor
        return tensor[to_torch_slices(local_slices)]

    def write(
        self,
        shard: ShardSpec,
        tensor: torch.Tensor,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> None:
        path = self._path_for(shard)
        if local_slices is None:
            if tuple(tensor.shape) != shard.local_shape:
                raise ValueError(
                    f"write shape mismatch for {shard.param_name!r}: "
                    f"got {tuple(tensor.shape)}, expected {shard.local_shape}"
                )
            torch.save(tensor.detach().cpu(), path)
            return

        if path.exists():
            current = torch.load(path, map_location="cpu", weights_only=True)
        else:
            current = torch.empty(shard.local_shape, dtype=tensor.dtype, device="cpu")
        current[to_torch_slices(local_slices)] = tensor.detach().cpu()
        torch.save(current, path)

    def _path_for(self, shard: ShardSpec) -> Path:
        placement_key = repr(shard.placement.sort_key()).encode("utf-8")
        placement_token = base64.urlsafe_b64encode(placement_key).decode("ascii").rstrip("=")
        param_token = shard.param_name.replace("/", "_").replace(".", "__")
        return self.root / f"{placement_token}--{param_token}.pt"
