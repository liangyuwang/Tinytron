from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol
import base64

import torch
import torch.distributed as dist

from .layout import Placement, ShardSpec, SliceSpec, slice_shape, to_torch_slices


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


class DistributedStateDictStore:
    """Live state-dict store backed by torch.distributed broadcasts."""

    def __init__(
        self,
        local_state_dict: dict[str, torch.Tensor] | None = None,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DistributedStateDictStore requires an initialized process group")
        self.local_state_dict = {} if local_state_dict is None else local_state_dict
        self.process_group = process_group
        self.rank = dist.get_rank()
        self.device = torch.device(device) if device is not None else None

    def read(
        self,
        shard: ShardSpec,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> torch.Tensor:
        src_rank = int(shard.placement.rank or 0)
        if self.rank == src_rank:
            tensor = self.local_state_dict[shard.param_name]
            if tuple(tensor.shape) != shard.local_shape:
                raise ValueError(
                    f"stored tensor shape mismatch for {shard.param_name!r} at rank {self.rank}: "
                    f"got {tuple(tensor.shape)}, expected {shard.local_shape}"
                )
            if local_slices is not None:
                tensor = tensor[to_torch_slices(local_slices)]
            tensor = tensor.contiguous()
        else:
            shape = slice_shape(local_slices) if local_slices is not None else shard.local_shape
            like = self.local_state_dict.get(shard.param_name)
            if like is None:
                raise KeyError(f"missing local tensor metadata for {shard.param_name!r}")
            device = self.device or like.device
            tensor = torch.empty(shape, dtype=like.dtype, device=device)

        dist.broadcast(tensor, src=src_rank, group=self.process_group)
        return tensor

    def read_local(
        self,
        shard: ShardSpec,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> torch.Tensor:
        tensor = self.local_state_dict[shard.param_name]
        if tuple(tensor.shape) != shard.local_shape:
            raise ValueError(
                f"stored tensor shape mismatch for {shard.param_name!r} at rank {self.rank}: "
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
        dst_rank = int(shard.placement.rank or 0)
        if self.rank != dst_rank:
            return
        if local_slices is None:
            if tuple(tensor.shape) != shard.local_shape:
                raise ValueError(
                    f"write shape mismatch for {shard.param_name!r}: "
                    f"got {tuple(tensor.shape)}, expected {shard.local_shape}"
                )
            self.local_state_dict[shard.param_name] = tensor.clone()
            return

        current = self.local_state_dict.get(shard.param_name)
        if current is None:
            current = torch.empty(shard.local_shape, dtype=tensor.dtype, device=tensor.device)
            self.local_state_dict[shard.param_name] = current
        if tuple(current.shape) != shard.local_shape:
            raise ValueError(
                f"stored tensor shape mismatch for {shard.param_name!r} at rank {self.rank}: "
                f"got {tuple(current.shape)}, expected {shard.local_shape}"
            )
        current[to_torch_slices(local_slices)] = tensor

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.local_state_dict


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


class StateDictShardFileStore:
    """Read tensor shards from per-placement state_dict files."""

    def __init__(
        self,
        path_for_placement: Callable[[Placement], str | Path],
        map_location: str | torch.device = "cpu",
        cache: bool = True,
    ):
        self.path_for_placement = path_for_placement
        self.map_location = map_location
        self.cache = cache
        self._state_dict_cache: dict[Placement, dict[str, torch.Tensor]] = {}

    def read(
        self,
        shard: ShardSpec,
        local_slices: tuple[SliceSpec, ...] | None = None,
    ) -> torch.Tensor:
        state_dict = self._state_dict_for(shard.placement)
        try:
            tensor = state_dict[shard.param_name]
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
        raise NotImplementedError("StateDictShardFileStore is read-only")

    def _state_dict_for(self, placement: Placement) -> dict[str, torch.Tensor]:
        if self.cache and placement in self._state_dict_cache:
            return self._state_dict_cache[placement]
        path = Path(self.path_for_placement(placement))
        state_dict = torch.load(path, map_location=self.map_location, weights_only=True)
        if self.cache:
            self._state_dict_cache[placement] = state_dict
        return state_dict
