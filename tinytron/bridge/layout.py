from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable


SliceSpec = tuple[int, int]


@dataclass(frozen=True)
class RankCoord:
    """Logical rank coordinates without assuming a specific process group."""

    axes: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_axes(cls, **axes: int) -> "RankCoord":
        return cls(tuple(sorted((str(axis), int(value)) for axis, value in axes.items())))

    def get(self, axis: str, default: int | None = None) -> int | None:
        values = dict(self.axes)
        return values.get(axis, default)


@dataclass(frozen=True)
class Placement:
    """Physical or logical location for a shard.

    `system` can name any world: training, inference, canonical, file, remote, etc.
    `rank`, `coord`, `group_id`, and `address` are descriptive metadata; routes decide
    which fields matter for a concrete materialization path.
    """

    system: str
    rank: int | None = None
    coord: RankCoord | None = None
    group_id: str | None = None
    address: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def sort_key(self) -> tuple:
        coord_key = self.coord.axes if self.coord is not None else ()
        return (
            self.system,
            -1 if self.rank is None else self.rank,
            "" if self.group_id is None else self.group_id,
            "" if self.address is None else self.address,
            coord_key,
            self.metadata,
        )


def full_slice(shape: tuple[int, ...]) -> tuple[SliceSpec, ...]:
    return tuple((0, int(dim)) for dim in shape)


def slice_shape(slices: tuple[SliceSpec, ...]) -> tuple[int, ...]:
    return tuple(stop - start for start, stop in slices)


def intersect_slices(
    left: tuple[SliceSpec, ...],
    right: tuple[SliceSpec, ...],
) -> tuple[SliceSpec, ...] | None:
    if len(left) != len(right):
        raise ValueError(f"slice rank mismatch: {len(left)} vs {len(right)}")

    out: list[SliceSpec] = []
    for (a0, a1), (b0, b1) in zip(left, right):
        start = max(a0, b0)
        stop = min(a1, b1)
        if start >= stop:
            return None
        out.append((start, stop))
    return tuple(out)


def to_torch_slices(slices: tuple[SliceSpec, ...]) -> tuple[slice, ...]:
    return tuple(slice(start, stop) for start, stop in slices)


def local_slices(
    shard_slices: tuple[SliceSpec, ...],
    global_slices: tuple[SliceSpec, ...],
) -> tuple[SliceSpec, ...]:
    if len(shard_slices) != len(global_slices):
        raise ValueError(f"slice rank mismatch: {len(shard_slices)} vs {len(global_slices)}")
    out: list[SliceSpec] = []
    for (shard_start, shard_stop), (global_start, global_stop) in zip(shard_slices, global_slices):
        if global_start < shard_start or global_stop > shard_stop:
            raise ValueError(
                f"global slice {(global_start, global_stop)} is outside shard slice "
                f"{(shard_start, shard_stop)}"
            )
        out.append((global_start - shard_start, global_stop - shard_start))
    return tuple(out)


@dataclass(frozen=True)
class ShardSpec:
    param_name: str
    global_shape: tuple[int, ...]
    global_slices: tuple[SliceSpec, ...]
    placement: Placement
    axis_tags: tuple[str, ...] = ()
    replica_group: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def local_shape(self) -> tuple[int, ...]:
        return slice_shape(self.global_slices)

    @classmethod
    def replicated(
        cls,
        param_name: str,
        global_shape: tuple[int, ...],
        placement: Placement,
        replica_group: str | None = None,
    ) -> "ShardSpec":
        return cls(
            param_name=param_name,
            global_shape=tuple(int(dim) for dim in global_shape),
            global_slices=full_slice(tuple(int(dim) for dim in global_shape)),
            placement=placement,
            replica_group=replica_group,
        )


@dataclass(frozen=True)
class LayoutIndex:
    name: str
    shards: tuple[ShardSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shards", tuple(self.shards))

    @classmethod
    def from_shards(cls, name: str, shards: Iterable[ShardSpec]) -> "LayoutIndex":
        return cls(name=name, shards=tuple(shards))

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(sorted({shard.param_name for shard in self.shards}))

    def shards_for(self, param_name: str) -> tuple[ShardSpec, ...]:
        return tuple(shard for shard in self.shards if shard.param_name == param_name)

    def local_shards(self, placement: Placement) -> tuple[ShardSpec, ...]:
        return tuple(shard for shard in self.shards if shard.placement == placement)

    def placements(self) -> tuple[Placement, ...]:
        return tuple(sorted({shard.placement for shard in self.shards}, key=lambda p: p.sort_key()))

    def by_param(self) -> dict[str, tuple[ShardSpec, ...]]:
        grouped: dict[str, list[ShardSpec]] = defaultdict(list)
        for shard in self.shards:
            grouped[shard.param_name].append(shard)
        return {
            name: tuple(sorted(items, key=lambda shard: shard.placement.sort_key()))
            for name, items in grouped.items()
        }
