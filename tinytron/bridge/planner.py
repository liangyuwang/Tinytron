from __future__ import annotations

from dataclasses import dataclass

from .layout import (
    LayoutIndex,
    Placement,
    ShardSpec,
    SliceSpec,
    intersect_slices,
    local_slices,
    slice_shape,
)


@dataclass(frozen=True)
class TensorMove:
    param_name: str
    src: Placement
    dst: Placement
    src_slices: tuple[SliceSpec, ...]
    dst_slices: tuple[SliceSpec, ...]
    global_slices: tuple[SliceSpec, ...]
    src_shard: ShardSpec
    dst_shard: ShardSpec
    op: str = "copy"

    @property
    def numel(self) -> int:
        numel = 1
        for dim in slice_shape(self.global_slices):
            numel *= dim
        return numel


class LayoutPlanner:
    """Build a tensor movement plan between two layout indices.

    The planner is intentionally transport-free. It does not know whether two
    placements are in the same process group, different distributed worlds, a
    filesystem, or a remote object store.
    """

    def plan(self, src: LayoutIndex, dst: LayoutIndex) -> tuple[TensorMove, ...]:
        src_by_param = src.by_param()
        moves: list[TensorMove] = []

        for dst_shard in sorted(dst.shards, key=self._shard_sort_key):
            src_shards = src_by_param.get(dst_shard.param_name)
            if not src_shards:
                raise KeyError(f"source layout has no shards for parameter {dst_shard.param_name!r}")

            chosen_by_global_slice: dict[tuple[SliceSpec, ...], ShardSpec] = {}
            for src_shard in sorted(src_shards, key=self._shard_sort_key):
                if src_shard.global_shape != dst_shard.global_shape:
                    raise ValueError(
                        f"shape mismatch for {dst_shard.param_name}: "
                        f"source {src_shard.global_shape}, target {dst_shard.global_shape}"
                    )
                overlap = intersect_slices(src_shard.global_slices, dst_shard.global_slices)
                if overlap is None:
                    continue
                current = chosen_by_global_slice.get(overlap)
                if current is None or self._prefer_source(src_shard, current, dst_shard):
                    chosen_by_global_slice[overlap] = src_shard

            self._validate_coverage(dst_shard, chosen_by_global_slice)
            for global_slice, src_shard in sorted(chosen_by_global_slice.items()):
                moves.append(
                    TensorMove(
                        param_name=dst_shard.param_name,
                        src=src_shard.placement,
                        dst=dst_shard.placement,
                        src_slices=local_slices(src_shard.global_slices, global_slice),
                        dst_slices=local_slices(dst_shard.global_slices, global_slice),
                        global_slices=global_slice,
                        src_shard=src_shard,
                        dst_shard=dst_shard,
                    )
                )

        return tuple(moves)

    def _validate_coverage(
        self,
        dst_shard: ShardSpec,
        chosen_by_global_slice: dict[tuple[SliceSpec, ...], ShardSpec],
    ) -> None:
        covered = 0
        for global_slice in chosen_by_global_slice:
            chunk_numel = 1
            for dim in slice_shape(global_slice):
                chunk_numel *= dim
            covered += chunk_numel

        expected = 1
        for dim in dst_shard.local_shape:
            expected *= dim
        if covered != expected:
            raise ValueError(
                f"source layout does not cover target shard {dst_shard.param_name!r} "
                f"at {dst_shard.placement}: covered {covered} elements, expected {expected}"
            )

    def _shard_sort_key(self, shard: ShardSpec) -> tuple:
        return (shard.param_name, shard.global_slices, shard.placement.sort_key())

    def _prefer_source(self, candidate: ShardSpec, current: ShardSpec, dst: ShardSpec) -> bool:
        return candidate.placement.rank == dst.placement.rank and current.placement.rank != dst.placement.rank
