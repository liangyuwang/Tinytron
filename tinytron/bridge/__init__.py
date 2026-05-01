from __future__ import annotations

from .layout import (
    LayoutIndex,
    Placement,
    RankCoord,
    ShardSpec,
    SliceSpec,
)
from .planner import LayoutPlanner, TensorMove
from .rules import (
    ParallelSpec,
    build_tinytron_canonical_layout,
    build_tinytron_inference_layout,
    build_tinytron_training_layout,
)

__all__ = [
    "BridgeContext",
    "DistributedCopyRoute",
    "DistributedStateDictStore",
    "LayoutIndex",
    "LayoutPlanner",
    "LocalCopyRoute",
    "local_training_state_dict_to_local_inference",
    "localize_layout",
    "Placement",
    "RankCoord",
    "RoutedMaterializer",
    "ShardSpec",
    "SliceSpec",
    "FileTensorStore",
    "StateDictShardFileStore",
    "StateDictTensorStore",
    "state_dict_to_local_inference",
    "TensorMove",
    "TensorRoute",
    "TensorStore",
    "ParallelSpec",
    "build_tinytron_canonical_layout",
    "build_tinytron_inference_layout",
    "build_tinytron_training_layout",
    "bridge_metadata",
    "canonical_state_dict_to_local_inference",
    "canonical_state_dict_to_local_training",
    "current_tinytron_parallel_spec",
    "current_tinytron_placement",
    "distributed_training_state_dict_to_local_inference",
]


def __getattr__(name: str):
    if name in {"BridgeContext", "DistributedCopyRoute", "LocalCopyRoute", "RoutedMaterializer", "TensorRoute"}:
        from .materializers import BridgeContext, DistributedCopyRoute, LocalCopyRoute, RoutedMaterializer, TensorRoute

        return {
            "BridgeContext": BridgeContext,
            "DistributedCopyRoute": DistributedCopyRoute,
            "LocalCopyRoute": LocalCopyRoute,
            "RoutedMaterializer": RoutedMaterializer,
            "TensorRoute": TensorRoute,
        }[name]
    if name in {"DistributedStateDictStore", "FileTensorStore", "StateDictShardFileStore", "StateDictTensorStore", "TensorStore"}:
        from .stores import DistributedStateDictStore, FileTensorStore, StateDictShardFileStore, StateDictTensorStore, TensorStore

        return {
            "DistributedStateDictStore": DistributedStateDictStore,
            "FileTensorStore": FileTensorStore,
            "StateDictShardFileStore": StateDictShardFileStore,
            "StateDictTensorStore": StateDictTensorStore,
            "TensorStore": TensorStore,
        }[name]
    if name in {
        "bridge_metadata",
        "canonical_state_dict_to_local_inference",
        "canonical_state_dict_to_local_training",
        "current_tinytron_parallel_spec",
        "current_tinytron_placement",
        "distributed_training_state_dict_to_local_inference",
        "local_training_state_dict_to_local_inference",
        "localize_layout",
        "state_dict_to_local_inference",
    }:
        from .model import (
            bridge_metadata,
            canonical_state_dict_to_local_inference,
            canonical_state_dict_to_local_training,
            current_tinytron_parallel_spec,
            current_tinytron_placement,
            distributed_training_state_dict_to_local_inference,
            local_training_state_dict_to_local_inference,
            localize_layout,
            state_dict_to_local_inference,
        )

        return {
            "bridge_metadata": bridge_metadata,
            "canonical_state_dict_to_local_inference": canonical_state_dict_to_local_inference,
            "canonical_state_dict_to_local_training": canonical_state_dict_to_local_training,
            "current_tinytron_parallel_spec": current_tinytron_parallel_spec,
            "current_tinytron_placement": current_tinytron_placement,
            "distributed_training_state_dict_to_local_inference": distributed_training_state_dict_to_local_inference,
            "local_training_state_dict_to_local_inference": local_training_state_dict_to_local_inference,
            "localize_layout": localize_layout,
            "state_dict_to_local_inference": state_dict_to_local_inference,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
