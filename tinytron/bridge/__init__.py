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
    TinytronParallelSpec,
    build_tinytron_canonical_layout,
    build_tinytron_inference_layout,
    build_tinytron_training_layout,
)

__all__ = [
    "BridgeContext",
    "LayoutIndex",
    "LayoutPlanner",
    "LocalCopyRoute",
    "Placement",
    "RankCoord",
    "RoutedMaterializer",
    "ShardSpec",
    "SliceSpec",
    "FileTensorStore",
    "StateDictTensorStore",
    "TensorMove",
    "TensorRoute",
    "TensorStore",
    "TinytronParallelSpec",
    "build_tinytron_canonical_layout",
    "build_tinytron_inference_layout",
    "build_tinytron_training_layout",
    "bridge_metadata",
    "canonical_state_dict_to_local_inference",
    "canonical_state_dict_to_local_training",
    "checkpoint_meta_path",
    "checkpoint_model_paths",
    "current_tinytron_parallel_spec",
    "export_tinytron_training_state_dict_to_canonical",
]


def __getattr__(name: str):
    if name in {"BridgeContext", "LocalCopyRoute", "RoutedMaterializer", "TensorRoute"}:
        from .materializers import BridgeContext, LocalCopyRoute, RoutedMaterializer, TensorRoute

        return {
            "BridgeContext": BridgeContext,
            "LocalCopyRoute": LocalCopyRoute,
            "RoutedMaterializer": RoutedMaterializer,
            "TensorRoute": TensorRoute,
        }[name]
    if name in {"FileTensorStore", "StateDictTensorStore", "TensorStore"}:
        from .stores import FileTensorStore, StateDictTensorStore, TensorStore

        return {
            "FileTensorStore": FileTensorStore,
            "StateDictTensorStore": StateDictTensorStore,
            "TensorStore": TensorStore,
        }[name]
    if name in {
        "bridge_metadata",
        "canonical_state_dict_to_local_inference",
        "canonical_state_dict_to_local_training",
        "checkpoint_meta_path",
        "checkpoint_model_paths",
        "current_tinytron_parallel_spec",
        "export_tinytron_training_state_dict_to_canonical",
    }:
        from .model import (
            bridge_metadata,
            canonical_state_dict_to_local_inference,
            canonical_state_dict_to_local_training,
            checkpoint_meta_path,
            checkpoint_model_paths,
            current_tinytron_parallel_spec,
            export_tinytron_training_state_dict_to_canonical,
        )

        return {
            "bridge_metadata": bridge_metadata,
            "canonical_state_dict_to_local_inference": canonical_state_dict_to_local_inference,
            "canonical_state_dict_to_local_training": canonical_state_dict_to_local_training,
            "checkpoint_meta_path": checkpoint_meta_path,
            "checkpoint_model_paths": checkpoint_model_paths,
            "current_tinytron_parallel_spec": current_tinytron_parallel_spec,
            "export_tinytron_training_state_dict_to_canonical": export_tinytron_training_state_dict_to_canonical,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
