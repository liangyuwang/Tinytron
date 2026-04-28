from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .planner import TensorMove

if TYPE_CHECKING:
    from .stores import TensorStore


@dataclass
class BridgeContext:
    src_store: "TensorStore"
    dst_store: "TensorStore"


class TensorRoute(Protocol):
    def can_materialize(self, move: TensorMove, context: BridgeContext) -> bool:
        ...

    def materialize(self, move: TensorMove, context: BridgeContext) -> None:
        ...


class LocalCopyRoute:
    """Route for stores that are reachable from the current Python process."""

    def can_materialize(self, move: TensorMove, context: BridgeContext) -> bool:
        return True

    def materialize(self, move: TensorMove, context: BridgeContext) -> None:
        tensor = context.src_store.read(move.src_shard, move.src_slices)
        context.dst_store.write(move.dst_shard, tensor, move.dst_slices)


class RoutedMaterializer:
    """Execute a plan through route objects selected per move."""

    def __init__(self, routes: list[TensorRoute] | None = None):
        self.routes = routes or [LocalCopyRoute()]

    def materialize(self, plan: tuple[TensorMove, ...], context: BridgeContext) -> None:
        for move in plan:
            route = self._route_for(move, context)
            route.materialize(move, context)

    def _route_for(self, move: TensorMove, context: BridgeContext) -> TensorRoute:
        for route in self.routes:
            if route.can_materialize(move, context):
                return route
        raise RuntimeError(f"no bridge route can materialize move: {move}")
