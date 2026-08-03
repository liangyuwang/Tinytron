"""aimegatron.core.registry

Tiny name -> factory registry. This is the extension seam of the framework:
new optimizers, norms, or schedules are added by registering a factory here
instead of editing call sites. Kept deliberately small so it can be mutated
and audited in one read.
"""


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, callable] = {}

    def register(self, name: str):
        def decorator(factory):
            if name in self._items:
                raise ValueError(f"[{self.name}] '{name}' is already registered")
            self._items[name] = factory
            return factory
        return decorator

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"[{self.name}] unknown '{name}'; available: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


OPTIMIZERS = Registry("optimizers")
NORMS = Registry("norms")


def _register_builtins() -> None:
    import torch.optim
    from aimegatron.model.norm import create_layernorm, create_rmsnorm

    @OPTIMIZERS.register("adam")
    def create_adamw(params, lr: float, weight_decay: float):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    @NORMS.register("layernorm")
    def _layernorm(hidden_size: int):
        return create_layernorm(hidden_size)

    @NORMS.register("rmsnorm")
    def _rmsnorm(hidden_size: int):
        return create_rmsnorm(hidden_size)


_register_builtins()
