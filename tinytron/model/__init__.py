from __future__ import annotations

from .config import ModelConfig, add_model_config_args, build_model_config

__all__ = [
    "ModelConfig",
    "add_model_config_args",
    "build_model_config",
    "GPT",
    "Block",
    "TextEmbedding",
    "ImgEmbedding",
    "Attention",
    "MLP",
    "MoE",
    "LayerNorm",
]


def __getattr__(name: str):
    if name in {"GPT", "Block"}:
        from .gpt import GPT, Block

        return {"GPT": GPT, "Block": Block}[name]
    if name in {"TextEmbedding", "ImgEmbedding"}:
        from .modules.emb import TextEmbedding, ImgEmbedding

        return {"TextEmbedding": TextEmbedding, "ImgEmbedding": ImgEmbedding}[name]
    if name == "Attention":
        from .modules.attn import Attention

        return Attention
    if name in {"MLP", "MoE"}:
        from .modules.mlp import MLP, MoE

        return {"MLP": MLP, "MoE": MoE}[name]
    if name == "LayerNorm":
        from .modules.norm import LayerNorm

        return LayerNorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
