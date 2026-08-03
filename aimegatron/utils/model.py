"""aimegatron.utils.model"""

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count unique parameters (tied weights counted once)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        total += p.numel()
    return total
