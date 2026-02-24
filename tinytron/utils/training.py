import math
import argparse
import numpy as np
import random
import os
import torch
import warnings
from typing import Any
from packaging import version

def get_training_info(
    num_samples: int,
    tokens_per_sample: int,
    global_token_batch_size: int, 
    samples_per_dp_rank_per_micro_step: int, 
    dp_world_size: int,
    max_steps=None,
    max_epochs=None,
) -> dict[str, Any]:
    """
    Calculate training hyperparameters based on the dataset and hardware configuration.

    Priority rule:
    - If max_steps is provided, max_steps is authoritative.
    - If max_steps is not provided, derive max_steps from max_epochs.

    Args:
        num_samples (int): Total number of samples in the dataset.
        tokens_per_sample (int): Number of tokens in each sample.
        global_token_batch_size (int): Total number of tokens processed globally in each batch.
        samples_per_dp_rank_per_micro_step (int): Number of samples processed per dp rank in each training micro step.
        dp_world_size (int): DP world size used for training.
        max_steps (int, optional): Maximum number of training steps.
        max_epochs (int, optional): Maximum number of epochs to train.

    Returns:
        dict: A dictionary containing the computed training parameters.

    Raises:
        ValueError: If neither max_steps nor max_epochs is provided, or if dataset size is invalid.
    """
    if max_steps is None and max_epochs is None:
        raise ValueError("At least one of max_steps or max_epochs must be provided.")

    tokens_per_dp_rank_per_step = tokens_per_sample * samples_per_dp_rank_per_micro_step
    total_tokens_per_micro_step = tokens_per_dp_rank_per_step * dp_world_size
    grad_accum_steps = int(global_token_batch_size / total_tokens_per_micro_step)
    total_tokens_in_dataset = num_samples * tokens_per_sample

    if total_tokens_in_dataset <= 0:
        raise ValueError(
            f"Invalid dataset token count: num_samples={num_samples}, tokens_per_sample={tokens_per_sample}."
        )

    if max_steps is not None:
        if max_epochs is not None:
            warnings.warn(
                f"Both max_steps={max_steps} and max_epochs={max_epochs} were provided; "
                "max_steps takes precedence.",
                stacklevel=2,
            )
        max_epochs = (max_steps * global_token_batch_size) / total_tokens_in_dataset
    else:
        max_steps = int((max_epochs * total_tokens_in_dataset) / global_token_batch_size)

    return {
        "epochs": max_epochs,
        "max_steps": max_steps,
        "grad_accum_steps": grad_accum_steps,
        "total_tokens_per_micro_step": total_tokens_per_micro_step
    }

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if deterministic:
        torch.use_deterministic_algorithms(True)
