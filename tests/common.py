"""Shared harness for multi-process distributed tests (gloo on CPU).

Usage:
    run_distributed(world_size, worker, tp_size=..., pp_size=..., ep_size=...,
                    sequence_parallel=...)

where worker(rank) runs inside a spawned process with torch.distributed and
the aimegatron mesh already initialized.
"""

import functools
import os
import tempfile
import uuid

import torch.distributed as dist
import torch.multiprocessing as mp

from aimegatron.core import mesh
from aimegatron.core.config import ParallelConfig


def _spawn_entry(rank, world_size, init_file, tp_size, pp_size, ep_size,
                 sequence_parallel, worker):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        mesh.initialize_parallel(ParallelConfig(
            tp_size=tp_size, pp_size=pp_size, ep_size=ep_size,
            sequence_parallel=sequence_parallel))
        worker(rank)
    finally:
        mesh.destroy_parallel()
        dist.destroy_process_group()


def run_distributed(world_size, worker, tp_size=1, pp_size=1, ep_size=1,
                    sequence_parallel=False):
    init_file = os.path.join(tempfile.gettempdir(), f"aimegatron_test_{uuid.uuid4().hex}")
    wrapped = functools.partial(worker)
    try:
        mp.spawn(
            _spawn_entry,
            args=(world_size, init_file, tp_size, pp_size, ep_size,
                  sequence_parallel, wrapped),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)
