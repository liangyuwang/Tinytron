from __future__ import annotations

import os
import glob
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.checkpoint import state_dict_saver, state_dict_loader
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader

from tinytron.training import Trainer, build_config, build_parser
from tinytron.training.config import Config

# git clone https://github.com/liangyuwang/Streaming-Dataloader.git external/streaming_dataloader
from external.streaming_dataloader.dataset import DistributedDataset


@dataclass
class StreamingDatasetConfig:
    data_dir: str = "../data/fineweb-edu-sample-10BT/"
    shuffle: bool = True
    strict: bool = True
    global_skip_batches: int = 0


dataset_cfg: StreamingDatasetConfig | None = None


def parse_args():
    parser = build_parser()

    # override / extend Tinytron args for streaming dataset
    parser.add_argument(
        "--streaming_data_dir",
        type=str,
        default=None,
        help="Path to processed Streaming-Dataloader data directory "
             "(contains chunk_*.bin and meta.json). "
             "If not set, falls back to --dataset_path.",
    )
    parser.add_argument(
        "--streaming_shuffle",
        action="store_true",
        help="Enable deterministic sample shuffle inside Streaming-Dataloader.",
    )
    parser.add_argument(
        "--streaming_strict",
        action="store_true",
        help="Raise error if total samples < dp_world_size * num_workers.",
    )
    parser.add_argument(
        "--streaming_global_skip_batches",
        type=int,
        default=0,
        help="Number of globally-consumed samples to skip at dataset start.",
    )
    return parser.parse_args()


class OurTrainer(Trainer):
    def _init_dataset(self, config: Config):
        if config.data.use_mock_data:
            return super()._init_dataset(config)

        data_dir = dataset_cfg.data_dir

        if self.master_process:
            print(f"[dataset] loading Streaming-Dataloader from: {data_dir}")

        self.train_dataset = DistributedDataset(
            data_dir=data_dir,
            seq_len=config.train.seq_len,
            shuffle=dataset_cfg.shuffle,
            seed=config.seed.seed,
            strict=dataset_cfg.strict,
            global_skip_batches=dataset_cfg.global_skip_batches,
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_world_size,
        )

        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None

        # For IterableDataset, do NOT use DistributedSampler.
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=False,
        )
        self.val_loader = None
        self.num_train_samples = self.train_dataset.total_samples
        self.train_loader_iter_idx = 0

        if self.master_process:
            print(f"[dataset] total_tokens={self.train_dataset.total_tokens}")
            print(f"[dataset] total_samples={len(self.train_dataset)}")
            print(
                f"[dataset] batch_size={config.train.batch_size}, "
                f"seq_len={config.train.seq_len}, "
                f"num_workers={config.data.num_workers}, "
                f"dp_world_size={self.dp_world_size}"
            )

    def _next_train_batch(self):
        try:
            iter_idx, batch = next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_dataset, "set_epoch"):
                current_epoch = getattr(self.train_dataset, "epoch", 0)
                self.train_dataset.set_epoch(current_epoch + 1)
                if self.master_process:
                    print(f"[dataset] restart iterator at dataset epoch={current_epoch + 1}")
            self.train_loader_iter_idx = 0
            self.train_loader_iter = enumerate(self.train_loader)
            iter_idx, batch = next(self.train_loader_iter)
        self.train_loader_iter_idx = iter_idx + 1
        return batch

    def _resume_from_checkpoint(self):
        ckpt_dir = self.config.ckpt.resume_path or self.log_dir
        pattern = os.path.join(ckpt_dir, "*_model.pt")
        ckpts = sorted(glob.glob(pattern))

        if not ckpts:
            self.start_step = 0
            return

        ckpt_prefix = ckpts[-1].replace("_model.pt", "")
        meta_path = f"{ckpt_prefix}_meta.pt"
        meta = torch.load(meta_path, map_location="cpu")

        # 1) model
        state_dict = torch.load(
            f"{ckpt_prefix}_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.raw_model.load_state_dict(state_dict)

        # 2) optimizer
        opt_key = f"optimizer/rank{self.rank}"
        opt_state_placeholder = {opt_key: self.raw_optimizer.state_dict()}
        state_dict_loader.load(
            state_dict=opt_state_placeholder,
            storage_reader=FileSystemReader(f"{ckpt_prefix}_opt"),
        )
        self.raw_optimizer.load_state_dict(opt_state_placeholder[opt_key])

        # 3) dataset state (streaming version)
        dataset_state = meta.get("dataset_state", {})

        epoch = int(dataset_state.get("epoch", 0))
        global_skip_batches = int(dataset_state.get("global_skip_batches", 0))

        if hasattr(self.train_dataset, "global_skip_batches"):
            self.train_dataset.global_skip_batches = global_skip_batches
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

        self.train_loader_iter_idx = 0
        self.train_loader_iter = enumerate(self.train_loader)

        # 4) next step
        step = meta.get("step", None)
        self.start_step = (step + 1) if (step is not None) else 0

        if self.master_process:
            print(
                f"=> Resumed from {ckpt_dir} | next_step={self.start_step}, "
                f"dataset_epoch={epoch}, global_skip_batches={global_skip_batches}"
            )

        # 5) RNG
        rng_path = f"{ckpt_prefix}_rng/rank{self.rank}.pt"
        if os.path.exists(rng_path):
            rng = torch.load(rng_path, map_location="cpu")
            torch.set_rng_state(rng["torch"].to(torch.uint8).cpu())
            torch.cuda.set_rng_state(rng["cuda"].to(torch.uint8).cpu(), self.local_rank)
            np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])

        dist.barrier()
        torch.cuda.synchronize()

    def save(self, step: int | None = None):
        checkpoint_path = os.path.join(self.log_dir, f"{step:05d}")
        rng_dir = f"{checkpoint_path}_rng"
        os.makedirs(rng_dir, exist_ok=True)

        next_step = (step if step is not None else 0) + 1

        # Streaming-Dataloader state:
        # global_skip_batches is "number of globally-consumed samples to skip".
        # One optimizer step consumes:
        #   grad_accum_steps * batch_size * dp_world_size
        # global samples.
        global_samples_per_step = (
            self.training_info["grad_accum_steps"]
            * self.config.train.batch_size
            * self.dp_world_size
        )
        consumed_global_samples_next = next_step * global_samples_per_step

        total_samples = int(len(self.train_dataset))
        dataset_epoch_next = consumed_global_samples_next // total_samples
        global_skip_batches_next = consumed_global_samples_next % total_samples

        state_dict_saver.save(
            state_dict={f"optimizer/rank{self.rank}": self.raw_optimizer.state_dict()},
            storage_writer=FileSystemWriter(f"{checkpoint_path}_opt"),
        )

        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.local_rank),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        torch.save(rng_state, os.path.join(rng_dir, f"rank{self.rank}.pt"))
        dist.barrier()

        if self.master_process:
            torch.save(self.raw_model.state_dict(), f"{checkpoint_path}_model.pt")

            checkpoint = {
                "config": self.config.as_dict(),
                "step": step,
                "this_step_results": self.one_step_results,
                "opt_part_assignment": (
                    self.optimizer.part_assignment
                    if hasattr(self.optimizer, "part_assignment")
                    else None
                ),
                "dataset_state": {
                    "epoch": int(dataset_epoch_next),
                    "global_skip_batches": int(global_skip_batches_next),
                },
                "rng_state": rng_state,
            }
            torch.save(checkpoint, f"{checkpoint_path}_meta.pt")
        dist.barrier()


def main():
    args = parse_args()
    cfg = build_config(args)

    assert not cfg.train.do_val, "This example currently only wires train split."

    global dataset_cfg
    dataset_cfg = StreamingDatasetConfig(
        data_dir=args.streaming_data_dir or cfg.data.dataset_path,
        shuffle=bool(args.streaming_shuffle),
        strict=bool(args.streaming_strict),
        global_skip_batches=int(args.streaming_global_skip_batches),
    )

    trainer = OurTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()