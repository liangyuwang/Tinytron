from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import AutoTokenizer

from tinytron.training import Trainer, parse_args, build_config
from tinytron.training.config import Config


@dataclass
class DatasetConfig:
    hf_dataset_repo: str = "HuggingFaceFW/fineweb-edu"
    hf_dataset_name: Optional[str] = None
    hf_split: str = "train"
    tokenizer_name_or_path: str = "gpt2"
    max_samples: int = 200000
    min_chars: int = 50
    add_eos_token: bool = True
    seed_shuffle_buffer: int = 0


class PackedTokenDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int,
        add_eos_token: bool = True,
    ):
        self.seq_len = seq_len
        self.samples = []

        eos_id = tokenizer.eos_token_id
        if eos_id is None and add_eos_token:
            raise ValueError("tokenizer.eos_token_id is None")

        token_buffer: List[int] = []

        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            token_buffer.extend(ids)
            if add_eos_token:
                token_buffer.append(eos_id)

        chunk_len = seq_len + 1
        total = len(token_buffer) // chunk_len
        for i in range(total):
            chunk = token_buffer[i * chunk_len : (i + 1) * chunk_len]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.samples.append({"input_ids": x, "labels": y})

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No packed samples produced. token_buffer_len={len(token_buffer)}, "
                f"seq_len={seq_len}, chunk_len={seq_len+1}. "
                "Try increasing max_samples or reducing seq_len."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class OurTrainer(Trainer):
    def _init_dataset(self, config: Config):
        if config.data.use_mock_data:
            return super()._init_dataset(config)

        if self.master_process:
            print(f"[dataset] loading streaming dataset: {dataset_cfg.hf_dataset_repo} split={dataset_cfg.hf_split}")

        ds = load_dataset(
            path=dataset_cfg.hf_dataset_repo,
            name=dataset_cfg.hf_dataset_name,
            split=dataset_cfg.hf_split,
            streaming=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(dataset_cfg.tokenizer_name_or_path, use_fast=True)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        assert config.model.vocab_size >= len(self.tokenizer), (
            f"model.vocab_size={config.model.vocab_size} < tokenizer_size={len(self.tokenizer)}"
        )
        texts: List[str] = []
        for ex in islice(ds, dataset_cfg.max_samples):
            text = ex.get("text", None)
            if not text or len(text) < dataset_cfg.min_chars:
                continue
            texts.append(text)

        if self.master_process:
            print(f"[dataset] collected docs in memory: {len(texts)} (requested max_samples={dataset_cfg.max_samples})")

        train_texts = texts

        self.train_dataset = PackedTokenDataset(
            train_texts, tokenizer=self.tokenizer, seq_len=config.train.seq_len, add_eos_token=dataset_cfg.add_eos_token
        )
        self.val_dataset = None

        if self.master_process:
            print(f"[dataset] train samples={len(self.train_dataset)}")
        
        if self.master_process and len(self.train_dataset) < self.dp_world_size:
            print(
                f"[warn] train_dataset too small: {len(self.train_dataset)} samples "
                f"for dp_world_size={self.dp_world_size}. Increase max_samples or reduce seq_len."
            )
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        self.val_sampler = None
        self.val_loader = None
        self.num_train_samples = len(self.train_dataset)


def main():
    args = parse_args()
    cfg = build_config(args)

    assert cfg.train.do_val == False

    global dataset_cfg
    dataset_cfg = DatasetConfig()

    trainer = OurTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()