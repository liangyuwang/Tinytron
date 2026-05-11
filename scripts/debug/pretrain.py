from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinytron.training import parse_args, build_config, Trainer

# You can also modify the trainer class to customize the training process, 
# and model modules to customize the model architecture.
# Following is a minimal example of how to use the trainer class.

def main():
    args = parse_args()
    cfg = build_config(args)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
