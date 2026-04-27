from __future__ import annotations

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
