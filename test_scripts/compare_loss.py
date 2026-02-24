import json
import os
import matplotlib.pyplot as plt


def _read_log(log_file):
    cfg = None
    steps = []
    losses = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            if cfg is None and isinstance(data, dict) and "config" in data:
                cfg = data["config"]
            if data.get("stage") == "train":
                steps.append(int(data["step"]))
                losses.append(float(data["loss"]))
    return cfg, steps, losses


def _format_cfg(label, cfg):
    if not cfg:
        return f"{label}: <no config found>"
    train = cfg.get("train", {})
    parallel = cfg.get("parallel", {})
    model = cfg.get("model", {})
    seed = cfg.get("seed", {})

    parts = [
        f"{label}: sep_size={parallel.get('sep_size', 'NA')}, batch={train.get('batch_size', 'NA')}, seq={train.get('seq_len', 'NA')}",
        f"  total_batch={train.get('total_batch_size', 'NA')}, precision={train.get('precision', 'NA')}, tf32_off={train.get('disable_tf32', False)}, deterministic={seed.get('deterministic', False)}",
        f"  max_steps={train.get('max_steps', 'NA')}, max_epochs={train.get('max_epochs', 'NA')}",
    ]
    if bool(model.get("use_moe", False)):
        parts.append(
            "  MoE: enabled, "
            f"num_experts={model.get('num_experts', 'NA')}, topk={model.get('num_experts_per_tok', 'NA')}, "
            f"moe_hidden={model.get('moe_intermediate_size', 'NA')}"
        )
    else:
        parts.append("  MoE: disabled (dense MLP)")
    return "\n".join(parts)


def plot_jsonl_loss(files, labels, output_fig):
    fig, ax = plt.subplots(figsize=(12, 7))
    run_cfg_texts = []
    runs = []

    for file, label in zip(files, labels):
        cfg, steps, losses = _read_log(file)
        runs.append((label, steps, losses))
        run_cfg_texts.append(_format_cfg(label, cfg))
        ax.plot(steps, losses, label=label, linestyle="-", marker="o", markersize=3.5, linewidth=1.3, alpha=0.85)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Comparison (DP vs SP+EP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if len(runs) >= 2:
        diff_lines = []
        maps = []
        for label, steps, losses in runs:
            maps.append({k: v for k, v in zip(steps, losses)})
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                l_i, l_j = runs[i][0], runs[j][0]
                common = sorted(set(maps[i]) & set(maps[j]))
                if common:
                    diffs = [abs(maps[i][k] - maps[j][k]) for k in common]
                    diff_lines.append(
                        f"{l_i} vs {l_j}:  n={len(common)}  max={max(diffs):.3e}  mean={sum(diffs)/len(diffs):.3e}"
                    )
        if diff_lines:
            ax.text(
                0.99,
                0.01,
                "\n".join(diff_lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, boxstyle="round,pad=0.25"),
            )

    cfg_text = "\n".join(run_cfg_texts)
    fig.tight_layout()
    fig.savefig(output_fig, dpi=160)
    print(f"Saved plot to {output_fig}")
    print(f"Plot config summary:\n{cfg_text}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python compare_loss.py <log1.txt> <label1> <log2.txt> <label2> [output_fig]")
        sys.exit(1)

    output_fig = "loss_comparison.png"
    if len(sys.argv) % 2 == 0:
        output_fig = sys.argv[-1]
        args = sys.argv[1:-1]
    else:
        args = sys.argv[1:]

    if len(args) % 2 != 0:
        raise ValueError("Input arguments must be pairs of <log> <label>.")

    files = []
    labels = []
    for i in range(0, len(args), 2):
        files.append(args[i])
        labels.append(args[i + 1])

    output_dir = os.path.dirname(output_fig)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plot_jsonl_loss(files, labels, output_fig)
