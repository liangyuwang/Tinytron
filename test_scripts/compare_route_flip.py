import json
import sys
from collections import defaultdict


def load_trace(path: str) -> dict[tuple[int, int, int, int, int], tuple[int, ...]]:
    records = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                int(row["step"]),
                int(row["micro_step"]),
                int(row["layer"]),
                int(row["sample_idx"]),
                int(row["token_pos"]),
            )
            records[key] = tuple(int(x) for x in row["topk"])
    return records


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_scripts/compare_route_flip.py <dp_trace.jsonl> <sp_ep_trace.jsonl>")
        sys.exit(1)

    dp_path = sys.argv[1]
    sp_ep_path = sys.argv[2]
    dp = load_trace(dp_path)
    sp = load_trace(sp_ep_path)
    overlap = sorted(set(dp.keys()) & set(sp.keys()))

    if not overlap:
        print("No overlapping route keys found. Cannot compute flip rate.")
        print(f"dp_records={len(dp)}, sp_ep_records={len(sp)}")
        sys.exit(0)

    top1_flip = 0
    topk_flip = 0
    per_step = defaultdict(lambda: {"n": 0, "top1": 0, "topk": 0})

    for k in overlap:
        a = dp[k]
        b = sp[k]
        step = k[0]
        is_top1_flip = int(a[0] != b[0])
        is_topk_flip = int(a != b)
        top1_flip += is_top1_flip
        topk_flip += is_topk_flip
        per_step[step]["n"] += 1
        per_step[step]["top1"] += is_top1_flip
        per_step[step]["topk"] += is_topk_flip

    n = len(overlap)
    print(f"DP trace:      {dp_path}")
    print(f"SP+EP trace:   {sp_ep_path}")
    print(f"overlap keys:  {n}")
    print(f"top1 flip rate: {top1_flip / n:.6%} ({top1_flip}/{n})")
    print(f"topk flip rate: {topk_flip / n:.6%} ({topk_flip}/{n})")
    print(f"dp-only keys:   {len(dp) - n}")
    print(f"sp-only keys:   {len(sp) - n}")
    print("")
    print("Per-step route flip:")
    for step in sorted(per_step.keys()):
        row = per_step[step]
        print(
            f"  step {step:4d} | overlap={row['n']:6d} "
            f"| top1={row['top1']/row['n']:.6%} ({row['top1']:5d}) "
            f"| topk={row['topk']/row['n']:.6%} ({row['topk']:5d})"
        )


if __name__ == "__main__":
    main()
