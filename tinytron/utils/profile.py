import torch

GPU_PEAK_FLOPS = {
    "T4":   {"fp32": 8.1e12, "fp16": 65e12, "bf16": 0},
    "V100": {"fp32": 15.7e12, "fp16": 125e12, "bf16": 0},
    "A100": {"fp32": 19.5e12, "fp16": 312e12, "bf16": 312e12},
    "A40":  {"fp32": 37.4e12, "fp16": 149e12, "bf16": 149e12},
    "A5000": {"fp32": 27.8e12, "fp16": 111.2e12, "bf16": 111.2e12},
    "A30":  {"fp32": 10.3e12, "fp16": 165e12, "bf16": 165e12},
    "RTX 6000 Ada": {"fp32": 91e12, "fp16": 181e12, "bf16": 181e12},
    "L4":   {"fp32": 30e12, "fp16": 120e12, "bf16": 120e12},
    "L40":  {"fp32": 91e12, "fp16": 181e12, "bf16": 181e12},
    "L40S": {"fp32": 91e12, "fp16": 181e12, "bf16": 181e12},
    "3090": {"fp32": 35.6e12, "fp16": 142e12, "bf16": 142e12},
    "4090": {"fp32": 82.6e12, "fp16": 330e12, "bf16": 330e12},
    "H100": {"fp32": 60e12, "fp16": 989e12, "bf16": 989e12, "fp8": 1979e12},
    "H800": {"fp32": 34e12, "fp16": 734e12, "bf16": 734e12, "fp8": 1468e12},
    "H200": {"fp32": 67e12, "fp16": 989e12, "bf16": 989e12, "fp8": 1979e12},
    "H20":  {"fp32": 21e12, "fp16": 494e12, "bf16": 494e12, "fp8": 988e12},
}

_WARNED_GPU_DTYPE: set[tuple[str, str]] = set()


def get_gpu_peak_flops(dtype="bf16", per_device=True):
    """Detect GPU type and return theoretical peak FLOPs/s (for all GPUs)."""
    gpu_name = torch.cuda.get_device_name(0)
    dtype = dtype.lower()
    peak = None
    for k, v in GPU_PEAK_FLOPS.items():
        if k in gpu_name:
            peak = v.get(dtype, None)
            break
    if peak is None or peak == 0:
        warn_key = (gpu_name, dtype)
        if warn_key not in _WARNED_GPU_DTYPE:
            print(f"Warning: unknown or unsupported FLOPs for GPU {gpu_name} with dtype {dtype}")
            _WARNED_GPU_DTYPE.add(warn_key)
        peak = 0
    num_gpus = torch.cuda.device_count()
    return peak if per_device else peak * num_gpus

def compute_mfu(model, batch_size, seq_len, time, ga=1, dtype="bf16"):
    """
    Approximate MFU per training step.
    """
    total_flops = model.get_flops_per_fwd_bwd(batch_size, seq_len)
    total_flops *= ga   # gradient accumulation
    actual_flops_per_sec = total_flops / time if time > 0 else 0.0
    peak_flops = get_gpu_peak_flops(dtype=dtype, per_device=True)
    mfu = actual_flops_per_sec / peak_flops if peak_flops != 0 else 0
    return mfu, actual_flops_per_sec, peak_flops
