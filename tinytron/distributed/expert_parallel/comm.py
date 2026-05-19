import os
import time

import torch
import torch._dynamo
import torch.distributed as dist


def _trace(message: str, tensor: torch.Tensor | None = None, *, sync: bool = False):
    if os.environ.get("TINYTRON_EP_TRACE", "0") != "1":
        return
    if sync and tensor is not None and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    now = time.strftime("%H:%M:%S")
    print(f"[ep-trace {now} rank={rank}] {message}", flush=True)


def _all_to_all_single_min1_splits(input_tensor, input_splits, output_splits, group):
    input_splits = [int(x) for x in input_splits]
    output_splits = [int(x) for x in output_splits]
    local_zero_total = torch.tensor(
        int(sum(input_splits) == 0),
        device=input_tensor.device,
        dtype=torch.int32,
    )
    _trace(f"zero-total check start local={int(local_zero_total.item())}", input_tensor, sync=True)
    dist.all_reduce(local_zero_total, op=dist.ReduceOp.MAX, group=group)
    _trace(f"zero-total check done any={int(local_zero_total.item())}", input_tensor, sync=True)
    if local_zero_total.item() != 0:
        return _all_to_all_via_all_gather(input_tensor, input_splits, output_splits, group)

    padded_input_splits = [max(x, 1) for x in input_splits]
    padded_output_splits = [max(x, 1) for x in output_splits]
    trailing_shape = tuple(input_tensor.shape[1:])

    chunks = []
    start = 0
    for actual in input_splits:
        if actual == 0:
            chunks.append(input_tensor.new_zeros((1, *trailing_shape)))
        else:
            chunks.append(input_tensor.narrow(0, start, actual))
        start += actual
    padded_input = torch.cat(chunks, dim=0).contiguous()

    padded_output = torch.empty(
        (sum(padded_output_splits), *trailing_shape),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).contiguous()
    dist.all_to_all_single(
        padded_output,
        padded_input,
        output_split_sizes=padded_output_splits,
        input_split_sizes=padded_input_splits,
        group=group,
    )

    outputs = []
    start = 0
    for padded, actual in zip(padded_output_splits, output_splits):
        chunk = padded_output.narrow(0, start, padded)
        outputs.append(chunk[:actual])
        start += padded
    return torch.cat(outputs, dim=0).contiguous()


def _all_to_all_via_all_gather(input_tensor, input_splits, output_splits, group):
    world_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    trailing_shape = tuple(input_tensor.shape[1:])
    _trace(
        f"all_gather fallback start input={input_splits} output={output_splits} rows={input_tensor.size(0)}",
        input_tensor,
        sync=True,
    )

    local_splits = torch.tensor(input_splits, device=input_tensor.device, dtype=torch.long)
    gathered_splits = [torch.empty_like(local_splits) for _ in range(world_size)]
    dist.all_gather(gathered_splits, local_splits, group=group)
    _trace("all_gather fallback splits done", input_tensor, sync=True)

    local_rows = torch.tensor(input_tensor.size(0), device=input_tensor.device, dtype=torch.long)
    max_rows = local_rows.clone()
    dist.all_reduce(max_rows, op=dist.ReduceOp.MAX, group=group)
    max_rows_int = int(max_rows.item())

    if input_tensor.size(0) < max_rows_int:
        pad = input_tensor.new_zeros((max_rows_int - input_tensor.size(0), *trailing_shape))
        padded_input = torch.cat([input_tensor, pad], dim=0).contiguous()
    else:
        padded_input = input_tensor.contiguous()

    gathered_inputs = [torch.empty_like(padded_input) for _ in range(world_size)]
    dist.all_gather(gathered_inputs, padded_input, group=group)
    _trace("all_gather fallback tensors done", input_tensor, sync=True)

    outputs = []
    for src_rank, (src_tensor, src_splits_tensor, actual) in enumerate(
        zip(gathered_inputs, gathered_splits, output_splits)
    ):
        if actual == 0:
            continue
        src_splits = [int(x) for x in src_splits_tensor.detach().cpu().tolist()]
        start = sum(src_splits[:group_rank])
        outputs.append(src_tensor.narrow(0, start, actual))

    if not outputs:
        result = input_tensor.new_empty((0, *trailing_shape))
    else:
        result = torch.cat(outputs, dim=0).contiguous()
    _trace(f"all_gather fallback done rows={result.size(0)}", result, sync=True)
    return result


class EPAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, input_splits, output_splits, ep_group):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.ep_group = ep_group
        _trace(f"forward start input={list(map(int, input_splits))} output={list(map(int, output_splits))}", hidden_states, sync=True)
        out = _all_to_all_single_min1_splits(
            hidden_states.contiguous(),
            input_splits,
            output_splits,
            ep_group,
        )
        _trace(f"forward done rows={out.size(0)}", out, sync=True)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        _trace(
            f"backward start input={list(map(int, ctx.output_splits))} output={list(map(int, ctx.input_splits))}",
            grad_output,
            sync=True,
        )
        grad_input = _all_to_all_single_min1_splits(
            grad_output.contiguous(),
            ctx.output_splits,
            ctx.input_splits,
            ctx.ep_group,
        )
        _trace(f"backward done rows={grad_input.size(0)}", grad_input, sync=True)
        return grad_input, None, None, None

@torch._dynamo.disable  # torch compile may remove .contiguous()
def ep_all_to_all(hidden_states, input_splits, output_splits, ep_group):
    return EPAllToAll.apply(hidden_states, input_splits, output_splits, ep_group)


@torch._dynamo.disable
def ep_all_to_all_no_grad(tensor, input_splits, output_splits, ep_group):
    return _all_to_all_single_min1_splits(
        tensor.contiguous(),
        input_splits,
        output_splits,
        ep_group,
    )
