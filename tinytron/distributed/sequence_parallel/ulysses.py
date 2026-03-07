import torch
import torch._dynamo
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class UlyssesAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, sp_group):
        ctx.sp_group = sp_group
        input_tensor = input_tensor.contiguous()
        output_tensor = torch.empty_like(input_tensor)
        dist.all_to_all_single(output_tensor, input_tensor, group=sp_group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        sp_group = ctx.sp_group
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output, group=sp_group)
        return grad_input, None

@torch._dynamo.disable  # torch compile may remove .contiguous()
def ulysses_all_to_all(input_tensor, sp_group):
    return UlyssesAllToAll.apply(input_tensor, sp_group)


@torch.no_grad()
def allreduce_non_expert_grads_across_sp(
    model,
    sp_group,
    sp_world_size: int,
    expert_local_param_suffixes: tuple[str, ...],
    bucket_size_mb: int = 25,
):
    def _yield_tensor_buckets(
        tensors: list[torch.Tensor],
        bucket_size_mb: int,
    ):
        bucket_size_bytes = max(1, bucket_size_mb) * 1024 * 1024
        current_bucket: list[torch.Tensor] = []
        current_bucket_bytes = 0

        for tensor in tensors:
            tensor_bytes = tensor.numel() * tensor.element_size()

            # If adding this tensor would overflow the bucket, flush current bucket first.
            if current_bucket and current_bucket_bytes + tensor_bytes > bucket_size_bytes:
                yield current_bucket
                current_bucket = []
                current_bucket_bytes = 0

            # A single large tensor may exceed bucket_size_bytes and will form its own bucket.
            current_bucket.append(tensor)
            current_bucket_bytes += tensor_bytes

        if current_bucket:
            yield current_bucket

    def _allreduce_tensor_bucket(
        tensors: list[torch.Tensor],
        group,
    ):
        flat_buffer = parameters_to_vector(tensors)
        dist.all_reduce(flat_buffer, op=dist.ReduceOp.SUM, group=group)
        # flat_buffer.div_(sp_world_size)  # enable if average is desired
        vector_to_parameters(flat_buffer, tensors)

    if sp_world_size <= 1:
        return

    grads_by_device_dtype: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if name.endswith(expert_local_param_suffixes):
            continue

        key = (grad.device, grad.dtype)
        grads_by_device_dtype.setdefault(key, []).append(grad)

    for grads in grads_by_device_dtype.values():
        for bucket in _yield_tensor_buckets(grads, bucket_size_mb=bucket_size_mb):
            _allreduce_tensor_bucket(bucket, group=sp_group)

# Example usage
if __name__ == "__main__":
    # Create a dummy tensor
    input_tensor = torch.randn(10, 10)
    # Create a dummy communication group
    sp_group = dist.new_group(backend='nccl', ranks=list(range(4)))
    # Forward pass
    output_tensor = ulysses_all_to_all(input_tensor, sp_group)
    # Backward pass
    grad_input = torch.randn_like(output_tensor)
    grad_output = ulysses_all_to_all(grad_input, sp_group)
    print(grad_output)