import torch
import torch.distributed as dist
import heapq
from collections import OrderedDict

class DistributedOptimizer:
    def __init__(
            self, 
            optimizer, 
            process_group,      # torch.distributed group, e.g. dp_group
            ranks_map=None, 
            num_parts=None, 
            verbose=False
        ):
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "process_group", process_group)
        object.__setattr__(self, "ranks_map", ranks_map)
        object.__setattr__(self, "num_parts", num_parts)
        object.__setattr__(self, "verbose", verbose)
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        if self.num_parts is None:
            self.num_parts = self.world_size
        assert 1 <= self.num_parts <= self.world_size
        self.apply_zero1()

    def apply_zero1(self, part_assignment = None):
        """
        Apply ZeRO-1 optimization by partitioning optimizer states across data parallel ranks.
        """
        self.tensor_dict: OrderedDict[tuple[int,int], torch.Tensor] = OrderedDict()
        self._key_by_param_id = {}
        for g_idx, group in enumerate(self.optimizer.param_groups):
            for p_idx, param in enumerate(group["params"]):
                if not isinstance(param, torch.Tensor):
                    continue
                if param.requires_grad:
                    key = (g_idx, p_idx)
                    self.tensor_dict[key] = param
                    self._key_by_param_id[id(param)] = key
        if part_assignment == None:
            if self.ranks_map is None:
                owners = list(range(self.num_parts))
            else:
                owners = [dist.get_group_rank(self.process_group, r) for r in self.ranks_map]  # global -> group
                self.num_parts = len(owners)
            assert 1 <= self.num_parts <= self.world_size
            assert len(set(owners)) == len(owners), f"Duplicate owners in ranks_map: {self.ranks_map}"
            part_assignment, _ = partition_tensors(
                self.tensor_dict,
                ranks_map=owners,
                num_parts=self.num_parts,
                verbose=self.verbose,
            )
        self.part_assignment = part_assignment
        for g_idx, group in enumerate(self.optimizer.param_groups):
            new_params = []
            for p_idx, param in enumerate(group["params"]):
                if not (isinstance(param, torch.Tensor) and param.requires_grad):
                    continue
                key = (g_idx, p_idx)
                part = self.part_assignment[key]
                if self.rank == part:
                    new_params.append(param)
            group["params"] = new_params
        dist.barrier(self.process_group)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def __setattr__(self, name, value):
        if name in {
            "optimizer", "process_group", "ranks_map", "num_parts", "partition_strategy",
            "verbose", "part_assignment", "tensor_dict", "_key_by_param_id",
            "rank", "world_size"
        } or name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.optimizer, name, value)

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__.keys()) + dir(self.optimizer)))

    @torch.no_grad()
    def step(self, *args, **kwargs):
        """
        1) Perform local optimizer step on locally-owned params only.
        2) Broadcast the updated local-owned params to all ranks in the group,
           so that all replicas have identical parameters before next forward.
        TODO: make broadcast overlap with optimizer step or next forward.
        """
        out = self.optimizer.step(*args, **kwargs)
        if self.world_size > 1:
            self._broadcast_all_params_from_owners()
        return out
    
    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for p in self.tensor_dict.values():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def _broadcast_all_params_from_owners(self):
        """
        Every rank must call broadcast on the same sequence of tensors with the same src.
        We iterate over the stable-ordered self.tensor_dict to ensure identical ordering.
        """
        # Early exit for single-process runs
        if self.world_size == 1:
            return
        for key, p in self.tensor_dict.items():
            group_src = self.part_assignment[key]
            # All ranks call broadcast for this tensor, using the same src
            src = dist.get_global_rank(self.process_group, group_src)
            dist.broadcast(p.data, src=src, group=self.process_group)


def partition_tensors(
    tensor_dict: "OrderedDict[tuple[int,int], torch.Tensor]",
    ranks_map: list = None,
    num_parts: int = None,
    verbose: bool = False,
    deterministic: bool = True,
):
    """
    Partition tensors across multiple parts (e.g., data parallel ranks) with greedy load balancing.

    Args:
        tensor_dict (OrderedDict): mapping (group_idx, param_idx) -> tensor.
        ranks_map (list, optional): explicit mapping of parts to rank ids. Defaults to None.
        num_parts (int, optional): number of partitions if ranks_map is not provided.
        verbose (bool): whether to print warnings and debug info.
        deterministic (bool): if True, tie-breaking is stable by sorting keys as secondary criterion.

    Returns:
        part_assignment (dict): mapping from (group_idx, param_idx) -> partition id.
        tensor_dict (OrderedDict): original tensor dict, returned for convenience.
    """
    if ranks_map is not None:
        num_parts = len(ranks_map)
    else:
        assert num_parts and num_parts > 0, "num_parts must be positive integer"

    # Collect tensors with sizes
    tensors = [(key, tensor.numel()) for key, tensor in tensor_dict.items()]

    # Sort by size descending, break ties by key if deterministic
    if deterministic:
        tensors.sort(key=lambda x: (-x[1], x[0]))
    else:
        tensors.sort(key=lambda x: -x[1])

    # Initialize heap: (current_size, part_id)
    heap = [(0, i) for i in range(num_parts)]
    heapq.heapify(heap)

    part_assignment = {key: None for key in tensor_dict.keys()}
    parts_sizes = {i: 0 for i in range(num_parts)}

    # Assign tensors greedily
    for key, size in tensors:
        cur_size, part_id = heapq.heappop(heap)

        owner = ranks_map[part_id] if ranks_map is not None else part_id
        part_assignment[key] = owner
        new_size = cur_size + size
        parts_sizes[part_id] = new_size

        heapq.heappush(heap, (new_size, part_id))

    # Debug / warnings
    if verbose:
        for part_id in range(num_parts):
            if parts_sizes[part_id] == 0:
                print(f"[ZeRO-1] Warning: Partition {part_id} is empty.")
        total_elems = sum(s for s in parts_sizes.values())
        max_size = max(parts_sizes.values())
        min_size = min(parts_sizes.values())
        imbalance = (max_size - min_size) / (total_elems / num_parts + 1e-6)
        print(f"[ZeRO-1] Partition load imbalance: {imbalance:.2%}")

    return part_assignment, tensor_dict