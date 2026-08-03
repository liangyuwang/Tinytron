"""aimegatron.parallel.pipeline

Pipeline-parallel micro-batch scheduling. The schedule is a strategy
object: the trainer hands it the stage module and a batch provider, and it
runs one optimizer step's worth of forward/backward with P2P activation
handoff between neighboring stages.

Implemented schedule: canonical 1F1B (warmup forwards, steady-state
one-forward-one-backward, cooldown backwards). P2P connects identical
(dp, tp) positions of neighboring stages, so with sequence parallelism on,
sequence-sharded activations flow across stage boundaries unchanged.

Edit contract: the schedule owns P2P and micro-batch bookkeeping only.
Gradient finalization / clipping / optimizer steps stay in the trainer.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.distributed as dist

from aimegatron.core import mesh


class OneFOneBSchedule:

    def __init__(self, stage, batch_size: int, seq_len: int, hidden_size: int,
                 dtype: torch.dtype, num_microbatches: int, aux_loss_coeff: float = 0.0):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.aux_loss_coeff = aux_loss_coeff
        self.device = next(stage.parameters()).device
        self.dtype = dtype

        self.pp_rank = mesh.get_pp_rank()
        self.pp_size = mesh.get_pp_world_size()
        stride = mesh.get_dp_world_size() * mesh.get_tp_world_size()
        global_rank = dist.get_rank()
        self.prev_rank = global_rank - stride if self.pp_rank > 0 else None
        self.next_rank = global_rank + stride if self.pp_rank < self.pp_size - 1 else None

        seq_dim = seq_len // mesh.get_tp_world_size() if mesh.sequence_parallel() else seq_len
        self.activation_shape = (batch_size, seq_dim, hidden_size)
        # Outstanding async sends. Gloo sends are rendezvous, so a send must
        # not be waited on immediately: while this rank waits, the peer may
        # be blocked on its own send (grad vs activation), deadlocking the
        # pair. Sends are posted async and drained at the end of the step;
        # tensor references are kept alive here until completion.
        self._pending_sends = []

    # -- P2P primitives -----------------------------------------------------

    def _send(self, tensor: torch.Tensor, dst: int) -> None:
        work = dist.isend(tensor.contiguous(), dst)
        self._pending_sends.append((tensor, work))

    def _recv(self, src: int, requires_grad: bool = False) -> torch.Tensor:
        buf = torch.empty(self.activation_shape, dtype=self.dtype, device=self.device)
        dist.irecv(buf, src).wait()
        if requires_grad:
            buf.requires_grad_(True)
        return buf

    def _drain_sends(self) -> None:
        for _, work in self._pending_sends:
            work.wait()
        self._pending_sends.clear()

    # -- one optimizer step -------------------------------------------------

    def run_step(self, batch_provider) -> float | None:
        """Execute num_microbatches forwards and backwards in 1F1B order.
        `batch_provider()` yields the next micro-batch dict (every stage
        consumes the stream; only first/last use its contents). Returns the
        step's logging loss on the last stage, None elsewhere."""
        nm = self.num_microbatches
        num_warmup = min(self.pp_size - self.pp_rank - 1, nm)

        inputs, outputs, auxes = deque(), deque(), deque()
        scaled_losses = deque()          # last stage only
        logging_sum = 0.0

        def forward_micro():
            nonlocal logging_sum
            batch = batch_provider()
            if self.stage.is_first:
                input_ids = batch["input_ids"].to(self.device)
                out, aux = self.stage(input_ids)
                inputs.append(None)          # embedding input needs no grad
            else:
                hidden = self._recv(self.prev_rank, requires_grad=True)
                inputs.append(hidden)
                if self.stage.is_last:
                    labels = batch["labels"].to(self.device)
                    loss, logging_loss, aux = self.stage(hidden, labels)
                    scaled = loss / nm
                    if aux is not None:
                        scaled = scaled + self.aux_loss_coeff * aux / nm
                    scaled_losses.append(scaled)
                    logging_sum += logging_loss.item() / nm
                    outputs.append(None)     # backward keys off scaled_losses
                    auxes.append(None)
                    return
                out, aux = self.stage(hidden)
            if self.next_rank is not None:
                self._send(out, self.next_rank)
            outputs.append(out)
            auxes.append(aux)

        def backward_micro():
            if self.stage.is_last:
                scaled_losses.popleft().backward()
                auxes.popleft()
                input_t = inputs.popleft()
            else:
                out, aux = outputs.popleft(), auxes.popleft()
                input_t = inputs.popleft()
                out_grad = self._recv(self.next_rank)
                need_aux = aux is not None and aux.requires_grad
                out.backward(out_grad, retain_graph=need_aux)
                if need_aux:
                    (self.aux_loss_coeff * aux / nm).backward()
            if input_t is not None:
                self._send(input_t.grad, self.prev_rank)

        for _ in range(num_warmup):
            forward_micro()
        for _ in range(nm - num_warmup):
            forward_micro()
            backward_micro()
        for _ in range(num_warmup):
            backward_micro()
        self._drain_sends()

        if self.stage.is_last:
            return logging_sum
        return None

    def broadcast_loss(self, local_loss: float | None) -> float:
        """Share the last stage's logging loss with the whole PP group."""
        if self.pp_size == 1:
            return local_loss if local_loss is not None else 0.0
        t = torch.tensor([local_loss if local_loss is not None else 0.0],
                         dtype=torch.float32)
        # The last stage owns the value; every other stage contributes 0.
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=mesh.get_pp_group())
        return t.item()
