"""Prompt-aware ranking on top of the latent projector (phaseW/rank).

Pieces used by phaseW/train_pilot_frozen_s4j.py when --rank_negatives is set:

  RankHead          g_psi: a residual MLP on DINO-space unit vectors, identity at initialisation
                    (last layer zero), L2-normalised output. Applied to BOTH the projected
                    generation P(z_K) and the anchor u(x_ref), so R = <g(P(z_K)), g(u_ref)> equals
                    the plain projector reward at step 0 and lambda_rank = 0 recovers "ours" exactly.
  student_rollout   the student's own 4-step Euler sampling at guidance 1 for a batch of prompts
                    from ONE shared noise (paired: the prompts are the only difference), with or
                    without gradient through every step (Variant A backpropagates the ranking loss
                    through the whole chain; the Variant B arms sample without gradient).
  ranking_loss      partial order "positive > every negative": -log softmax(R / kappa)[positive].
  pairwise_acc      fraction of negatives the positive outranks.
  AlignHook         Variant B2: capture one DiT block's image-token stream during a forward and
                    project it (REPA's per-token MLP, mean-pooled) into the shaped DINO space.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankHead(nn.Module):
    def __init__(self, d: int = 768, width: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d, width)
        self.fc2 = nn.Linear(width, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return F.normalize(e + self.fc2(F.silu(self.fc1(e))), dim=-1)


def head_apply(head: RankHead, e: torch.Tensor, train_head: bool) -> torch.Tensor:
    """g(e). With train_head=False the head's weights are detached, so a loss on the result trains
    whatever produced `e` (the student, through the projector) but never the head."""
    if train_head:
        return head(e)
    params = {k: v.detach() for k, v in head.named_parameters()}
    return torch.func.functional_call(head, params, (e,))


def rollout_schedule(scheduler, steps: int, teacher_steps: int, device):
    """(sigmas, timesteps) of a `steps`-step schedule, leaving the scheduler set back to the
    teacher's `teacher_steps` so nothing else in the trainer sees a changed schedule."""
    scheduler.set_timesteps(steps, device=device)
    sig = scheduler.sigmas.to(device, torch.float32).clone()
    ts = scheduler.timesteps.to(device).clone()
    scheduler.set_timesteps(teacher_steps, device=device)
    return sig, ts


def student_rollout(model, z0: torch.Tensor, emb: torch.Tensor, pooled: torch.Tensor, sigmas: torch.Tensor,
                    timesteps: torch.Tensor, grad: bool, grad_last: bool = False, hook=None, hook_step: int = -1) -> torch.Tensor:
    """Euler sampling at guidance 1 (one conditional pass per step) for a batch of prompts.

    z0 [m,C,H,W] (bf16, typically one noise expanded over the m prompts), emb [m,L,D], pooled [m,P].
    grad=True keeps the autograd graph through every step (Variant A); grad_last=True keeps it
    through ONE forward only, the hooked step (Variant B2's alignment term reads that forward's
    hidden state through `hook`): hook_step = -1 is the last step (input sigma ~0.009, the review's
    "autoencoding" regime), 0 the first (sigma 1.0), 1 the mid step (sigma 0.86). Returns the
    terminal latent in float32.
    """
    n = len(sigmas) - 1
    hs = hook_step % n
    z = z0
    for k in range(n):
        last = k == n - 1
        hooked = k == hs
        ctx = torch.enable_grad() if (grad or (grad_last and hooked)) else torch.no_grad()
        with ctx:
            if hook is not None and hooked:
                hook.arm()
            with torch.autocast("cuda", torch.bfloat16):
                v = model(hidden_states=z, timestep=timesteps[k].expand(z.shape[0]),
                          encoder_hidden_states=emb, pooled_projections=pooled, return_dict=False)[0]
            if hook is not None and hooked:
                hook.disarm()
            z_next = z.float() + (sigmas[k + 1] - sigmas[k]) * v.float()
        # keep the graph only where it is wanted: after the hooked step in grad_last mode the chain
        # is cut so the alignment gradient reaches the student through that forward alone
        z = z_next.to(torch.bfloat16) if not last else z_next
        if grad_last and not grad and hooked:
            z = z.detach()
    return z.float()


def rank_rows(head, e_pos: torch.Tensor, e_negs, u: torch.Tensor, train_head: bool) -> torch.Tensor:
    """Reward rows R [S, 1 + n_neg]: one row per state s, column 0 the positive prompt's embedding
    e_pos[s], columns 1.. the negatives' e_negs[j, s]; every embedding and the anchor u [1, d] go
    through the head (identity if head is None). e_negs may be None (no negatives: R is [S, 1])."""
    S = e_pos.shape[0]
    cols = [e_pos]
    if e_negs is not None and e_negs.shape[0] > 0:
        cols.append(e_negs.transpose(0, 1).reshape(-1, e_pos.shape[-1]))      # [S*n_neg, d], state-major
    E = torch.cat(cols, 0)
    if head is not None:
        E = head_apply(head, E, train_head)
        u = head_apply(head, u, train_head)
    R = (E * u).sum(-1)
    R_pos = R[:S]
    R_neg = R[S:].view(S, -1) if R.shape[0] > S else R.new_zeros(S, 0)
    return torch.cat([R_pos[:, None], R_neg], 1)


def rows_loss(R: torch.Tensor, kappa: float) -> torch.Tensor:
    """Mean over states of the partial-order loss; 0 (with graph) when there is no negative."""
    if R.shape[1] < 2:
        return R.sum() * 0.0
    logits = R / kappa
    return (torch.logsumexp(logits, 1) - logits[:, 0]).mean()


def rows_acc(R: torch.Tensor) -> float:
    return float("nan") if R.shape[1] < 2 else float((R[:, :1] > R[:, 1:]).float().mean())


def rows_margin(R: torch.Tensor) -> float:
    return float("nan") if R.shape[1] < 2 else float((R[:, 0] - R[:, 1:].mean(1)).mean())


def ranking_loss(R: torch.Tensor, kappa: float) -> torch.Tensor:
    """R [m] with R[0] the positive prompt's reward: -log softmax(R/kappa)[0]."""
    logits = R / kappa
    return torch.logsumexp(logits, 0) - logits[0]


def pairwise_acc(R: torch.Tensor) -> float:
    if R.numel() < 2:
        return float("nan")
    return float((R[0] > R[1:]).float().mean())


def margin(R: torch.Tensor) -> float:
    if R.numel() < 2:
        return float("nan")
    return float(R[0] - R[1:].mean())


class AlignHook:
    """Forward hook on one JointTransformerBlock of the SD3.5 DiT that keeps the IMAGE-token stream
    (out[-1]) while armed. The projector is REPA's per-token MLP (width 2048), mean-pooled over
    tokens and L2-normalised, so the alignment target is one unit vector per prompt."""

    def __init__(self, transformer, layer: int, in_dim: int, out_dim: int = 768, width: int = 2048, device=None):
        blocks = transformer.transformer_blocks
        assert 1 <= layer <= len(blocks), f"align_layer {layer} must be in [1, {len(blocks)}]"
        self.layer, self._armed, self._h = layer, False, None
        self.proj = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, out_dim)).to(device)

        def _hook(_mod, _inp, out):
            if self._armed:
                self._h = out[-1] if isinstance(out, (tuple, list)) else out

        self._handle = blocks[layer - 1].register_forward_hook(_hook)

    def arm(self):
        self._h = None
        self._armed = True

    def disarm(self):
        self._armed = False

    def features(self) -> torch.Tensor:
        """[m, out_dim] unit vectors from the captured stream; raises if the hook did not fire or
        the captured activation carries no grad (reentrant checkpointing would sever the student)."""
        h = self._h
        if h is None:
            raise RuntimeError(f"align hook on block {self.layer} did not fire")
        if not h.requires_grad:
            raise RuntimeError("align hook: captured hidden state has no grad_fn (reentrant checkpointing?)")
        return F.normalize(self.proj(h.float()).mean(1), dim=-1)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def all_reduce_grads(module: nn.Module, world: int) -> None:
    if world <= 1:
        return
    import torch.distributed as dist
    for p in module.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad)
        p.grad.div_(world)


def broadcast_params(module: nn.Module, world: int) -> None:
    if world <= 1:
        return
    import torch.distributed as dist
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def rollout_noise(seed: int, rank: int, micro: int, shape, device) -> torch.Tensor:
    """One private noise per (run seed, rank, caption visit): the training RNG streams are untouched."""
    g = torch.Generator(device=device).manual_seed((seed * 1_000_003 + rank * 1009 + micro * 97 + 12345) % (2 ** 62))
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)


__all__ = ["RankHead", "head_apply", "rollout_schedule", "student_rollout", "ranking_loss", "pairwise_acc", "margin",
           "rank_rows", "rows_loss", "rows_acc", "rows_margin",
           "AlignHook", "all_reduce_grads", "broadcast_params", "rollout_noise", "math"]
