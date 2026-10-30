"""Custom advantage estimators for continuous (partial-credit) rewards.

Registered into verl's advantage registry on import; select with
``algorithm.adv_estimator=<name>``. The import must happen inside the trainer
process (see scripts/main_ppo_plugins.py).

drgrpo_scaled
    A_i = (r_i - mean_g) / c, c fixed (env ADV_SCALE, default 18).
    This is verl's GRPO centering with the per-group std normalization removed
    (Dr. GRPO) and a sample-independent scale, so the policy gradient is
    unbiased for J = E[r] up to the constant (n-1)/(n*c). Groups whose scores
    are all equal get 0, as in GRPO. c should be about the RMS within-group
    reward std of plain-GRPO rollouts (18 on the FrontierSmith baseline), which
    makes RMS(A) match GRPO's so the KL coefficient and grad clip keep their
    meaning. Pair it with actor.loss_agg_mode=seq-mean-token-sum-norm; the
    default per-sequence token-mean aggregation is itself biased.

grpo_quantized
    Binarize each rollout's reward against its group mean, then run vanilla GRPO:
        b_i = 1  if r_i > mean_g + tol,   else 0,   tol = 1e-6 * max(1, |mean_g|)
        A_i = GRPO(b)_i = (b_i - mean_g(b)) / (std_g(b) + eps)
    (or b_i - mean_g(b) with algorithm.norm_adv_by_std_in_grpo=False). r_i is the
    sequence reward (token_level_rewards summed over the response); scores equal
    to the mean up to float rounding count as NOT higher, so a flat group gives
    b = 0 everywhere and A = 0. A group of k "above-average" rollouts out of n
    gets +sqrt((n-k)(n-1)/(n k)) for them and -sqrt(k(n-1)/(n(n-k))) for the
    rest, whatever the raw margins were. On rewards that are already 0/1 this
    is exactly vanilla GRPO. Singleton groups (n=1) use vanilla GRPO's
    convention (baseline 0, std 1): b = 1[r > 0], A = b / (1 + eps).
    The raw judge scores are still what is logged (critic/score, val); only the
    advantages see the binarized reward. epsilon and norm_adv_by_std_in_grpo
    are those of vanilla GRPO (algorithm.norm_adv_by_std_in_grpo, default True).
"""

from __future__ import annotations

import os

import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, register_adv_est


@register_adv_est("drgrpo_scaled")
def compute_drgrpo_scaled_advantage(token_level_rewards, response_mask, index, config=None, **kwargs):
    scale = float(os.environ.get("ADV_SCALE", "18"))
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=False,
    )
    advantages = advantages / scale
    return advantages, advantages


def _config_get(config, key, default):
    """Read ``key`` from an AlgoConfig, dict, DictConfig or None."""
    if config is None:
        return default
    getter = getattr(config, "get", None)
    value = getter(key, default) if callable(getter) else getattr(config, key, default)
    return default if value is None else value


def quantize_by_group_mean(scores: torch.Tensor, index, rel_tol: float = 1e-6) -> torch.Tensor:
    """b_i = 1.0 if scores[i] > mean of its group (same index) + tol, else 0.0; same dtype/device as scores.

    Groups of size 1 use baseline 0 (vanilla GRPO's convention). Means are taken in float64.
    """
    if len(index) != scores.shape[0]:
        raise ValueError(f"index has {len(index)} entries but there are {scores.shape[0]} scores")
    if scores.numel() == 0:
        return torch.zeros_like(scores)
    group_of = {}
    gid = torch.tensor([group_of.setdefault(key, len(group_of)) for key in index], device=scores.device)
    s = scores.detach().to(torch.float64)
    sums = torch.zeros(len(group_of), dtype=torch.float64, device=s.device).index_add_(0, gid, s)
    counts = torch.zeros_like(sums).index_add_(0, gid, torch.ones_like(s))
    means = torch.where(counts > 1, sums / counts, torch.zeros_like(sums))[gid]
    tol = rel_tol * means.abs().clamp(min=1.0)
    return (s > means + tol).to(scores.dtype)


def _put_on_last_valid_token(values: torch.Tensor, response_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Tensor shaped like ``like``, zero except ``values[i]`` at row i's last unmasked position (0 if none)."""
    positions = torch.arange(like.shape[-1], device=like.device).expand(like.shape)
    last = torch.where(response_mask.bool(), positions, torch.full_like(positions, -1)).amax(dim=-1).clamp(min=0)
    out = torch.zeros_like(like)
    out.scatter_(-1, last.unsqueeze(-1), values.to(like.dtype).unsqueeze(-1))
    return out


@register_adv_est("grpo_quantized")
def compute_grpo_quantized_advantage(
    token_level_rewards,
    response_mask,
    index,
    config=None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool | None = None,
    **kwargs,
):
    if norm_adv_by_std_in_grpo is None:
        norm_adv_by_std_in_grpo = bool(_config_get(config, "norm_adv_by_std_in_grpo", True))
    binary = quantize_by_group_mean(token_level_rewards.detach().sum(dim=-1), index)
    quantized_rewards = _put_on_last_valid_token(binary, response_mask, like=token_level_rewards.detach())
    return compute_grpo_outcome_advantage(
        token_level_rewards=quantized_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
