from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from nanorl.rl.reinforce import tokenize_with_assistant_mask
from nanorl.rollout.episode import EpisodeRollout


def _group_key(group_id: object) -> str | int | None:
    if group_id is None:
        return None
    if isinstance(group_id, (str, int)):
        return group_id
    return str(group_id)


def compute_grpo_advantages(
    rollouts: Sequence[EpisodeRollout],
) -> tuple[list[float], dict[str, float]]:
    rewards = [float(sum(rollout.rewards)) for rollout in rollouts]
    group_ids = [_group_key(rollout.group_id) for rollout in rollouts]

    if not rollouts:
        metrics = {
            "avg_reward": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "group_count": 0.0,
            "group_size_mean": 0.0,
        }
        return [], metrics

    if all(group_id is None for group_id in group_ids):
        group_map = {None: list(range(len(rollouts)))}
    else:
        group_map: dict[str | int | None, list[int]] = {}
        for idx, group_id in enumerate(group_ids):
            group_map.setdefault(group_id, []).append(idx)

    advantages = [0.0 for _ in rollouts]
    for indices in group_map.values():
        if not indices:
            continue
        group_rewards = [rewards[idx] for idx in indices]
        group_mean = sum(group_rewards) / len(group_rewards)
        for idx in indices:
            advantages[idx] = rewards[idx] - group_mean

    avg_reward = sum(rewards) / len(rewards)
    adv_mean = sum(advantages) / len(advantages)
    if len(advantages) > 1:
        variance = sum((adv - adv_mean) ** 2 for adv in advantages) / len(advantages)
        adv_std = math.sqrt(variance)
    else:
        adv_std = 0.0

    group_count = len(group_map)
    group_size_mean = sum(len(indices) for indices in group_map.values()) / max(group_count, 1)
    metrics = {
        "avg_reward": avg_reward,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "group_count": float(group_count),
        "group_size_mean": group_size_mean,
    }
    return advantages, metrics


def compute_grpo_loss(
    rollouts: Sequence[EpisodeRollout],
    advantages: Sequence[float],
    model: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    normalize_by_tokens: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(rollouts) != len(advantages):
        raise ValueError(f"rollouts and advantages length mismatch: {len(rollouts)} vs {len(advantages)}")

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_logprob = 0.0

    for rollout, advantage in zip(rollouts, advantages, strict=False):
        input_ids, mask = tokenize_with_assistant_mask(
            tokenizer,
            rollout.messages,
            device=device,
        )
        if not mask.any():
            continue

        logits = model(input_ids.unsqueeze(0)).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        target_ids = input_ids[1:]
        token_logprobs = log_probs[0, :-1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
        token_mask = mask[1:]

        masked_logprobs = token_logprobs[token_mask]
        if masked_logprobs.numel() == 0:
            continue

        logprob_term = masked_logprobs.mean() if normalize_by_tokens else masked_logprobs.sum()
        advantage_tensor = torch.as_tensor(float(advantage), device=device)
        total_loss = total_loss - advantage_tensor * logprob_term

        total_logprob += float(masked_logprobs.sum().item())
        total_tokens += int(token_mask.sum().item())

    loss = total_loss / max(len(rollouts), 1)
    metrics = {
        "avg_logprob": total_logprob / max(total_tokens, 1),
        "assistant_tokens": float(total_tokens),
    }
    return loss, metrics
