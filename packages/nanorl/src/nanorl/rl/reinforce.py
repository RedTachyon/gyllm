from typing import Any

import torch

from gyllm.envs import Message
from nanorl.rollout.episode import EpisodeRollout


def _encode_chat_messages(
    tokenizer: Any,
    messages: list[Message],
) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded["input_ids"][0]


def tokenize_with_assistant_mask(
    tokenizer: Any,
    messages: list[Message],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = _encode_chat_messages(tokenizer, messages).to(device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    prev_len = 0
    for idx, message in enumerate(messages):
        prefix_ids = _encode_chat_messages(tokenizer, messages[: idx + 1]).to(device)
        curr_len = prefix_ids.shape[0]
        if message["role"] == "assistant":
            mask[prev_len:curr_len] = True
        prev_len = curr_len

    return input_ids, mask


def compute_reinforce_loss(
    rollouts: list[EpisodeRollout],
    model: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    normalize_by_tokens: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = torch.tensor([sum(r.rewards) for r in rollouts], device=device)
    baseline = rewards.mean()

    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_logprob = 0.0

    for rollout, reward in zip(rollouts, rewards, strict=False):
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

        advantage = reward - baseline
        total_loss = total_loss - advantage * logprob_term

        total_logprob += float(masked_logprobs.sum().item())
        total_tokens += int(token_mask.sum().item())

    loss = total_loss / max(len(rollouts), 1)
    metrics = {
        "avg_reward": float(rewards.mean().item()) if rollouts else 0.0,
        "baseline": float(baseline.item()) if rollouts else 0.0,
        "avg_logprob": total_logprob / max(total_tokens, 1),
        "assistant_tokens": float(total_tokens),
    }
    return loss, metrics
