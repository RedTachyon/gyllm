from collections.abc import Sequence
from html import escape
from typing import Any, Protocol

import torch

from gyllm.envs import Message
from nanorl.rollout.episode import EpisodeRollout


class _ChatTokenizerWithTokens(Protocol):
    def apply_chat_template(self, messages: list[Message], **kwargs: Any) -> Any: ...


def mean_episode_reward(rollouts: Sequence[EpisodeRollout]) -> float:
    if not rollouts:
        return 0.0
    totals = [sum(rollout.rewards) for rollout in rollouts]
    return float(sum(totals) / len(totals))


def _tokenize_rollouts(rollouts: Sequence[EpisodeRollout], tokenizer: _ChatTokenizerWithTokens) -> torch.Tensor:
    if not rollouts:
        return torch.empty((0, 0), dtype=torch.long)

    sequences: list[torch.Tensor] = []
    for rollout in rollouts:
        tokens = tokenizer.apply_chat_template(
            rollout.messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokens["input_ids"]
        if input_ids.ndim == 2 and input_ids.shape[0] == 1:
            input_ids = input_ids[0]
        sequences.append(input_ids)

    max_len = max(seq.shape[-1] for seq in sequences)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    batch_device = sequences[0].device
    batch = torch.full(
        (len(sequences), max_len),
        pad_id,
        dtype=sequences[0].dtype,
        device=batch_device,
    )
    for idx, seq in enumerate(sequences):
        length = seq.shape[-1]
        batch[idx, :length] = seq.to(batch_device)

    return batch


def detokenize_tokens(tokenizer: Any, input_ids: list[int]) -> str:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        while input_ids and input_ids[-1] == pad_id:
            input_ids.pop()
    return tokenizer.decode(input_ids, skip_special_tokens=False)


def summarize_rollouts(
    rollouts: Sequence[EpisodeRollout],
    tokenizer: Any,
) -> tuple[torch.Tensor, float, str]:
    tokens = _tokenize_rollouts(rollouts, tokenizer)
    mean_reward = mean_episode_reward(rollouts)
    sample_text = ""
    if tokens.numel() > 0:
        sample_text = detokenize_tokens(tokenizer, tokens[0].tolist())
    return tokens, mean_reward, sample_text


def render_rollouts_html(rollouts: Sequence[EpisodeRollout]) -> str:
    total = len(rollouts)
    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<style>",
        "body { font-family: sans-serif; color: #111; }",
        ".controls { display: flex; gap: 8px; align-items: center; margin: 8px 0 16px; }",
        ".rollout { display: none; margin-bottom: 16px; }",
        ".rollout.active { display: block; }",
        ".meta { color: #555; font-size: 0.9em; }",
        ".message { border: 1px solid #e5e5e5; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }",
        ".raw { border-color: #f2c94c; background: #fff9e6; }",
        ".raw .role { color: #8a6d00; }",
        ".role { font-weight: 600; font-size: 0.85em; margin-bottom: 4px; }",
        "pre { margin: 0; white-space: pre-wrap; }",
        "</style>",
        "</head>",
        "<body>",
        "<h2>Rollout Summary</h2>",
        f'<p class="meta">Total rollouts: {total}.</p>',
    ]

    if total == 0:
        parts.append('<p class="meta">No rollouts available.</p>')
        parts.append("</body></html>")
        return "\n".join(parts)

    parts.extend(
        [
            '<div class="controls">',
            '<label for="rollout-select">Select rollout:</label>',
            '<select id="rollout-select"></select>',
            '<button id="prev-btn" type="button">Previous</button>',
            '<button id="next-btn" type="button">Next</button>',
            "</div>",
        ]
    )

    for idx, rollout in enumerate(rollouts, start=1):
        total_reward = sum(rollout.rewards)
        active = " active" if idx == 1 else ""
        parts.append(f'<div class="rollout{active}" data-index="{idx}">')
        parts.append(f"<h3>Rollout {idx}: actor={escape(str(rollout.actor))} total_reward={total_reward:.3f}</h3>")
        raw_actions = getattr(rollout, "raw_actions", None) or []
        assistant_idx = 0
        for msg_idx, msg in enumerate(rollout.messages):
            role = escape(msg.get("role", ""))
            content = escape(msg.get("content", ""))
            parts.append('<div class="message">')
            parts.append(f'<div class="role">{msg_idx}: {role}</div>')
            parts.append(f"<pre>{content}</pre>")
            parts.append("</div>")
            if msg.get("role") == "assistant":
                if assistant_idx < len(raw_actions):
                    raw = raw_actions[assistant_idx]
                    raw_text = "" if raw is None else str(raw)
                    if raw_text.strip() and raw_text.strip() != msg.get("content", ""):
                        parts.append('<div class="message raw">')
                        parts.append(f'<div class="role">raw {assistant_idx}</div>')
                        parts.append(f"<pre>{escape(raw_text)}</pre>")
                        parts.append("</div>")
                assistant_idx += 1
        parts.append(f'<div class="meta">Rewards: {escape(str(rollout.rewards))}</div>')
        parts.append("</div>")

    options = []
    for idx, rollout in enumerate(rollouts, start=1):
        total_reward = sum(rollout.rewards)
        label = f"Rollout {idx} (actor={rollout.actor}, reward={total_reward:.3f})"
        options.append(f'<option value="{idx}">{escape(label)}</option>')

    parts.extend(
        [
            "<script>",
            "const select = document.getElementById('rollout-select');",
            f"select.innerHTML = `{''.join(options)}`;",
            "select.value = '1';",
            "const rollouts = Array.from(document.querySelectorAll('.rollout'));",
            "const setActive = (idx) => {",
            "  rollouts.forEach((el) => {",
            "    el.classList.toggle('active', el.dataset.index === String(idx));",
            "  });",
            "  select.value = String(idx);",
            "};",
            "select.addEventListener('change', (e) => setActive(e.target.value));",
            "document.getElementById('prev-btn').addEventListener('click', () => {",
            "  const current = Number(select.value);",
            "  const next = Math.max(1, current - 1);",
            "  setActive(next);",
            "});",
            "document.getElementById('next-btn').addEventListener('click', () => {",
            "  const current = Number(select.value);",
            "  const next = Math.min(rollouts.length, current + 1);",
            "  setActive(next);",
            "});",
            "</script>",
        ]
    )

    parts.append("</body></html>")
    return "\n".join(parts)
