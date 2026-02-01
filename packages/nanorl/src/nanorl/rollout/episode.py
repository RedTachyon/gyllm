from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from vllm import SamplingParams

from gyllm.envs import ActionError, ActorId, LLMEnv, Message, Request


class ChatTokenizer(Protocol):
    """Protocol for chat template tokenizers used by rollouts."""

    def apply_chat_template(
        self,
        messages: list[Message],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render a chat prompt from messages.

        Args:
            messages: Ordered chat messages to render.
            tokenize: Whether to return tokens instead of a string.
            add_generation_prompt: Whether to append a generation marker.

        Returns:
            Rendered chat prompt.
        """
        ...


class LLMRunner(Protocol):
    """Protocol for LLM runners that can generate completions."""

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Any:
        """Generate model outputs for one or more prompts.

        Args:
            prompts: Prompt string or batch of prompts.
            sampling_params: vLLM sampling parameters.
            **kwargs: Additional backend-specific arguments.

        Returns:
            Backend-generated outputs.
        """
        ...


ActionExtractor = Callable[[str], str]


@dataclass(frozen=True)
class EpisodeRollout:
    """Rollout data for a single episode.

    Attributes:
        actor: Actor identifier for the episode.
        messages: Full message history for the episode.
        rewards: Per-step rewards aligned with actions.
        actions: Actions produced during the episode.
        group_id: Optional grouping key for batched algorithms.
        raw_actions: Raw model outputs aligned with actions (if captured).
    """

    actor: ActorId
    messages: list[Message]
    rewards: list[float]
    actions: list[str]
    group_id: str | int | None = None
    raw_actions: list[str] = field(default_factory=list)


def _init_episode_buffers(
    request: Request,
) -> tuple[list[Message], list[float], list[str], list[str], int, str | int | None]:
    """Return fresh buffers and identifiers for a new actor episode."""
    system_message = request.get("system_message")
    if system_message is None:
        raise ValueError("Request missing system_message for episode start.")
    messages = [system_message, request["message"]]
    return messages, [], [], [], request["episode_id"], request.get("group_id")


def rollout_episode(
    env: LLMEnv,
    llm: LLMRunner,
    tokenizer: ChatTokenizer,
    sampling_params: SamplingParams,
    *,
    action_extractor: ActionExtractor | None = None,
) -> EpisodeRollout:
    """Run a full episode rollout and return messages, rewards, and actions.

    Args:
        env: Single-instance environment to roll out.
        llm: LLM runner to generate actions.
        tokenizer: Chat tokenizer for prompt construction.
        sampling_params: vLLM sampling parameters.
        action_extractor: Optional function to extract actions from text.

    Returns:
        Episode rollout containing messages, rewards, and actions.

    Raises:
        ValueError: If the env does not return exactly one request.
    """

    (obs,) = env.reset()
    actor = obs["actor"]
    system_message = obs.get("system_message")
    if system_message is None:
        raise ValueError("Initial request missing system_message.")
    messages = [system_message, obs["message"]]
    rewards: list[float] = []
    actions: list[str] = []
    raw_actions: list[str] = []
    group_id = obs.get("group_id")

    action_extractor = str.strip if action_extractor is None else action_extractor

    while obs["needs_action"]:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
        action_text = outputs[0].outputs[0].text
        action = action_extractor(action_text)
        try:
            (obs,) = env.step({actor: action})
        except ActionError:
            continue

        if int(obs.get("repeat_count", 0)) > 0:
            continue
        if group_id is None:
            group_id = obs.get("group_id")

        messages.append({"role": "assistant", "content": action})
        messages.append(obs["message"])
        actions.append(action)
        raw_actions.append(action_text)
        rewards.append(obs["reward"])

    return EpisodeRollout(
        actor=actor,
        messages=messages,
        rewards=rewards,
        actions=actions,
        group_id=group_id,
        raw_actions=raw_actions,
    )


def rollout_episode_batched(
    env: LLMEnv,
    llm: LLMRunner,
    tokenizer: ChatTokenizer,
    sampling_params: SamplingParams,
    *,
    action_extractor: ActionExtractor | None = None,
) -> dict[ActorId, EpisodeRollout]:
    """Run a batched rollout and return per-actor messages, rewards, and actions.

    Args:
        env: Batched environment to roll out.
        llm: LLM runner to generate actions.
        tokenizer: Chat tokenizer for prompt construction.
        sampling_params: vLLM sampling parameters.
        action_extractor: Optional function to extract actions from text.

    Returns:
        Mapping of actor IDs to episode rollouts.
    """

    requests = env.reset()
    if not requests:
        return {}

    action_extractor = str.strip if action_extractor is None else action_extractor

    messages_by_actor: dict[ActorId, list[Message]] = {}
    rewards_by_actor: dict[ActorId, list[float]] = {}
    actions_by_actor: dict[ActorId, list[str]] = {}
    raw_actions_by_actor: dict[ActorId, list[str]] = {}
    group_ids_by_actor: dict[ActorId, str | int | None] = {}

    for request in requests:
        actor = request["actor"]
        system_message = request.get("system_message")
        if system_message is None:
            raise ValueError("Initial request missing system_message.")
        messages_by_actor[actor] = [system_message, request["message"]]
        rewards_by_actor[actor] = []
        actions_by_actor[actor] = []
        raw_actions_by_actor[actor] = []
        group_ids_by_actor[actor] = request.get("group_id")

    pending = [request for request in requests if request["needs_action"]]
    while pending:
        prompts: list[str] = []
        actors: list[ActorId] = []
        for request in pending:
            actor = request["actor"]
            prompt = tokenizer.apply_chat_template(
                messages_by_actor[actor],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            actors.append(actor)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        actions: dict[ActorId, str] = {}
        raw_by_actor: dict[ActorId, str] = {}
        for actor, output in zip(actors, outputs, strict=False):
            action_text = output.outputs[0].text
            action = action_extractor(action_text)
            actions[actor] = action
            raw_by_actor[actor] = action_text

        try:
            requests = env.step(actions)
        except ActionError:
            continue

        for request in requests:
            actor = request["actor"]
            if int(request.get("repeat_count", 0)) > 0:
                continue
            if actor not in messages_by_actor:
                system_message = request.get("system_message")
                if system_message is None:
                    raise ValueError("Request missing system_message for new episode.")
                messages_by_actor[actor] = [system_message, request["message"]]
                rewards_by_actor[actor] = []
                actions_by_actor[actor] = []
                raw_actions_by_actor[actor] = []
                group_ids_by_actor[actor] = request.get("group_id")
                continue
            if group_ids_by_actor.get(actor) is None and request.get("group_id") is not None:
                group_ids_by_actor[actor] = request.get("group_id")

            action = actions.get(actor)
            if action is not None:
                messages_by_actor[actor].append({"role": "assistant", "content": action})
                actions_by_actor[actor].append(action)
                raw_actions_by_actor[actor].append(raw_by_actor.get(actor, action))
            messages_by_actor[actor].append(request["message"])
            rewards_by_actor[actor].append(request["reward"])

        pending = [request for request in requests if request["needs_action"]]

    return {
        actor: EpisodeRollout(
            actor=actor,
            messages=messages_by_actor[actor],
            rewards=rewards_by_actor[actor],
            actions=actions_by_actor[actor],
            group_id=group_ids_by_actor.get(actor),
            raw_actions=raw_actions_by_actor.get(actor, []),
        )
        for actor in messages_by_actor
    }


def rollout_autoreset_batched(
    env: LLMEnv,
    llm: LLMRunner,
    tokenizer: ChatTokenizer,
    sampling_params: SamplingParams,
    *,
    max_episodes: int | None = None,
    max_steps: int | None = None,
    action_extractor: ActionExtractor | None = None,
) -> list[EpisodeRollout]:
    """Run a batched rollout with autoresets until targets are reached.

    Wrap batched envs with AutoResetWrapper to keep a fixed batch size.
    max_steps counts action selections across all actors and may overshoot
    when stepping a full batch.

    Args:
        env: Batched environment with autoreset behavior.
        llm: LLM runner to generate actions.
        tokenizer: Chat tokenizer for prompt construction.
        sampling_params: vLLM sampling parameters.
        max_episodes: Optional cap on number of completed episodes.
        max_steps: Optional cap on number of action selections.
        action_extractor: Optional function to extract actions from text.

    Returns:
        Completed episode rollouts in the order they finish.

    Raises:
        ValueError: If both max_episodes and max_steps are omitted or invalid.
    """

    if max_episodes is None and max_steps is None:
        raise ValueError("rollout_autoreset_batched requires max_episodes or max_steps.")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be > 0; got {max_episodes}")
    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be > 0; got {max_steps}")

    requests = env.reset()
    if not requests:
        return []

    action_extractor = str.strip if action_extractor is None else action_extractor

    messages_by_actor: dict[ActorId, list[Message]] = {}
    rewards_by_actor: dict[ActorId, list[float]] = {}
    actions_by_actor: dict[ActorId, list[str]] = {}
    raw_actions_by_actor: dict[ActorId, list[str]] = {}
    episode_ids_by_actor: dict[ActorId, int] = {}
    group_ids_by_actor: dict[ActorId, str | int | None] = {}
    completed: list[EpisodeRollout] = []

    for request in requests:
        actor = request["actor"]
        messages, rewards, action_log, raw_action_log, episode_id, group_id = _init_episode_buffers(request)
        messages_by_actor[actor] = messages
        rewards_by_actor[actor] = rewards
        actions_by_actor[actor] = action_log
        raw_actions_by_actor[actor] = raw_action_log
        episode_ids_by_actor[actor] = episode_id
        group_ids_by_actor[actor] = group_id

    pending = [request for request in requests if request["needs_action"]]
    step_count = 0

    while pending:
        if max_steps is not None and step_count >= max_steps:
            break
        if max_episodes is not None and len(completed) >= max_episodes:
            break

        prompts: list[str] = []
        actors: list[ActorId] = []
        for request in pending:
            actor = request["actor"]
            if actor not in messages_by_actor:
                messages, rewards, action_log, raw_action_log, episode_id, group_id = _init_episode_buffers(request)
                messages_by_actor[actor] = messages
                rewards_by_actor[actor] = rewards
                actions_by_actor[actor] = action_log
                raw_actions_by_actor[actor] = raw_action_log
                episode_ids_by_actor[actor] = episode_id
                group_ids_by_actor[actor] = group_id
            prompt = tokenizer.apply_chat_template(
                messages_by_actor[actor],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            actors.append(actor)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        actions: dict[ActorId, str] = {}
        raw_by_actor: dict[ActorId, str] = {}
        for actor, output in zip(actors, outputs, strict=False):
            action_text = output.outputs[0].text
            action = action_extractor(action_text)
            actions[actor] = action
            raw_by_actor[actor] = action_text

        try:
            requests = env.step(actions)
        except ActionError:
            continue

        step_count += len(actions)
        for request in requests:
            actor = request["actor"]
            episode_id = request["episode_id"]
            if (
                request.get("episode_start")
                or actor not in messages_by_actor
                or episode_ids_by_actor.get(actor) != episode_id
            ):
                messages, rewards, action_log, raw_action_log, episode_id, group_id = _init_episode_buffers(request)
                messages_by_actor[actor] = messages
                rewards_by_actor[actor] = rewards
                actions_by_actor[actor] = action_log
                raw_actions_by_actor[actor] = raw_action_log
                episode_ids_by_actor[actor] = episode_id
                group_ids_by_actor[actor] = group_id
                if request.get("episode_end"):
                    completed.append(
                        EpisodeRollout(
                            actor=actor,
                            messages=messages_by_actor[actor],
                            rewards=rewards_by_actor[actor],
                            actions=actions_by_actor[actor],
                            group_id=group_ids_by_actor.get(actor),
                            raw_actions=raw_actions_by_actor.get(actor, []),
                        )
                    )
                continue
            if group_ids_by_actor.get(actor) is None and request.get("group_id") is not None:
                group_ids_by_actor[actor] = request.get("group_id")

            if int(request.get("repeat_count", 0)) > 0:
                continue

            action = actions.get(actor)
            if action is not None:
                messages_by_actor[actor].append({"role": "assistant", "content": action})
                actions_by_actor[actor].append(action)
                raw_actions_by_actor[actor].append(raw_by_actor.get(actor, action))

            messages_by_actor[actor].append(request["message"])
            rewards_by_actor[actor].append(request["reward"])

            if request.get("episode_end"):
                completed.append(
                    EpisodeRollout(
                        actor=actor,
                        messages=messages_by_actor[actor],
                        rewards=rewards_by_actor[actor],
                        actions=actions_by_actor[actor],
                        group_id=group_ids_by_actor.get(actor),
                        raw_actions=raw_actions_by_actor.get(actor, []),
                    )
                )

        if max_episodes is not None and len(completed) >= max_episodes:
            break

        pending = [request for request in requests if request["needs_action"]]

    if max_episodes is not None:
        return completed[:max_episodes]
    return completed
