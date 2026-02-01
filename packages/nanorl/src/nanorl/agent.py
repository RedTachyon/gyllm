"""Agent abstractions for nanorl."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from vllm import SamplingParams

from gyllm.envs import ActorId, LLMEnv, Message, Request
from nanorl.rollout.episode import ChatTokenizer, EpisodeRollout, LLMRunner

# Reasoning retention modes for how assistant messages treat chain-of-thought.
ReasoningMode = Literal["keep", "drop", "truncate"]
# (actor_id, episode_id) -> stable key across batched envs using Request metadata.
EpisodeKey = tuple[ActorId, int | None]
# Extract the last <action>...</action> span from a model completion.
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


@dataclass(slots=True)
class PendingAction:
    """Parsed model output waiting to be committed to an episode."""

    action: str
    assistant_message: str
    raw_text: str


@dataclass(slots=True)
class EpisodeState:
    """Per-episode transcript and bookkeeping for a single actor."""

    actor: ActorId
    episode_id: int | None
    group_id: str | int | None
    messages: list[Message]
    rewards: list[float]
    actions: list[str]
    raw_actions: list[str]
    pending: PendingAction | None = None

    def clone(self) -> EpisodeState:
        """Return a shallow copy suitable for functional-style updates."""
        return EpisodeState(
            actor=self.actor,
            episode_id=self.episode_id,
            group_id=self.group_id,
            messages=list(self.messages),
            rewards=list(self.rewards),
            actions=list(self.actions),
            raw_actions=list(self.raw_actions),
            pending=self.pending,
        )

    def to_rollout(self) -> EpisodeRollout:
        """Convert the episode state to a rollout record."""
        return EpisodeRollout(
            actor=self.actor,
            messages=self.messages,
            rewards=self.rewards,
            actions=self.actions,
            group_id=self.group_id,
            raw_actions=self.raw_actions,
        )


@dataclass(slots=True)
class AgentState:
    """Mutable container for active episodes and completed rollouts."""

    episodes: dict[EpisodeKey, EpisodeState] = field(default_factory=dict)
    completed: list[EpisodeRollout] = field(default_factory=list)

    def clone(self) -> AgentState:
        """Return a shallow copy with per-episode buffers duplicated."""
        return AgentState(
            episodes={key: episode.clone() for key, episode in self.episodes.items()},
            completed=list(self.completed),
        )

    def pop_completed(self) -> list[EpisodeRollout]:
        """Return and clear completed rollouts."""
        completed = list(self.completed)
        self.completed.clear()
        return completed


class Agent(ABC):
    """Base agent interface with explicit state."""

    @abstractmethod
    def act(
        self,
        requests: Sequence[Request],
        state: AgentState | None = None,
    ) -> tuple[dict[ActorId, str], AgentState]:
        """Return actions for a batch of requests and the updated state."""


class FixedAgent(Agent):
    """Agent that always returns a fixed action."""

    def __init__(self, action: str) -> None:
        """Initialize the agent with a fixed action string."""
        self.action = action

    def act(
        self,
        requests: Sequence[Request],
        state: AgentState | None = None,
    ) -> tuple[dict[ActorId, str], AgentState]:
        """Return the fixed action for all pending requests."""
        next_state = state.clone() if state is not None else AgentState()
        actions = {request["actor"]: self.action for request in requests if request.get("needs_action")}
        return actions, next_state


class LLMAgent(Agent):
    """Agent backed by an LLM with shared rollout utilities."""

    def __init__(
        self,
        model: Any,
        llm: LLMRunner,
        tokenizer: ChatTokenizer,
        sampling_params: SamplingParams,
        *,
        action_parser: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the LLM-backed agent with shared components."""
        self.model = model
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self._action_parser = action_parser or str.strip

    def parse_action(self, text: str) -> str:
        """Parse the model output into an environment action."""
        return self._action_parser(text)

    def assistant_message(self, text: str, action: str) -> str:
        """Return the assistant message committed to the transcript."""
        del text
        return action

    def _new_episode_state(self, request: Request) -> EpisodeState:
        """Create a new episode state seeded with the initial messages."""
        system_message = request.get("system_message")
        if system_message is None:
            raise ValueError("Request missing system_message for episode start.")
        return EpisodeState(
            actor=request["actor"],
            episode_id=request.get("episode_id"),
            group_id=request.get("group_id"),
            messages=[system_message, request["message"]],
            rewards=[],
            actions=[],
            raw_actions=[],
            pending=None,
        )

    def _apply_request(self, request: Request, state: AgentState) -> None:
        """Update episode buffers with an environment request."""
        key = (request["actor"], request.get("episode_id"))
        episode_start = bool(request.get("episode_start", False))
        needs_action = bool(request.get("needs_action", False))
        episode_end = bool(request.get("episode_end", not needs_action))

        episode = state.episodes.get(key)
        if episode is None or episode_start:
            state.episodes[key] = self._new_episode_state(request)
            if episode_end:
                state.completed.append(state.episodes.pop(key).to_rollout())
            return

        if episode.group_id is None and request.get("group_id") is not None:
            episode.group_id = request.get("group_id")

        if int(request.get("repeat_count", 0)) > 0:
            if episode.pending is not None:
                episode.pending = None
            # Repeated requests should replace the latest user message in-place.
            if episode.messages:
                last = episode.messages[-1]
                if last["role"] == "user" and request["message"]["role"] == "user":
                    episode.messages[-1] = request["message"]
            return

        if episode.pending is not None:
            episode.messages.append({"role": "assistant", "content": episode.pending.assistant_message})
            episode.actions.append(episode.pending.action)
            episode.raw_actions.append(episode.pending.raw_text)
            episode.pending = None

        episode.messages.append(request["message"])
        episode.rewards.append(float(request["reward"]))

        if episode_end:
            state.completed.append(state.episodes.pop(key).to_rollout())

    def act(
        self,
        requests: Sequence[Request],
        state: AgentState | None = None,
    ) -> tuple[dict[ActorId, str], AgentState]:
        """Return actions and updated state for the request batch."""
        next_state = state.clone() if state is not None else AgentState()

        for request in requests:
            self._apply_request(request, next_state)

        prompts: list[str] = []
        actors: list[ActorId] = []
        keys_by_actor: dict[ActorId, EpisodeKey] = {}
        for request in requests:
            needs_action = bool(request.get("needs_action", False))
            episode_end = bool(request.get("episode_end", not needs_action))
            if not needs_action or episode_end:
                continue
            key = (request["actor"], request.get("episode_id"))
            episode = next_state.episodes.get(key)
            if episode is None:
                episode = self._new_episode_state(request)
                next_state.episodes[key] = episode
            prompts.append(
                self.tokenizer.apply_chat_template(
                    episode.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            actors.append(request["actor"])
            keys_by_actor[request["actor"]] = key

        if not prompts:
            return {}, next_state

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        pending_by_actor: dict[ActorId, PendingAction] = {}
        for actor, output in zip(actors, outputs, strict=False):
            text = output.outputs[0].text
            action = self.parse_action(text)
            pending_by_actor[actor] = PendingAction(
                action=action,
                assistant_message=self.assistant_message(text, action),
                raw_text=text,
            )

        for actor, pending in pending_by_actor.items():
            key = keys_by_actor[actor]
            episode = next_state.episodes.get(key)
            if episode is not None:
                episode.pending = pending

        actions = {actor: pending.action for actor, pending in pending_by_actor.items()}
        return actions, next_state

    def rollout_autoreset_batched(
        self,
        env: LLMEnv,
        *,
        max_episodes: int | None = None,
        max_steps: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> list[EpisodeRollout]:
        """Collect rollouts using the agent state machine."""
        if max_episodes is None and max_steps is None:
            raise ValueError("rollout_autoreset_batched requires max_episodes or max_steps.")
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError(f"max_episodes must be > 0; got {max_episodes}")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be > 0; got {max_steps}")

        requests = env.reset()
        if not requests:
            return []

        state = AgentState()
        completed: list[EpisodeRollout] = []
        step_count = 0

        # Temporarily override sampling params for this rollout pass.
        original_sampling_params = self.sampling_params
        if sampling_params is not None:
            self.sampling_params = sampling_params

        try:
            while True:
                actions, state = self.act(requests, state)
                completed.extend(state.pop_completed())

                pending = [request for request in requests if request.get("needs_action")]
                if max_episodes is not None and len(completed) >= max_episodes:
                    break
                if max_steps is not None and step_count >= max_steps:
                    break
                if not pending:
                    break

                requests = env.step(actions)
                step_count += len(actions)
        finally:
            self.sampling_params = original_sampling_params

        if max_episodes is not None:
            return completed[:max_episodes]
        return completed


class InstructAgent(LLMAgent):
    """Agent that forwards model output as-is (optionally stripped)."""

    def __init__(
        self,
        model: Any,
        llm: LLMRunner,
        tokenizer: ChatTokenizer,
        sampling_params: SamplingParams,
        *,
        strip: bool = True,
        action_parser: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize an instruct-style agent."""
        if action_parser is None:
            action_parser = str.strip if strip else lambda text: text
        super().__init__(
            model,
            llm,
            tokenizer,
            sampling_params,
            action_parser=action_parser,
        )


class ReasoningAgent(LLMAgent):
    """Agent that extracts actions from <action>...</action> tags."""

    def __init__(
        self,
        model: Any,
        llm: LLMRunner,
        tokenizer: ChatTokenizer,
        sampling_params: SamplingParams,
        *,
        reasoning_mode: ReasoningMode = "keep",
        truncated_token: str = "TRUNCATED",
        system_prompt_extra: str | None = None,
        action_parser: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize a reasoning agent with action-tag parsing."""
        super().__init__(
            model,
            llm,
            tokenizer,
            sampling_params,
            action_parser=action_parser,
        )
        if reasoning_mode not in ("keep", "drop", "truncate"):
            raise ValueError(f"Unknown reasoning_mode: {reasoning_mode!r}")
        self.reasoning_mode = reasoning_mode
        self.truncated_token = truncated_token
        self.system_prompt_extra = system_prompt_extra

    def _new_episode_state(self, request: Request) -> EpisodeState:
        """Create a new episode state with an optional prompt addendum."""
        episode = super()._new_episode_state(request)
        extra = self.system_prompt_extra
        if extra:
            extra_text = str(extra).strip()
            if extra_text:
                system_message = cast(Message, dict(episode.messages[0]))
                content = str(system_message.get("content", "")).rstrip()
                if content:
                    system_message["content"] = f"{content}\n\n{extra_text}"
                else:
                    system_message["content"] = extra_text
                episode.messages[0] = system_message
        return episode

    def _find_action(self, text: str) -> tuple[str, str, str, str] | None:
        """Return the prefix, action text, action block, and suffix for the last tag."""
        matches = list(_ACTION_RE.finditer(text))
        if not matches:
            return None
        match = matches[-1]
        prefix = text[: match.start()]
        action_block = match.group(0)
        action_text = match.group(1)
        suffix = text[match.end() :]
        return prefix, action_text, action_block, suffix

    def parse_action(self, text: str) -> str:
        """Extract the action text from the last <action> tag."""
        match = self._find_action(text)
        if match is not None:
            _prefix, action_text, _action_block, _suffix = match
            action = action_text.strip()
            if action:
                return action
        return super().parse_action(text)

    def assistant_message(self, text: str, action: str) -> str:
        """Return the assistant message based on the reasoning retention mode."""
        match = self._find_action(text)
        if match is None:
            return text.strip()
        prefix, _action_text, action_block, suffix = match
        if self.reasoning_mode == "keep":
            return text.strip()
        payload = f"{action_block}{suffix}".strip()
        if self.reasoning_mode == "drop":
            return payload
        if prefix.strip():
            payload = f"{self.truncated_token}\n{payload}"
        return payload.strip()


__all__ = [
    "Agent",
    "AgentState",
    "EpisodeState",
    "FixedAgent",
    "InstructAgent",
    "LLMAgent",
    "PendingAction",
    "ReasoningAgent",
    "ReasoningMode",
]
