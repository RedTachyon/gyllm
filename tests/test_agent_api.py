from dataclasses import dataclass
from typing import cast

from gyllm.envs import ActorId, LLMEnv, Message, Request, make_actor_id
from nanorl.agent import FixedAgent, InstructAgent, ReasoningAgent
from nanorl.rollout.episode import SamplingParams

DUMMY_SAMPLING_PARAMS = cast(SamplingParams, object())


@dataclass
class _DummyCompletion:
    """Minimal completion wrapper with a text payload."""

    text: str


@dataclass
class _DummyResult:
    """Minimal vLLM-style result wrapper."""

    outputs: list[_DummyCompletion]


class _DummyLLM:
    """LLM stub that yields queued outputs in order."""

    def __init__(self, outputs: list[str]) -> None:
        """Initialize the stub with a queue of outputs."""
        self._outputs = list(outputs)
        self.prompts: list[list[str]] = []

    def generate(self, prompts, sampling_params, **kwargs):
        """Return dummy results for each prompt."""
        del sampling_params, kwargs
        prompt_list = list(prompts) if isinstance(prompts, list) else [prompts]
        self.prompts.append(prompt_list)
        results: list[_DummyResult] = []
        for _ in prompt_list:
            if not self._outputs:
                raise AssertionError("No more dummy outputs available.")
            text = self._outputs.pop(0)
            results.append(_DummyResult(outputs=[_DummyCompletion(text=text)]))
        return results


class _DummyTokenizer:
    """Tokenizer stub that renders a simple role/content prompt."""

    def apply_chat_template(self, messages: list[Message], *, tokenize: bool, add_generation_prompt: bool) -> str:
        """Render a simple prompt string for the provided messages."""
        if tokenize:
            raise AssertionError("Dummy tokenizer only supports tokenize=False.")
        suffix = "<gen>" if add_generation_prompt else ""
        return " | ".join(f"{msg['role']}:{msg['content']}" for msg in messages) + suffix


def _system_message(actor: str) -> Message:
    """Return a simple system prompt."""
    return {"role": "system", "content": f"system for {actor}"}


def _make_request(
    *,
    actor: str,
    content: str,
    reward: float,
    needs_action: bool,
    episode_id: int,
    episode_start: bool = False,
    episode_end: bool = False,
    repeat_count: int | None = None,
    system_message: Message | None = None,
) -> Request:
    """Create a minimal Request dict for tests."""
    request: Request = {
        "actor": actor,
        "reward": reward,
        "message": {"role": "user", "content": content},
        "needs_action": needs_action,
        "info": {},
        "episode_id": episode_id,
        "episode_start": episode_start,
        "episode_end": episode_end,
    }
    if episode_start:
        request["system_message"] = system_message or _system_message(actor)
    if repeat_count is not None:
        request["repeat_count"] = repeat_count
    return request


def test_instruct_agent_commits_pending_on_next_request() -> None:
    """Pending actions should commit on the next non-repeat observation."""
    llm = _DummyLLM(["A"])
    tokenizer = _DummyTokenizer()
    agent = InstructAgent(
        model=object(),
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=DUMMY_SAMPLING_PARAMS,
    )

    req0 = _make_request(
        actor="player",
        content="start",
        reward=0.0,
        needs_action=True,
        episode_id=0,
        episode_start=True,
    )
    actions, state = agent.act([req0], None)

    assert actions == {"player": "A"}
    episode = state.episodes[("player", 0)]
    assert [msg["role"] for msg in episode.messages] == ["system", "user"]
    assert episode.pending is not None and episode.pending.action == "A"

    req1 = _make_request(
        actor="player",
        content="done",
        reward=1.0,
        needs_action=False,
        episode_id=0,
        episode_end=True,
    )
    actions, state = agent.act([req1], state)

    assert actions == {}
    assert ("player", 0) not in state.episodes
    assert len(state.completed) == 1
    rollout = state.completed[0]
    assert rollout.actions == ["A"]
    assert rollout.rewards == [1.0]
    assert [msg["role"] for msg in rollout.messages] == ["system", "user", "assistant", "user"]


def test_instruct_agent_discards_pending_on_repeat() -> None:
    """Repeat requests should drop pending actions and re-prompt."""
    llm = _DummyLLM(["A", "B"])
    tokenizer = _DummyTokenizer()
    agent = InstructAgent(
        model=object(),
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=DUMMY_SAMPLING_PARAMS,
    )

    req0 = _make_request(
        actor="player",
        content="start",
        reward=0.0,
        needs_action=True,
        episode_id=0,
        episode_start=True,
    )
    _, state = agent.act([req0], None)

    repeat = _make_request(
        actor="player",
        content="start",
        reward=0.0,
        needs_action=True,
        episode_id=0,
        repeat_count=1,
    )
    actions, state = agent.act([repeat], state)

    assert actions == {"player": "B"}
    episode = state.episodes[("player", 0)]
    assert episode.pending is not None and episode.pending.action == "B"
    assert episode.actions == []
    assert len(episode.messages) == 2


def test_reasoning_agent_parses_action_tags() -> None:
    """ReasoningAgent should extract action tags and redact reasoning."""
    llm = _DummyLLM(["Thoughts\\n<action>MOVE</action>\\nMore"])
    tokenizer = _DummyTokenizer()
    agent = ReasoningAgent(
        model=object(),
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=DUMMY_SAMPLING_PARAMS,
        reasoning_mode="drop",
    )

    req0 = _make_request(
        actor="player",
        content="start",
        reward=0.0,
        needs_action=True,
        episode_id=0,
        episode_start=True,
    )
    actions, state = agent.act([req0], None)

    assert actions == {"player": "MOVE"}
    episode = state.episodes[("player", 0)]
    pending = episode.pending
    assert pending is not None
    assert pending.action == "MOVE"
    assert pending.assistant_message == "<action>MOVE</action>\\nMore"


def test_fixed_agent_returns_constant_action() -> None:
    """FixedAgent should always return the configured action."""
    agent = FixedAgent(action="X")
    req0 = _make_request(
        actor="player",
        content="start",
        reward=0.0,
        needs_action=True,
        episode_id=0,
        episode_start=True,
    )
    actions, state = agent.act([req0], None)

    assert actions == {"player": "X"}
    assert state.episodes == {}


def test_rollout_autoreset_batched_collects_rollout() -> None:
    """Rollout helper should produce a complete EpisodeRollout."""

    class _DummyEnv(LLMEnv):
        """Simple two-step env that emits a terminal request."""

        agents: list[str] = ["player"]

        def __init__(self) -> None:
            """Initialize a two-step episode counter."""
            super().__init__()
            self._step = 0

        def _system_message(self, actor: ActorId) -> Message:
            """Return a system prompt for the actor."""
            self.agent_id(actor)
            return {"role": "system", "content": "system"}

        def reset(self, options: dict[str, object] | None = None) -> list[Request]:
            """Return the initial request for a new episode."""
            del options
            self._step = 0
            episode_id = self._begin_episode()
            return [
                _make_request(
                    actor=make_actor_id("player"),
                    content="start",
                    reward=0.0,
                    needs_action=True,
                    episode_id=episode_id,
                    episode_start=True,
                    system_message=self._system_message(make_actor_id("player")),
                )
            ]

        def step(self, actions: dict[ActorId, str]) -> list[Request]:
            """Advance one step and return the next request."""
            del actions
            self._step += 1
            if self._step == 1:
                return [
                    _make_request(
                        actor=make_actor_id("player"),
                        content="next",
                        reward=1.0,
                        needs_action=True,
                        episode_id=self._episode_id,
                    )
                ]
            return [
                _make_request(
                    actor=make_actor_id("player"),
                    content="done",
                    reward=2.0,
                    needs_action=False,
                    episode_id=self._episode_id,
                    episode_end=True,
                )
            ]

    llm = _DummyLLM(["A", "B"])
    tokenizer = _DummyTokenizer()
    agent = InstructAgent(
        model=object(),
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=DUMMY_SAMPLING_PARAMS,
    )
    env = _DummyEnv()

    rollouts = agent.rollout_autoreset_batched(env, max_episodes=1)

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.actions == ["A", "B"]
    assert rollout.rewards == [1.0, 2.0]
    assert [msg["role"] for msg in rollout.messages] == ["system", "user", "assistant", "user", "assistant", "user"]
