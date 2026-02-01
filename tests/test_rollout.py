from dataclasses import dataclass
from typing import cast

import pytest

from gyllm.batch import BatchedEnv
from gyllm.envs import (
    ActionError,
    LLMEnv,
    Message,
    Request,
    actor_env_id,
    make_actor_id,
)
from nanorl.rollout.episode import SamplingParams, rollout_episode, rollout_episode_batched

DUMMY_SAMPLING_PARAMS = cast(SamplingParams, object())


@dataclass
class _MockCompletion:
    text: str


@dataclass
class _MockResult:
    outputs: list[_MockCompletion]


class _DummyBackend:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    def generate(self, prompts, sampling_params):
        def next_output() -> _MockResult:
            if not self._outputs:
                raise AssertionError("No more mock outputs available.")
            text = self._outputs.pop(0)
            return _MockResult(outputs=[_MockCompletion(text=text)])

        if isinstance(prompts, list):
            return [next_output() for _ in prompts]
        return [next_output()]


class _DummyRunner:
    def __init__(self, outputs: list[str]) -> None:
        self.llm = _DummyBackend(outputs)

    def generate(self, prompts, sampling_params, **kwargs):
        del kwargs
        return self.llm.generate(prompts, sampling_params)


class _DummyTokenizer:
    def apply_chat_template(self, messages: list[Message], *, tokenize: bool, add_generation_prompt: bool) -> str:
        del tokenize, add_generation_prompt
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


class _ToyEnv(LLMEnv):
    agents: list[str] = ["agent"]

    def __init__(self, expected_actions: list[str], rewards: list[float] | None = None) -> None:
        super().__init__()
        self._expected_actions = list(expected_actions)
        if rewards is None:
            rewards = [0.0 for _ in expected_actions]
        if len(rewards) != len(expected_actions):
            raise ValueError("rewards must match expected_actions length")
        self._rewards = list(rewards)
        self._step = 0

    def _system_message(self, actor: str) -> Message:
        self.agent_id(actor)
        return {"role": "system", "content": "system"}

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        del options
        self._step = 0
        self._begin_episode()
        return [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": "start"},
                "needs_action": True,
                "info": {},
                "episode_id": self._episode_id,
                "episode_start": True,
                "episode_end": False,
            }
        ]

    def step(self, actions: dict[str, str]) -> list[Request]:
        actions = self._normalize_actions(actions)
        if "agent" not in actions:
            raise ActionError("Missing action for 'agent'")
        if self._step >= len(self._expected_actions):
            return []
        action = actions["agent"]
        expected = self._expected_actions[self._step]
        if action != expected:
            raise ActionError(f"Expected {expected!r}, got {action!r}")
        reward = self._rewards[self._step]
        self._step += 1
        done = self._step >= len(self._expected_actions)
        return [
            {
                "actor": make_actor_id("agent"),
                "reward": reward,
                "message": {"role": "user", "content": f"state-{self._step}"},
                "needs_action": not done,
                "info": {},
                "episode_id": self._episode_id,
                "episode_start": False,
                "episode_end": done,
            }
        ]


def test_rollout_episode_collects_history() -> None:
    env = _ToyEnv(["A", "B"], rewards=[1.0, -1.0])
    llm = _DummyRunner(["A", "B"])
    tokenizer = _DummyTokenizer()
    rollout = rollout_episode(env, llm, tokenizer, sampling_params=DUMMY_SAMPLING_PARAMS)

    assert rollout.actor == "agent"
    assert rollout.actions == ["A", "B"]
    assert rollout.rewards == [1.0, -1.0]
    assert rollout.messages[0]["role"] == "system"
    assert rollout.messages[1]["role"] == "user"
    assert rollout.messages[2]["content"] == "A"
    assert rollout.messages[-1]["content"] == "state-2"


def test_rollout_episode_retries_after_action_error() -> None:
    env = _ToyEnv(["A"], rewards=[0.5])
    llm = _DummyRunner(["BAD", "A"])
    tokenizer = _DummyTokenizer()

    rollout = rollout_episode(env, llm, tokenizer, sampling_params=DUMMY_SAMPLING_PARAMS)
    assert rollout.actions == ["A"]
    assert rollout.rewards == [0.5]
    assert rollout.messages[2]["content"] == "A"


def test_rollout_episode_batched_collects_per_actor() -> None:
    env = BatchedEnv([_ToyEnv(["A"], rewards=[0.1]), _ToyEnv(["B"], rewards=[0.2])])
    llm = _DummyRunner(["A", "B"])
    tokenizer = _DummyTokenizer()

    rollouts = rollout_episode_batched(env, llm, tokenizer, sampling_params=DUMMY_SAMPLING_PARAMS)

    assert len(rollouts) == 2
    for actor, rollout in rollouts.items():
        env_id = actor_env_id(actor)
        if env_id == 0:
            assert rollout.actions == ["A"]
            assert rollout.rewards == [0.1]
        elif env_id == 1:
            assert rollout.actions == ["B"]
            assert rollout.rewards == [0.2]
        else:
            pytest.fail(f"Unexpected env_id {env_id}")

        assert rollout.messages[0]["role"] == "system"
        assert rollout.messages[1]["content"] == "start"
        assert rollout.messages[2]["role"] == "assistant"
