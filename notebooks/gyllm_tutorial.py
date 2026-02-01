# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GYLLM Tutorial
#
# Overview of the `gyllm` environment API:
# - Core types: `LLMEnv`, `Request`, `ActorId`
# - Episode loops with per-actor chat histories
# - Multi-agent and batched envs
# - Subprocess and Docker hosting
#
# Requirements: `gyllm` importable in the kernel (e.g. `uv sync` or
# `uv pip install -e packages/gyllm`).
#
# Conventions:
# - histories are `system` -> `user` -> `assistant` -> ...
# - env responses use role `"user"`
# - actions use role `"assistant"`

# %%

import ast
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gyllm  # noqa: F401

print("gyllm import OK")

# %% [markdown]
# ## 0) Imports
#
# `make()` uses the registry; `batch_envs()` combines env instances.

# %%
from gyllm import list_envs, make
from gyllm.batch import batch_envs
from gyllm.core import ActorId, LLMEnv, Message, Request, actor_agent_id, make_actor_id
from gyllm.envs.simple.iterated_games import TftIpdEnv

# %% [markdown]
# ## 1) Actors and requests
#
# Actors are identified by strings like `"agent"` or `"agent::env=3"`.
# Each step returns `Request` objects:
# - `actor`, `reward`, `message`, `needs_action`
# - episode metadata: `episode_id`, `episode_start`, `episode_end`
#
# The env returns only the next message; histories are maintained by the caller.

# %%
def show_requests(requests: list[Request], *, title: str | None = None) -> None:
    if title:
        print(title)
        print("-" * len(title))
    for r in requests:
        actor = r["actor"]
        msg = r["message"]
        preview = msg["content"].replace("\n", "\\n")
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(
            f"actor={actor} episode={r['episode_id']} start={r['episode_start']} "
            f"end={r['episode_end']} needs_action={r['needs_action']} reward={r['reward']:.3f} "
            f"| message.role={msg['role']!r} message={preview!r}"
        )


# Preferred: instantiate via the registry.
env = make("openenv/echo")
requests = env.reset()
show_requests(requests, title="make('openenv/echo').reset()")

# %% [markdown]
# ## 2) Single-agent loop (manual history)
#
# Maintain history locally:
# - start with `request["system_message"]` on episode start (`system`)
# - append env messages (`user`)
# - append actions (`assistant`)
#
# Action formatting is env-specific.

# %%
env = make("openenv/echo")
actor: ActorId = make_actor_id("agent")

requests = env.reset()
history: list[Message] = [requests[0]["system_message"], requests[0]["message"]]

print("System message role:", history[0]["role"])
print("First env message role:", history[1]["role"])
print("First env message:", history[1]["content"])

# %%
def echo_policy(messages: list[Message], *, turn: int) -> str:
    # Minimal "agent": say something different each time.
    return f"hello from turn={turn}"


total_reward = 0.0
max_turns = 3

for turn in range(1, max_turns + 1):
    pending = [r for r in requests if r["needs_action"]]
    if not pending:
        break

    completion = echo_policy(history, turn=turn)
    history.append({"role": "assistant", "content": completion})

    requests = env.step({actor: completion})
    if not requests:
        break

    total_reward += requests[0]["reward"]
    history.append(requests[0]["message"])

    print(f"turn={turn} reward={requests[0]['reward']:.3f} env says: {requests[0]['message']['content']!r}")

    if not requests[0]["needs_action"]:
        break

print("total_reward:", total_reward)

# %% [markdown]
# ## 3) Reusable episode runner
#
# Works for single-agent, multi-agent, batched, and remote envs. The only change is the number of actors.

# %%
Policy = Callable[[ActorId, list[Message]], str]


@dataclass(slots=True)
class EpisodeResult:
    histories: dict[ActorId, list[Message]]
    total_reward: dict[ActorId, float]
    steps: int


def run_episode(
    env: LLMEnv,
    *,
    policy: Policy,
    max_steps: int = 50,
    verbose: bool = False,
) -> EpisodeResult:
    histories: dict[ActorId, list[Message]] = {}
    totals: dict[ActorId, float] = {}

    requests = env.reset()
    for req in requests:
        histories[req["actor"]] = [req["system_message"], req["message"]]
        totals[req["actor"]] = float(req["reward"])

    if verbose:
        show_requests(requests, title="reset() -> requests")

    steps = 0
    while steps < max_steps:
        pending = [r["actor"] for r in requests if r["needs_action"]]
        if not pending:
            break

        actions: dict[ActorId, str] = {}
        for actor in pending:
            completion = policy(actor, histories[actor])
            actions[actor] = completion
            histories[actor].append({"role": "assistant", "content": completion})

        requests = env.step(actions)
        if not requests:
            break

        for req in requests:
            histories.setdefault(req["actor"], []).append(req["message"])
            totals.setdefault(req["actor"], 0.0)
            totals[req["actor"]] += float(req["reward"])

        if verbose:
            show_requests(requests, title=f"step={steps + 1} -> requests")

        steps += 1

    return EpisodeResult(histories=histories, total_reward=totals, steps=steps)


res = run_episode(make("openenv/echo"), policy=lambda _actor, _msgs: "hi", max_steps=3, verbose=True)
print("steps:", res.steps)
print("total_reward:", res.total_reward)

# %% [markdown]
# ## 2a) Registry and direct instantiation
#
# Use `make(...)` for registry-based envs; direct class construction is also supported.

# %%
print("Some registered envs:", list_envs()[:8])

# Preferred:
env = make("simple/tft_ipd", env_kwargs={"num_turns": 2})

# Direct construction (still supported):
direct_env = TftIpdEnv(num_turns=2)

# %% [markdown]
# ## 4) Multi-agent example: Connect4
#
# The env exposes multiple actors and uses `needs_action` to indicate whose turn it is.
# This policy parses `Legal actions: [...]` and picks the first legal column.

# %%
_LEGAL_ACTIONS_RE = re.compile(r"Legal actions:\s*(\[[^\]]*\])")


def parse_legal_actions_from_text(text: str) -> list[int]:
    m = _LEGAL_ACTIONS_RE.search(text)
    if not m:
        raise ValueError("Expected 'Legal actions: [...]' in the last env message.")
    value = ast.literal_eval(m.group(1))
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"Could not parse legal actions from: {m.group(1)!r}")
    return value


def connect4_policy(_actor: ActorId, messages: list[Message]) -> str:
    legal = parse_legal_actions_from_text(messages[-1]["content"])
    return str(legal[0])


env = make("openenv/connect4", env_kwargs={"opponent": None})
res = run_episode(env, policy=connect4_policy, max_steps=8, verbose=False)

print("actors:", sorted(res.histories.keys()))
print("steps:", res.steps)
for actor, total in sorted(res.total_reward.items()):
    print(actor, "total_reward:", total)

# %% [markdown]
# ## 5) Vectorization: many worlds, same API
#
# `make(..., num_envs=N)` returns a batched env. Use `autoreset=True` to restart completed envs.
# Actor ids include `env=` metadata (e.g. `agent::env=1`).

# %%
venv = make("simple/tft_ipd", env_kwargs={"num_turns": 2}, num_envs=4, autoreset=True)


def ipd_policy(_actor: ActorId, _messages: list[Message]) -> str:
    # Always cooperate.
    return "<action>A</action>"


res = run_episode(venv, policy=ipd_policy, max_steps=5, verbose=False)

print("num_actors:", len(res.histories))
print("actors:", sorted(res.histories.keys())[:6], "..." if len(res.histories) > 6 else "")
print("steps:", res.steps)
print("total_reward_by_actor:", {k: round(v, 3) for k, v in sorted(res.total_reward.items())})

# %% [markdown]
# ### 5b) Vectorized + multi-agent
#
# Two env instances with two agents each produce four actors:
# `"player_a::env=0"`, `"player_b::env=0"`, `"player_a::env=1"`, `"player_b::env=1"`.

# %%
venv2 = make("openenv/connect4", env_kwargs={"opponent": None}, num_envs=2, autoreset=True)
res = run_episode(venv2, policy=connect4_policy, max_steps=6, verbose=False)

print("actors:", sorted(res.histories.keys()))
print("steps:", res.steps)

# %% [markdown]
# ### 5c) Batching heterogeneous envs
#
# `make(..., num_envs=N)` is the standard path. You can also batch explicit env instances,
# including mixed env types.

# %%
mixed = batch_envs(
    [
        make("openenv/echo"),
        make("simple/tft_ipd", env_kwargs={"num_turns": 1}),
    ]
)


def mixed_policy(actor: ActorId, _messages: list[Message]) -> str:
    agent_id = actor_agent_id(actor)
    if agent_id == "agent":
        return "hi"
    if agent_id == "player":
        return "<action>A</action>"
    raise KeyError(f"Unknown agent_id: {agent_id!r}")


res = run_episode(mixed, policy=mixed_policy, max_steps=2)
print("mixed actors:", sorted(res.histories.keys()))
print("mixed total_reward:", {k: round(v, 3) for k, v in sorted(res.total_reward.items())})

# %% [markdown]
# ## 6) Out-of-process hosting
#
# `make(..., mode="subprocess")` runs `python -m gyllm.rpc_server ...`
# `make(..., mode="docker")` runs `docker run ... python -m gyllm.rpc_server ...`
#
# The returned client implements the same API as an in-memory env.

# %% [markdown]
# ### 6a) Subprocess-hosted env
#
# No image build required.

# %%
remote = make(
    "openenv/echo",
    mode="subprocess",
)
res = run_episode(remote, policy=lambda _a, _m: "hello", max_steps=2, verbose=True)
remote.close()
print("remote total_reward:", res.total_reward)

# %% [markdown]
# ### 6b) Docker-hosted env
#
# The container must be able to import `gyllm`. If you're using uv locally, build an
# image that installs `gyllm` (e.g. `uv pip install gyllm`) or mount the repo and set
# `PYTHONPATH` accordingly.

# %%
RUN_DOCKER_DEMO = False  # set True to run

if RUN_DOCKER_DEMO:
    # This assumes you're running the notebook inside a container or VM with `/workspace` mounted
    # and access to the docker daemon (/var/run/docker.sock).
    #
    # We reuse the current container's mounted volumes so the nested container can import `gyllm`
    # from `/workspace/src` without needing a custom-built image.
    container_id = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    remote = make(
        "openenv/echo",
        mode="docker",
        image="gyllm:dev",
        docker_args=[
            "--pull=never",
            "--volumes-from",
            f"{container_id}:ro",
            "-w",
            "/workspace",
            "-e",
            "PYTHONPATH=/workspace/src",
        ],
    )
    res = run_episode(remote, policy=lambda _a, _m: "hi from docker", max_steps=2)
    remote.close()
    print("docker total_reward:", res.total_reward)

# %% [markdown]
# ### 6c) Vectorizing remote envs
#
# Host multiple envs and batch the clients with `batch_envs`.

# %%
with ExitStack() as stack:
    envs = [
        stack.enter_context(
            make(
                "openenv/echo",
                mode="subprocess",
            )
        )
        for _ in range(3)
    ]
    vremote = batch_envs(envs)
    res = run_episode(vremote, policy=lambda _a, _m: "batched", max_steps=2)
    print("actors:", sorted(res.histories.keys()))
    print("total_reward:", {k: round(v, 3) for k, v in sorted(res.total_reward.items())})

# %% [markdown]
# ## 7) Writing an env
#
# Define `agents` and implement `_system_message()`, `reset()`, and `step()`.
# Env implementations set episode metadata on requests.

# %%
class CounterEnv(LLMEnv):
    agents = ["agent"]

    def __init__(self, *, target: int = 3) -> None:
        super().__init__()
        self.target = int(target)
        self.value = 0

    def _system_message(self, actor: ActorId) -> Message:
        agent = self.agent_id(actor)
        if agent != "agent":
            raise KeyError(f"Unknown agent: {agent!r}")
        return {
            "role": "system",
            "content": (
                "You are in CounterEnv.\n"
                "Each turn, increment the counter by sending `inc`.\n"
                "When the counter reaches the target, the episode ends."
            ),
        }

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        self._begin_episode()
        self.value = 0
        requests = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": f"Counter reset. value={self.value} target={self.target}"},
                "needs_action": True,
            }
        ]
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = False
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        actions = self._normalize_actions(actions)
        action = actions["agent"].strip().lower()
        if action != "inc":
            requests = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": -1.0,
                    "message": {"role": "user", "content": f"Invalid action {action!r}. Expected 'inc'."},
                    "needs_action": True,
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = bool(done)
            return requests

        self.value += 1
        done = self.value >= self.target
        reward = 1.0 if done else 0.0
        requests = [
            {
                "actor": make_actor_id("agent"),
                "reward": reward,
                "message": {"role": "user", "content": f"value={self.value} done={done}"},
                "needs_action": not done,
            }
        ]
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = bool(done)
        return requests


env = CounterEnv(target=3)
res = run_episode(env, policy=lambda _a, _m: "inc", max_steps=10, verbose=True)
print("total_reward:", res.total_reward)

# %% [markdown]
# ## 8) Summary
#
# - Actors unify single-agent, multi-agent, and batched usage.
# - `make(...)` works for local, batched, and out-of-process envs.
