# %% [markdown]
# # GYLLM env API demo (text-in / text-out)
#
# This notebook shows:
# - Using a text-based env directly (you maintain message history)
#
# Prerequisite: `gyllm` should be importable in your notebook kernel (e.g. `pip install -e .`).

# %%

import gyllm

# %%
from gyllm.core import ActorId, actor_agent_id, make_actor_id

# %% [markdown]
# ## 1) Single-agent env (manual history)
#
# Convention: use `gyllm.make(...)` to create envs by name. Direct class instantiation
# still works for custom experiments, but `make` is the default entry point.
#
# The env returns a `Request`:
# - `actor`: string id like `agent` or `agent::env=3`
# - `reward`: reward from the last transition
# - `message`: a single environment "user" message to append to your history
# - `needs_action`: whether the agent should respond next
# - `episode_id`: integer episode index for this env instance
# - `episode_start`: True on the first request of a new episode
# - `episode_end`: True when the episode is finished
#
# Note: action formatting is env-specific. `TftIpdEnv` expects actions in `<action>...</action>` tags.

# %%
env = gyllm.make("simple/tft_ipd", env_kwargs={"num_turns": 3})
player = make_actor_id("player")

requests = env.reset()

req = requests[0]
history: list[dict[str, str]] = [req["system_message"], req["message"]]

print("actor:", req["actor"])
print("needs_action:", req["needs_action"])
print("last message:", history[-1])

# %%
def simple_agent_policy(messages: list[dict[str, str]]) -> str:
    # For demo purposes: always cooperate.
    return "<action>A</action>"


total_reward = 0.0

while True:
    action_text = simple_agent_policy(history)
    actions = {player: action_text}

    history.append({"role": "assistant", "content": action_text})

    requests = env.step(actions)
    if not requests:
        break

    req = requests[0]
    total_reward += req["reward"]

    history.append(req["message"])

    print("reward:", req["reward"], "| needs_action:", req["needs_action"])
    print("env says:", history[-1]["content"])

    if not req["needs_action"]:
        break

print("total_reward:", total_reward)

# %% [markdown]
# ## 2) Two-agent env (separate histories per agent)
#
# `IpdEnv` returns one request per agent each step.
# Each agent gets its own private message stream (the env shares only actions).

# %%
env2 = gyllm.make("simple/ipd", env_kwargs={"num_turns": 2})

a = make_actor_id("player_a")
b = make_actor_id("player_b")

histories: dict[ActorId, list[dict[str, str]]] = {
    a: [],
    b: [],
}
requests = env2.reset()
for req in requests:
    histories[req["actor"]] = [req["system_message"], req["message"]]


def policy_a(messages: list[dict[str, str]]) -> str:
    return "<action>A</action>"


def policy_b(messages: list[dict[str, str]]) -> str:
    return "<action>B</action>"


totals = {"player_a": 0.0, "player_b": 0.0}

while True:
    actions = {a: policy_a(histories[a]), b: policy_b(histories[b])}

    histories[a].append({"role": "assistant", "content": actions[a]})
    histories[b].append({"role": "assistant", "content": actions[b]})

    requests = env2.step(actions)
    if not requests:
        break

    for req in requests:
        agent_name = actor_agent_id(req["actor"])
        totals[agent_name] += req["reward"]
        histories[req["actor"]].append(req["message"])
        print(req["actor"], "reward:", req["reward"], "| last:", histories[req["actor"]][-1]["content"])

    if all(not r["needs_action"] for r in requests):
        break

print("totals:", totals)

# %% [markdown]
# ## 3) Vectorization (multi-world batching)
#
# Vectorization is just “more actors”: each env copy gets its own env tag (e.g. `agent::env=2`).
# This uses the same API as multi-agent envs.
# Use `autoreset=True` if you want finished envs to restart automatically.

# %%
venv = gyllm.make("simple/tft_ipd", env_kwargs={"num_turns": 1}, num_envs=4, autoreset=True)
reqs = venv.reset()
print("actors:", [r["actor"] for r in reqs])

actions = {r["actor"]: "<action>A</action>" for r in reqs}
reqs2 = venv.step(actions)
print("step rewards:", {r["actor"]: r["reward"] for r in reqs2})

# %% [markdown]
# ## 4) Dual-mode hosting (in-memory vs out-of-process)
#
# The same env can be used:
# - in-process: `env = gyllm.make(\"simple/tft_ipd\", ...)`
# - out-of-process: `env = gyllm.make(\"simple/tft_ipd\", mode=\"subprocess\", ...)`
# - in Docker: `env = gyllm.make(\"simple/tft_ipd\", mode=\"docker\", image=..., ...)`
#   (the image must have `gyllm` installed; e.g. `uv pip install gyllm`)
#
# Out-of-process is useful for isolation and for matching the “env in a container” deployment style.

# %%
remote = gyllm.make(
    "simple/tft_ipd",
    env_kwargs={"num_turns": 1},
    mode="subprocess",
)
print("remote actors:", remote.actors)
reqs = remote.reset()
reqs2 = remote.step({reqs[0]["actor"]: "<action>A</action>"})
print("remote reward:", reqs2[0]["reward"])
remote.close()
