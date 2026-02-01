# GyLLM

Lightweight LLM environments and rollout tools for reinforcement learning.

Packages:
- `gyllm`: environment API + registry + wrappers (`packages/gyllm`).
- `nanorl`: rollout + RL utilities (`packages/nanorl`).

## Preamble

This is a side project built by one man and one clanker. 
There will be sharp edges, some parts are undertested, and the RL code isn't going to scale by design.

Still, it seems to work at a fundamental level - reward goes up.

## What is it?

Gyllm = Gym + LLM. Clever, right?

It's my redesign of what an RL env could look like in the era of LLMs. The central abstraction is a `Request`, which carries an observation, maybe a reward, and it might call for an action.
Every env step returns zero or more requests.

This simple structure allows seamlessly implementing batched/vectorized environments, multi-agent environments, heterogeneously batched environments, all under a single abstraction.

NanoRL = Nano + RL. Clever, right?

It's a basic implementation of a few RL algorithms to work well with gyllm on a single GPU. Right now it has REINFORCE, PPO and GRPO.

The unusual thing about NanoRL is that it completely removes the need for separate inference and training copies of the model, removing the weight update overhead.
It does so by hacking into vllm's internal model and sharing weights with the transformers-based training model.

This approach, naturally, doesn't scale to more than one GPU. I only have one GPU at home, so that's fine.

## Status

| Area                       | Status          | Notes                                                   |
|----------------------------|-----------------|---------------------------------------------------------|
| Gyllm core API             | ✅ Stable-ish    | API is mostly stable.                                   |
| Batching/vectorization     | 🧪 Experimental | Works in practice, still evolving.                      |
| Subprocess/Docker runtimes | 🧩 Prototype    | Newer implementation, expect rough edges.               |
| OpenEnv environments       | 🧩 Prototype    | Early integrations, subject to change, might be broken. |
| NanoRL package             | 🧱 PoC          | Proof-of-concept utilities with basic functionality.    |

## Quickstart

```python
import gyllm

env = gyllm.make("openenv/echo")
requests = env.reset()
actions = {r["actor"]: "hi" for r in requests if r["needs_action"]}
requests = env.step(actions)
```

Run a couple of RL training scripts with `uv run`:

```bash
# PPO on tic tac toe
uv run scripts/train_ppo_agent.py --config scripts/configs/ppo_ttt.yaml

# GRPO on MATH
uv run scripts/train_grpo_agent.py --config scripts/configs/grpo_math.yaml
```

Note: by default, the training scripts (and nanorl) require Weights & Biases (`wandb`).
To run without logging, set `WANDB_MODE=disabled` (or `WANDB_MODE=offline`) in your environment.

## Install (PyPI)

Python 3.11+ (3.12 recommended).

Using `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install gyllm nanorl
```

### DGX Spark (nanorl + vllm)

Use the Spark-specific wheels and extras:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --prerelease=allow \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.14.0/vllm-0.14.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl" \
  "nanorl[spark]"
```

## Colab

Notebook that mirrors `scripts/ttt_reinforce.py`:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/RedTachyon/nanorl/blob/main/notebooks/ttt_reinforce_colab.ipynb
)

## Development (uv)

```bash
uv venv --python 3.12
uv sync
# For DGX Spark:
uv sync --extra spark

uv run python -c "print('hello')"
uv run scripts/train_grpo_agent.py --config scripts/configs/grpo_gsm8k.yaml
```


## License

MIT. See `LICENSE`.

OpenEnv envs in `gyllm` are adapted from https://github.com/meta-pytorch/OpenEnv (BSD 3-Clause).
