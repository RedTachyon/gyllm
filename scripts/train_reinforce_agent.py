"""
Train a NanoLLM model with REINFORCE using an Agent layer.
"""

import argparse
import os
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import torch
import torch.distributed as dist
import wandb
from omegaconf import MISSING, OmegaConf
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import gyllm
from gyllm.envs import AutoResetWrapper
from nanorl.agent import FixedAgent, InstructAgent, ReasoningAgent
from nanorl.rl import compute_reinforce_loss
from nanorl.rollout import NanoLLM
from nanorl.rollout.reporting import render_rollouts_html, summarize_rollouts


@dataclass
class EnvConfig:
    """Env configuration section."""

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset configuration section."""

    id: str | None = None
    config: str | None = None
    split: str = "train"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingConfig:
    """Sampling configuration section."""

    temperature: float = 1.0
    max_tokens: int = 512


@dataclass
class NanoLLMConfig:
    """NanoLLM runtime configuration."""

    gpu_memory_utilization: float = 0.4
    enable_sleep_mode: bool = True


@dataclass
class WandBConfig:
    """Weights & Biases logging configuration."""

    project: str = "nanorl-reinforce"
    env_name: str | None = None


@dataclass
class AgentConfig:
    """Agent configuration section."""

    type: str = "instruct"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    nanollm: NanoLLMConfig = field(default_factory=NanoLLMConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    num_envs: int = 1
    episodes: int = 1
    num_updates: int = 100
    lr: float = 1.0e-5
    max_grad_norm: float = 1.0
    minibatch_size: int = 2
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "cuda"
    activation_checkpointing: bool = False
    max_steps: int | None = None
    normalizer: str | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train a NanoLLM model with REINFORCE using an agent.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config (defaults to scripts/configs/reinforce_gsm8k.yaml).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the merged config and exit.",
    )
    return parser


def _build_agent(agent_cfg: AgentConfig, *, model, llm, tokenizer, sampling_params):
    """Instantiate an agent from config and shared LLM components."""
    agent_type = str(agent_cfg.type).lower()
    agent_kwargs = dict(agent_cfg.kwargs)
    if agent_type == "instruct":
        return InstructAgent(
            model,
            llm,
            tokenizer,
            sampling_params,
            **agent_kwargs,
        )
    if agent_type == "reasoning":
        return ReasoningAgent(
            model,
            llm,
            tokenizer,
            sampling_params,
            **agent_kwargs,
        )
    if agent_type == "fixed":
        return FixedAgent(**agent_kwargs)
    raise ValueError(f"Unknown agent type: {agent_type!r}")


def main(argv: Iterable[str] | None = None) -> None:
    """Run REINFORCE training with a configurable agent implementation."""
    parser = build_parser()
    args, overrides = parser.parse_known_args(list(argv) if argv is not None else None)

    config_path = (
        Path(args.config) if args.config else Path(__file__).resolve().parent / "configs" / "reinforce_gsm8k.yaml"
    )
    config = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(config_path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))

    if args.print_config:
        print(OmegaConf.to_yaml(config))
        return

    if config.wandb.env_name is None:
        config.wandb.env_name = config.env.name

    cfg = OmegaConf.to_object(config)
    assert isinstance(cfg, TrainConfig), "Config must resolve to TrainConfig."

    env_name = cfg.env.name
    env_kwargs = dict(cfg.env.kwargs)

    if cfg.dataset.id is not None:
        env_kwargs["dataset_id"] = cfg.dataset.id
        env_kwargs["dataset_config"] = cfg.dataset.config
        env_kwargs["dataset_split"] = cfg.dataset.split
        env_kwargs["dataset_kwargs"] = dict(cfg.dataset.kwargs)

    with ExitStack() as stack:
        wandb_config = OmegaConf.to_container(config, resolve=True)
        assert isinstance(wandb_config, dict), "W&B config must resolve to a mapping."
        wandb.init(project=cfg.wandb.project, config=wandb_config)  # ty: ignore[invalid-argument-type]
        stack.callback(wandb.finish)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
        )
        if cfg.activation_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        llm = NanoLLM(
            model,
            tokenizer=tokenizer,
            gpu_memory_utilization=cfg.nanollm.gpu_memory_utilization,
            enable_sleep_mode=cfg.nanollm.enable_sleep_mode,
        )

        env = gyllm.make(
            env_name,
            env_kwargs=env_kwargs,
            num_envs=cfg.num_envs,
        )
        env = AutoResetWrapper(env)
        stack.callback(env.close)
        if dist.is_available():
            stack.callback(dist.destroy_process_group)  # ty: ignore[possibly-missing-attribute]

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        device = next(model.parameters()).device

        sampling_params = SamplingParams(
            temperature=cfg.sampling.temperature,
            max_tokens=cfg.sampling.max_tokens,
        )
        agent = _build_agent(
            cfg.agent,
            model=model,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )

        for update in trange(cfg.num_updates):
            iter_start = perf_counter()
            model.eval()
            with torch.no_grad():
                llm.wake_up()
                rollout_start = perf_counter()
                rollouts = agent.rollout_autoreset_batched(
                    env,
                    max_episodes=cfg.episodes,
                    max_steps=cfg.max_steps,
                )
                rollout_time = perf_counter() - rollout_start
            llm.sleep(1)

            _tokens, mean_reward, _sample_text = summarize_rollouts(rollouts, tokenizer)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_rollouts = len(rollouts)
            if total_rollouts == 0:
                iter_time = perf_counter() - iter_start
                log_data = {
                    "update": update,
                    "rollouts": 0,
                    "mean_reward": mean_reward,
                    "timing/rollout_s": rollout_time,
                    "timing/grad_s": 0.0,
                    "timing/iter_s": iter_time,
                }
                wandb.log(log_data, step=update)
                print(f"update={update} skipped (no rollouts) iter_s={iter_time:.2f}")
                continue

            html_summary = render_rollouts_html(rollouts)

            grad_start = perf_counter()
            total_loss_value = 0.0
            total_assistant_tokens = 0.0
            total_logprob = 0.0
            reward_sum = 0.0

            for start in range(0, total_rollouts, cfg.minibatch_size):
                minibatch = rollouts[start : start + cfg.minibatch_size]
                loss, mb_metrics = compute_reinforce_loss(
                    minibatch,
                    model,
                    tokenizer,
                    device=device,
                )
                mb_size = len(minibatch)
                reward_sum += mb_metrics["avg_reward"] * mb_size
                total_assistant_tokens += mb_metrics["assistant_tokens"]
                total_logprob += mb_metrics["avg_logprob"] * mb_metrics["assistant_tokens"]

                if mb_metrics["assistant_tokens"] <= 0:
                    continue

                scale = mb_size / total_rollouts
                (loss * scale).backward()
                total_loss_value += float(loss.item()) * scale

            if total_assistant_tokens == 0:
                iter_time = perf_counter() - iter_start
                log_data = {
                    "update": update,
                    "rollouts": len(rollouts),
                    "mean_reward": mean_reward,
                    "timing/rollout_s": rollout_time,
                    "timing/grad_s": perf_counter() - grad_start,
                    "timing/iter_s": iter_time,
                    "rollout_html": wandb.Html(html_summary),
                }
                wandb.log(log_data, step=update)
                print(f"update={update} skipped (no assistant tokens) iter_s={iter_time:.2f}")
                continue

            avg_reward = reward_sum / total_rollouts
            metrics = {
                "avg_reward": avg_reward,
                "baseline": avg_reward,
                "avg_logprob": total_logprob / total_assistant_tokens,
                "assistant_tokens": float(total_assistant_tokens),
            }

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            grad_time = perf_counter() - grad_start
            iter_time = perf_counter() - iter_start

            log_data = {
                "loss": float(total_loss_value),
                "update": update,
                "rollouts": len(rollouts),
                "mean_reward": mean_reward,
                "timing/rollout_s": rollout_time,
                "timing/grad_s": grad_time,
                "timing/iter_s": iter_time,
                "rollout_html": wandb.Html(html_summary),
                **metrics,
            }
            wandb.log(log_data, step=update)
            print(
                f"update={update} loss={log_data['loss']:.4f} "
                f"avg_reward={log_data['avg_reward']:.3f} "
                f"assistant_tokens={log_data['assistant_tokens']:.0f} "
                f"rollout_s={rollout_time:.2f} grad_s={grad_time:.2f} iter_s={iter_time:.2f}"
            )


if __name__ == "__main__":
    main()
