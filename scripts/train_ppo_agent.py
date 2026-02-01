"""
Train a NanoLLM model with PPO using an Agent layer and value head.
"""

import argparse
import os
import random
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
from nanorl.rl import (
    compute_ppo_policy_loss,
    compute_ppo_value_loss,
    ensure_value_head,
    prepare_ppo_rollouts,
)
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

    project: str = "nanorl-ppo"
    env_name: str | None = None


@dataclass
class AgentConfig:
    """Agent configuration section."""

    type: str = "instruct"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_range: float = 0.2
    clip_range_vf: float | None = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    normalize_advantages: bool = True
    epochs: int = 1
    target_kl: float | None = None


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    nanollm: NanoLLMConfig = field(default_factory=NanoLLMConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
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
    parser = argparse.ArgumentParser(description="Train a NanoLLM model with PPO using an agent.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config (defaults to scripts/configs/ppo_gsm8k.yaml).",
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
    """Run PPO training with a configurable agent implementation."""
    parser = build_parser()
    args, overrides = parser.parse_known_args(list(argv) if argv is not None else None)

    config_path = Path(args.config) if args.config else Path(__file__).resolve().parent / "configs" / "ppo_gsm8k.yaml"
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

        ensure_value_head(model)

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

            ppo_batches, prep_metrics = prepare_ppo_rollouts(
                rollouts,
                model,
                tokenizer,
                device=device,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
                normalize_advantages=cfg.ppo.normalize_advantages,
            )
            if not ppo_batches or prep_metrics["assistant_tokens"] <= 0:
                iter_time = perf_counter() - iter_start
                log_data = {
                    "update": update,
                    "rollouts": len(rollouts),
                    "mean_reward": mean_reward,
                    "timing/rollout_s": rollout_time,
                    "timing/grad_s": 0.0,
                    "timing/iter_s": iter_time,
                    "rollout_html": wandb.Html(html_summary),
                }
                wandb.log(log_data, step=update)
                print(f"update={update} skipped (no assistant tokens) iter_s={iter_time:.2f}")
                continue

            model.train()
            grad_start = perf_counter()
            total_tokens = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_kl = 0.0
            total_clip_fraction = 0.0
            total_value_clip_fraction = 0.0
            total_logprob = 0.0
            total_value = 0.0
            early_stop = False

            indices = list(range(len(ppo_batches)))
            for _epoch in range(cfg.ppo.epochs):
                random.shuffle(indices)
                for start in range(0, len(indices), cfg.minibatch_size):
                    minibatch = [ppo_batches[idx] for idx in indices[start : start + cfg.minibatch_size]]

                    optimizer.zero_grad(set_to_none=True)
                    policy_loss, policy_metrics = compute_ppo_policy_loss(
                        minibatch,
                        model,
                        device=device,
                        clip_range=cfg.ppo.clip_range,
                        entropy_coef=cfg.ppo.entropy_coef,
                    )
                    policy_loss.backward()

                    value_loss, value_metrics = compute_ppo_value_loss(
                        minibatch,
                        model,
                        device=device,
                        clip_range_vf=cfg.ppo.clip_range_vf,
                    )
                    (cfg.ppo.value_coef * value_loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    tokens = float(policy_metrics["assistant_tokens"])
                    if tokens <= 0:
                        continue

                    total_tokens += tokens
                    total_policy_loss += policy_metrics["policy_loss"] * tokens
                    total_value_loss += value_metrics["value_loss"] * tokens
                    total_entropy += policy_metrics["entropy"] * tokens
                    total_kl += policy_metrics["approx_kl"] * tokens
                    total_clip_fraction += policy_metrics["clip_fraction"] * tokens
                    total_value_clip_fraction += value_metrics["value_clip_fraction"] * tokens
                    total_logprob += policy_metrics["avg_logprob"] * tokens
                    total_value += value_metrics["value_mean"] * tokens

                    if cfg.ppo.target_kl is not None and policy_metrics["approx_kl"] > cfg.ppo.target_kl:
                        early_stop = True
                        break
                if early_stop:
                    break

            grad_time = perf_counter() - grad_start
            iter_time = perf_counter() - iter_start

            denom = max(total_tokens, 1.0)
            log_data = {
                "update": update,
                "rollouts": len(rollouts),
                "mean_reward": mean_reward,
                "avg_reward": prep_metrics["avg_reward"],
                "policy_loss": total_policy_loss / denom,
                "value_loss": total_value_loss / denom,
                "entropy": total_entropy / denom,
                "approx_kl": total_kl / denom,
                "clip_fraction": total_clip_fraction / denom,
                "value_clip_fraction": total_value_clip_fraction / denom,
                "avg_logprob": total_logprob / denom,
                "value_mean": total_value / denom,
                "assistant_tokens": total_tokens,
                "adv_mean": prep_metrics["adv_mean"],
                "adv_std": prep_metrics["adv_std"],
                "rollouts_used": prep_metrics["rollouts_used"],
                "rollouts_skipped": prep_metrics["rollouts_skipped"],
                "timing/rollout_s": rollout_time,
                "timing/grad_s": grad_time,
                "timing/iter_s": iter_time,
                "rollout_html": wandb.Html(html_summary),
            }
            wandb.log(log_data, step=update)
            print(
                f"update={update} policy_loss={log_data['policy_loss']:.4f} "
                f"value_loss={log_data['value_loss']:.4f} "
                f"avg_reward={log_data['avg_reward']:.3f} "
                f"assistant_tokens={log_data['assistant_tokens']:.0f} "
                f"rollout_s={rollout_time:.2f} grad_s={grad_time:.2f} iter_s={iter_time:.2f}"
            )


if __name__ == "__main__":
    main()
