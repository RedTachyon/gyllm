#!/usr/bin/env python3
"""
Benchmark NanoLLM vs regular vLLM.LLM throughput (tokens/sec).

Examples
  # NanoLLM (shares weights from an HF model object)
  python bench_nanollm.py --engine nanollm --model Qwen/Qwen3-0.6B

  # Regular vLLM
  python bench_nanollm.py --engine vllm --model Qwen/Qwen3-0.6B

Notes
  - NanoLLM requires vLLM V1 in-process mode:
      VLLM_USE_V1=1 and VLLM_ENABLE_V1_MULTIPROCESSING=0
    This script uses those settings by default for both engines to keep the
    comparison apples-to-apples.
"""

import argparse
import os
import random
import time
from typing import Any


def _maybe_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported --hf-torch-dtype {name!r}. Use one of: {[*sorted(mapping.keys()), 'auto']}")
    return mapping[name]


def _sample_non_special_token_id(rng: random.Random, vocab_size: int, special_ids: set[int]) -> int:
    # Avoid special token IDs for more stable / realistic prompts.
    for _ in range(1000):
        tid = rng.randrange(vocab_size)
        if tid not in special_ids:
            return tid
    return 0


def build_token_prompts(tokenizer: Any, *, batch_size: int, prompt_len: int, seed: int) -> list[dict[str, Any]]:
    base_text = (
        "The quick brown fox jumps over the lazy dog.\n"
        "Write a short technical explanation of KV cache in LLM inference.\n"
    )
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        eos = getattr(tokenizer, "eos_token_id", None)
        base_ids = [int(eos) if eos is not None else 0]

    vocab_size = len(tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    reps = (prompt_len + len(base_ids) - 1) // len(base_ids)
    template = (base_ids * reps)[:prompt_len]

    prompts: list[dict[str, Any]] = []
    for i in range(batch_size):
        ids = list(template)
        rng = random.Random(seed * 1_000_003 + i)
        # Make prompts distinct to avoid any cache hits.
        num_mut = min(8, prompt_len)
        for j in range(1, num_mut + 1):
            ids[-j] = _sample_non_special_token_id(rng, vocab_size, special_ids)
        prompts.append({"prompt_token_ids": ids})
    return prompts


def count_output_tokens(request_outputs: list[Any]) -> int:
    total = 0
    for ro in request_outputs:
        for out in getattr(ro, "outputs", []) or []:
            total += len(getattr(out, "token_ids", []) or [])
    return total


def _cuda_sync_if_available():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def bench_generate(
    llm: Any, prompts: list[dict[str, Any]], sampling_params: Any, *, warmup: int, iters: int
) -> dict[str, Any]:
    for _ in range(warmup):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    times: list[float] = []
    out_tokens: list[int] = []
    for _ in range(iters):
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        _cuda_sync_if_available()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        out_tokens.append(count_output_tokens(outputs))

    total_s = sum(times)
    total_out = sum(out_tokens)
    return {
        "iters": iters,
        "total_s": total_s,
        "total_out_tokens": total_out,
        "per_iter_s": times,
        "per_iter_out_tokens": out_tokens,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["nanollm", "vllm"], required=True, help="Which engine to benchmark.")
    p.add_argument("--model", required=True, help="Model repo id or local path.")
    p.add_argument("--tokenizer", default=None, help="Tokenizer repo id or local path (default: --model).")
    p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for HF/vLLM.")
    p.add_argument("--revision", default=None, help="Optional model revision (branch/tag/commit).")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # vLLM knobs (kept minimal)
    p.add_argument("--dtype", default="bfloat16", help="vLLM dtype (e.g. auto, float16, bfloat16).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument(
        "--distributed-executor-backend",
        default="uni",
        choices=["uni", "mp", "ray", "external_launcher"],
        help="vLLM distributed executor backend (NanoLLM requires uni).",
    )
    p.add_argument(
        "--model-impl", default="auto", help="vLLM model_impl for regular vLLM engine (e.g. auto, transformers)."
    )

    # HF-only knobs (NanoLLM path)
    p.add_argument(
        "--hf-torch-dtype",
        default="auto",
        help="HF torch_dtype for NanoLLM model load (auto, float16, bfloat16, float32).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # NanoLLM requires in-process EngineCore.
    if args.engine == "nanollm":
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    else:
        # Keep the default comparison in-process unless the user explicitly
        # overrides via env vars.
        os.environ.setdefault("VLLM_USE_V1", "1")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    if args.engine == "nanollm" and args.distributed_executor_backend != "uni":
        raise SystemExit("NanoLLM requires `--distributed-executor-backend uni`.")

    tokenizer_id = args.tokenizer or args.model

    # Build prompts with the HF tokenizer (stable token-length workload).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=args.trust_remote_code, revision=args.revision
    )
    prompts = build_token_prompts(tokenizer, batch_size=args.batch_size, prompt_len=args.prompt_len, seed=args.seed)

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_len,
        ignore_eos=True,
        detokenize=False,
    )

    vllm_common_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "distributed_executor_backend": args.distributed_executor_backend,
    }

    if args.engine == "nanollm":
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            torch_dtype=_maybe_torch_dtype(args.hf_torch_dtype),
        )

        from nanorl.rollout.nanollm import NanoLLM

        # NanoLLM requires transformers backend for name-aligned weight sharing.
        llm = NanoLLM(
            hf_model, tokenizer=tokenizer, model_id=args.model, model_impl="transformers", **vllm_common_kwargs
        )
    else:
        llm = LLM(model=args.model, tokenizer=tokenizer_id, model_impl=args.model_impl, **vllm_common_kwargs)

    # Try to sanity check model length after init (helps avoid silent truncation).
    try:
        max_len = int(llm.llm_engine.model_config.max_model_len)  # type: ignore[attr-defined]
        if args.prompt_len + args.output_len > max_len:
            raise SystemExit(
                f"prompt_len+output_len ({args.prompt_len}+{args.output_len}) "
                f"exceeds max_model_len={max_len}. Reduce lengths or pass "
                "--max-model-len."
            )
    except Exception:
        pass

    stats = bench_generate(llm, prompts, sampling_params, warmup=args.warmup, iters=args.iters)

    total_s = stats["total_s"]
    total_out = stats["total_out_tokens"]
    total_in = args.batch_size * args.prompt_len * args.iters
    out_tok_s = total_out / total_s if total_s > 0 else float("nan")
    total_tok_s = (total_in + total_out) / total_s if total_s > 0 else float("nan")

    print("=== Benchmark ===")
    print(f"engine: {args.engine}")
    print(f"model: {args.model}")
    print(f"tokenizer: {tokenizer_id}")
    print(f"batch_size: {args.batch_size}")
    print(f"prompt_len: {args.prompt_len} tokens")
    print(f"output_len: {args.output_len} tokens")
    print(f"iters: {args.iters} (warmup={args.warmup})")
    print(f"elapsed: {total_s:.3f} s")
    print(f"output tokens: {total_out}")
    print(f"throughput (output tok/s): {out_tok_s:.2f}")
    print(f"throughput (input+output tok/s): {total_tok_s:.2f}")


if __name__ == "__main__":
    main()
