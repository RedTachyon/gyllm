"""
NanoLLM: a lightweight, external vLLM frontend that reuses an *existing*
`transformers` model's weights (single GPU, single process).

Goal
  model = AutoModelForCausalLM.from_pretrained(...)
  llm = NanoLLM(model, tokenizer=..., ...)
  llm.generate(...)  # behaves like vllm.LLM

Key property
  After construction, vLLM parameters are rebound to share storage with the
  provided HF model parameters (and for padded vocab weights, the HF weights
  are rebound to views into vLLM's padded tensors). In-place weight edits will
  be reflected by inference without reloading.

Notes / limitations
  - Prioritizes correctness and compatibility on a single GPU.
  - Forces vLLM to run in single-process ("uni") mode; weight sharing will not
    work with multi-process workers.
  - Uses vLLM's "transformers" backend (`model_impl="transformers"`) so the
    module structure matches Hugging Face naming.
"""

import functools
import tempfile
from typing import Any

import torch

from nanorl.rollout.episode import SamplingParams


def _rebind_padded_vocab_weight(vllm_weight: Any, hf_weight_param: Any) -> None:
    """Bind HF vocab weights into a padded vLLM embedding tensor."""
    vocab = hf_weight_param.shape[0]
    if vllm_weight.ndim != 2 or hf_weight_param.ndim != 2:
        raise ValueError("Expected 2D embedding/LM head weights for sharing.")
    if vllm_weight.shape[1] != hf_weight_param.shape[1]:
        raise ValueError(
            "Hidden size mismatch for shared vocab weight: "
            f"vLLM={tuple(vllm_weight.shape)} vs "
            f"HF={tuple(hf_weight_param.shape)}"
        )
    if vllm_weight.shape[0] < vocab:
        raise ValueError(
            f"vLLM vocab weight is smaller than HF vocab weight: vLLM={vllm_weight.shape[0]} vs HF={vocab}"
        )

    with torch.no_grad():
        vllm_weight[:vocab].copy_(hf_weight_param.data)
        if vllm_weight.shape[0] > vocab:
            vllm_weight[vocab:].zero_()
        hf_weight_param.data = vllm_weight[:vocab]


def _lookup_hf_param(
    name: str,
    hf_params: dict[str, Any],
    prefixes: set[str],
) -> Any | None:
    """Find a HF parameter that matches a vLLM parameter name."""
    candidates = {name}
    for pref in prefixes:
        if name.startswith(pref):
            candidates.add(name[len(pref) :])
    for base in list(candidates):
        for pref in prefixes:
            candidates.add(pref + base)
    for from_pref in prefixes:
        if name.startswith(from_pref):
            rest = name[len(from_pref) :]
            for to_pref in prefixes:
                candidates.add(to_pref + rest)

    for cand in candidates:
        if cand in hf_params:
            return hf_params[cand]
    return None


def _bind_vllm_weights(vllm_model: Any, hf_model: Any) -> dict[str, int]:
    """Share vLLM parameters with HF model tensors and report stats."""
    tgt_param = next(iter(vllm_model.parameters()), None)
    tgt_device = tgt_param.device if tgt_param is not None else None
    tgt_dtype = next((p.dtype for p in vllm_model.parameters() if p.is_floating_point()), None)
    with torch.no_grad():
        if tgt_device is not None and tgt_dtype is not None:
            hf_model.to(device=tgt_device, dtype=tgt_dtype)
        elif tgt_device is not None:
            hf_model.to(device=tgt_device)
        hf_model.eval()

    vllm_base = getattr(vllm_model, "model", None)
    if vllm_base is not None and hasattr(vllm_base, "get_input_embeddings"):
        vllm_in = vllm_base.get_input_embeddings()
    else:
        vllm_in = None
    hf_in = hf_model.get_input_embeddings() if hasattr(hf_model, "get_input_embeddings") else None

    if vllm_in is not None and hf_in is not None:
        vllm_in_w = getattr(vllm_in, "weight", None)
        hf_in_w = getattr(hf_in, "weight", None)
        if isinstance(vllm_in_w, torch.Tensor) and isinstance(hf_in_w, torch.nn.Parameter):
            if vllm_in_w.shape == hf_in_w.shape:
                vllm_in_w.data = hf_in_w.data
            else:
                _rebind_padded_vocab_weight(vllm_in_w, hf_in_w)

    vllm_head = getattr(vllm_model, "lm_head", None)
    vllm_head_w = getattr(vllm_head, "weight", None) if vllm_head is not None else None

    hf_out = hf_model.get_output_embeddings() if hasattr(hf_model, "get_output_embeddings") else None
    if hf_out is None:
        hf_out = getattr(hf_model, "lm_head", None)
    hf_out_w = getattr(hf_out, "weight", None) if hf_out is not None else None

    if isinstance(vllm_head_w, torch.Tensor):
        if isinstance(hf_out_w, torch.nn.Parameter):
            if vllm_head_w.shape == hf_out_w.shape:
                vllm_head_w.data = hf_out_w.data
            else:
                _rebind_padded_vocab_weight(vllm_head_w, hf_out_w)
        else:
            text_cfg = getattr(getattr(hf_model, "config", None), "get_text_config", lambda: None)()
            tie = bool(getattr(text_cfg, "tie_word_embeddings", False))
            if not tie:
                raise ValueError(
                    "HF model has no output embeddings / lm_head, "
                    "but tie_word_embeddings is False. "
                    "Load the model with AutoModelForCausalLM "
                    "(or provide a model with output head weights)."
                )

    hf_params = dict(hf_model.named_parameters())
    prefixes = {"model.", "transformer."}
    base_prefix = getattr(hf_model, "base_model_prefix", None)
    if isinstance(base_prefix, str) and base_prefix:
        prefixes.add(base_prefix + ".")

    with torch.no_grad():
        linked = 0
        total = 0
        missing = 0
        shape_mismatch = 0
        for name, p in vllm_model.named_parameters():
            total += 1
            hf_p = _lookup_hf_param(name, hf_params, prefixes)
            if hf_p is None:
                missing += 1
                continue
            if p.shape != hf_p.shape:
                shape_mismatch += 1
                continue
            p.data = hf_p.data
            linked += 1

    return {
        "linked": linked,
        "total": total,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


class NanoLLM:
    """A thin wrapper that behaves like `vllm.LLM`, backed by an existing
    Hugging Face model object whose weights are shared in-place."""

    def __init__(
        self,
        hf_model: Any,
        *,
        tokenizer: str | Any | None = None,
        model_id: str | None = None,
        **vllm_kwargs: Any,
    ) -> None:
        """Initialize NanoLLM around an existing HF model instance.

        Args:
            hf_model: Hugging Face model instance to share weights with.
            tokenizer: Tokenizer object or identifier.
            model_id: Optional model id/path override for vLLM.
            **vllm_kwargs: Additional arguments forwarded to vLLM.

        Raises:
            ValueError: If model_id cannot be inferred from the HF model.
            RuntimeError: If vLLM is not configured for single-process mode.
        """
        self.hf_model = hf_model

        resolved_model_id = (
            model_id
            or getattr(hf_model, "name_or_path", None)
            or getattr(
                getattr(hf_model, "config", None),
                "_name_or_path",
                None,
            )
        )
        if not isinstance(resolved_model_id, str) or not resolved_model_id:
            raise ValueError("Pass model_id=... (model path or hub id) when hf_model has no name_or_path.")

        if tokenizer is None:
            tokenizer_path = resolved_model_id
        elif isinstance(tokenizer, str):
            tokenizer_path = tokenizer
        else:
            # Tokenizer object: save to a temp dir so vLLM can load it by path.
            tmp_dir = tempfile.mkdtemp(prefix="nanollm_tokenizer_")
            tokenizer.save_pretrained(tmp_dir)
            tokenizer_path = tmp_dir

        # Defaults tuned for: single GPU + transformers backend + weight sharing
        vllm_kwargs.setdefault("distributed_executor_backend", "uni")
        vllm_kwargs.setdefault("tensor_parallel_size", 1)
        vllm_kwargs.setdefault("pipeline_parallel_size", 1)
        vllm_kwargs.setdefault("model_impl", "transformers")
        vllm_kwargs.setdefault("load_format", "dummy")
        vllm_kwargs.setdefault("enforce_eager", True)

        # If not specified, infer trust_remote_code from the HF config.
        if "trust_remote_code" not in vllm_kwargs:
            auto_map = getattr(getattr(hf_model, "config", None), "auto_map", None)
            vllm_kwargs["trust_remote_code"] = bool(auto_map)

        # If not specified, match dtype to the provided HF model.
        if "dtype" not in vllm_kwargs:
            src_dtype = next((p.dtype for p in hf_model.parameters() if p.is_floating_point()), None)
            if src_dtype is not None:
                vllm_kwargs["dtype"] = src_dtype

        from vllm import LLM  # import late to respect env vars

        self._llm = LLM(model=resolved_model_id, tokenizer=tokenizer_path, **vllm_kwargs)

        backend = self._llm.llm_engine.vllm_config.parallel_config.distributed_executor_backend
        if backend != "uni":
            raise RuntimeError(
                "NanoLLM requires `distributed_executor_backend='uni'` "
                "(single-process) for weight sharing. Got "
                f"{backend!r}."
            )

        # Bind vLLM parameters to the provided HF model weights.
        self.sync_weights()

    @property
    def llm(self) -> Any:
        """Return the underlying vLLM instance.

        Returns:
            vLLM LLM instance.
        """
        return self._llm

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the vLLM instance.

        Args:
            name: Attribute name to resolve on the vLLM object.

        Returns:
            Attribute value from the underlying vLLM instance.
        """
        # Delegate to vllm.LLM for API compatibility (generate, chat, encode…)
        return getattr(self._llm, name)

    def generate(self, prompts: str | list[str], sampling_params: SamplingParams, **kwargs: Any) -> Any:
        return self._llm.generate(prompts, sampling_params, **kwargs)

    def sync_weights(self, *, strict: bool = True) -> None:
        """(Re)bind vLLM parameters to the current HF model tensors.

        Args:
            strict: Whether to enforce a minimum linkage ratio.

        Raises:
            RuntimeError: If too few parameters were linked in strict mode.
        """

        hf_model = self.hf_model

        # Must be single-process; in uni this is a direct call.
        results = self._llm.apply_model(functools.partial(_bind_vllm_weights, hf_model=hf_model))
        stats = results[0] if results else None
        if not strict or not isinstance(stats, dict):
            return
        total = int(stats.get("total", 0) or 0)
        linked = int(stats.get("linked", 0) or 0)
        if total and linked / total < 0.9:
            raise RuntimeError(
                "NanoLLM weight sharing linked too few parameters "
                f"({linked}/{total}). This usually means HF/vLLM parameter "
                "names didn't align. Try passing an HF *causal LM* model "
                "(e.g., AutoModelForCausalLM) and/or set `model_id=` to the "
                "original repo/path used to load the model."
            )
