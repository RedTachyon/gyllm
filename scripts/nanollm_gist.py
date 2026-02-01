#!/usr/bin/env python3
"""
NanoLLM: a lightweight, external vLLM frontend that reuses an existing
`transformers` model's weights (single GPU, single process).
"""

import io
import os
import sys
import tempfile
from typing import Any

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["VLLM_NO_STDOUT_REDIRECT"] = "1"

# Fix for colab
try:
    sys.stdout.fileno()
except io.UnsupportedOperation:
    sys.stdout.fileno = lambda: 1  # ty: ignore
    sys.stderr.fileno = lambda: 2  # ty: ignore

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def _rebind_padded_vocab_weight(vllm_weight: torch.Tensor, hf_weight_param: torch.nn.Parameter) -> None:
    vocab = hf_weight_param.shape[0]
    with torch.no_grad():
        vllm_weight[:vocab].copy_(hf_weight_param.data)
        if vllm_weight.shape[0] > vocab:
            vllm_weight[vocab:].zero_()
        hf_weight_param.data = vllm_weight[:vocab]


def _match_param(name: str, hf_params: dict[str, torch.nn.Parameter]) -> torch.nn.Parameter | None:
    if name in hf_params:
        return hf_params[name]
    if name.startswith("model.") and name[len("model.") :] in hf_params:
        return hf_params[name[len("model.") :]]
    if name.startswith("transformer.") and name[len("transformer.") :] in hf_params:
        return hf_params[name[len("transformer.") :]]
    return None


def _bind_vllm_weights(vllm_model: Any, hf_model: Any) -> None:
    tgt_param = next(iter(vllm_model.parameters()))
    hf_model.to(device=tgt_param.device, dtype=tgt_param.dtype)
    hf_model.eval()

    vllm_in = vllm_model.model.get_input_embeddings()
    hf_in = hf_model.get_input_embeddings()
    vllm_in_w = vllm_in.weight
    hf_in_w = hf_in.weight
    if vllm_in_w.shape == hf_in_w.shape:
        vllm_in_w.data = hf_in_w.data
    else:
        _rebind_padded_vocab_weight(vllm_in_w, hf_in_w)

    vllm_head_w = vllm_model.lm_head.weight
    hf_out_w = hf_model.get_output_embeddings().weight
    if vllm_head_w.shape == hf_out_w.shape:
        vllm_head_w.data = hf_out_w.data
    else:
        _rebind_padded_vocab_weight(vllm_head_w, hf_out_w)

    hf_params = dict(hf_model.named_parameters())
    with torch.no_grad():
        for name, p in vllm_model.named_parameters():
            hf_p = _match_param(name, hf_params)
            if hf_p is None or p.shape != hf_p.shape:
                continue
            p.data = hf_p.data


class NanoLLM:
    def __init__(
        self,
        hf_model: Any,
        *,
        tokenizer: str | Any | None = None,
        model_id: str | None = None,
        **vllm_kwargs: Any,
    ) -> None:
        self.hf_model = hf_model
        resolved_model_id = model_id or getattr(hf_model, "name_or_path", None) or hf_model.config._name_or_path

        if tokenizer is None:
            tokenizer_path = resolved_model_id
        elif isinstance(tokenizer, str):
            tokenizer_path = tokenizer
        else:
            tmp_dir = tempfile.mkdtemp(prefix="nanollm_tokenizer_")
            tokenizer.save_pretrained(tmp_dir)
            tokenizer_path = tmp_dir

        vllm_kwargs.setdefault("distributed_executor_backend", "uni")
        vllm_kwargs.setdefault("tensor_parallel_size", 1)
        vllm_kwargs.setdefault("pipeline_parallel_size", 1)
        vllm_kwargs.setdefault("model_impl", "transformers")
        vllm_kwargs.setdefault("load_format", "dummy")
        vllm_kwargs.setdefault("enforce_eager", True)

        self._llm = LLM(model=resolved_model_id, tokenizer=tokenizer_path, **vllm_kwargs)
        self.sync_weights()

    @property
    def llm(self) -> Any:
        return self._llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def generate(self, prompts: str | list[str], sampling_params: SamplingParams, **kwargs: Any) -> Any:
        return self._llm.generate(prompts, sampling_params, **kwargs)

    def sync_weights(self) -> None:
        self._llm.apply_model(lambda vllm_model: _bind_vllm_weights(vllm_model, self.hf_model))


def main() -> None:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)

    messages = [
        {"role": "system", "content": "You are a concise, technical assistant."},
        {"role": "user", "content": "Write one sentence about in-place weight sharing."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = NanoLLM(hf_model, tokenizer=tokenizer, model_id=model_id)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=48)

    outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
    before = outputs[0].outputs[0].text.strip()
    print("=== before ===")
    print(before)

    with torch.no_grad():
        for param in hf_model.parameters():
            if torch.is_floating_point(param):
                param.zero_()

    outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
    after = outputs[0].outputs[0].text.strip()
    print("=== after ===")
    print(after)


if __name__ == "__main__":
    main()
