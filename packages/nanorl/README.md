# nanorl

Rollout and RL utilities for LLM agents, built on top of `gyllm`.

Repo: https://github.com/RedTachyon/nanorl

## Install

```bash
uv pip install nanorl
```

### DGX Spark (nanorl + vllm)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --prerelease=allow \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.14.0/vllm-0.14.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl" \
  "nanorl[spark]"
```
