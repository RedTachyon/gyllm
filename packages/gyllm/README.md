# gyllm

Environment API and wrappers for LLM agents. This package powers the env layer used by nanorl.

Repo: https://github.com/RedTachyon/nanorl

## Install

```bash
uv pip install gyllm
```

## Web UI

Install the optional web extras and launch the server:

```bash
uv pip install "gyllm[web]"
uv run gyllm-web --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in your browser.
