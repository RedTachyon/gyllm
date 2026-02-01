# Repository Guidelines

## Project Structure & Module Organization

- `packages/nanorl/src/nanorl/`: internal library for learning/training code (currently minimal/skeleton).
- `packages/gyllm/src/gyllm/`: internal library providing an environment API (in the spirit of `gym`/`gymnasium`) plus supporting LLM utilities.
- `scripts/`: runnable benchmarking and demo scripts (e.g., `bench_llm_throughput.py`, `nano_llm.py`).
- `notebooks/`: Jupyter notebooks for interactive exploration.
- `deps/`: vendored dependencies (e.g., `deps/OpenEnv/`); avoid editing unless you are intentionally updating the vendored project.

## Build, Test, and Development Commands

Use `uv` for local workflows.

- Bootstrap:
  - `uv venv --python 3.12`
  - `uv sync` (DGX Spark: `uv sync --extra spark`)
- One-off commands (preferred): `uv run ...`
  - Hello world: `uv run python -c "print('hello')"`
  - Run a script: `uv run python check_vllm.py`
  - Type check: `uv run ty check --extra-search-path /usr/local/lib/python3.12/dist-packages`
- If you prefer activation:
  - `source .venv/bin/activate`
  - then run `python ...`

Tip: most scripts support `--help`.

## Hardware Notes (DGX Spark)

This repo is often developed on an NVIDIA DGX Spark “personal AI” workstation:

- Compute: NVIDIA GB10 Grace Blackwell Superchip (20-core Arm CPU: 10× Cortex-X925 + 10× Cortex-A725) + Blackwell GPU (5th-gen Tensor Cores).
- Memory: 128 GB coherent unified LPDDR5x (256-bit, ~273 GB/s).
- Storage: typically up to a 4 TB self-encrypting NVMe M.2 SSD.
- Networking: 10 GbE (RJ-45) + NVIDIA ConnectX-7 NIC (up to 200 Gbps).
- Software: NVIDIA DGX OS + NVIDIA AI software stack (local model development, fine-tuning, inference).

## Coding Style & Naming Conventions

- Indentation: 4 spaces; follow PEP 8.
- Prefer type hints and modern typing (`list[int]`, `str | None`) as used in `packages/gyllm/src/gyllm/`.
- Keep changes scoped: don’t reformat unrelated files (no repo-wide formatter config is enforced yet).
- Naming: modules/functions in `snake_case`, classes in `PascalCase`.

## Testing Guidelines

- There is no top-level test suite yet. If you add one, prefer `pytest` and place tests under `tests/` with names like `test_*.py`.
- Vendored tests under `deps/` are for that project and should not gate this repo unless explicitly wired in.

## Commit & Pull Request Guidelines

- No established commit message history yet (repo has no commits). Prefer Conventional Commits: `feat: ...`, `fix: ...`, `chore: ...`.
- PRs should include: a short summary, how to run/validate (commands), and any benchmark outputs if changing inference/throughput code.

## Security & Configuration Notes

- Large-model workflows may touch local caches (e.g., Hugging Face, `uv`); do not commit cache artifacts or secrets.
- When adding new models/checkpoints, prefer referencing by ID/path rather than checking weights into the repo.
