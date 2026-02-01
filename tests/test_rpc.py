import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gyllm.batch import batch_envs
from gyllm.core import LLMEnv, actor_env_id, make_actor_id
from gyllm.rpc import docker_env, subprocess_env


def test_subprocess_env_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    client = subprocess_env(
        env="gyllm.envs.openenv.echo:EchoEnv",
        env_kwargs={},
        pythonpath=str(repo_root / "packages" / "gyllm" / "src"),
    )
    try:
        assert isinstance(client, LLMEnv)
        assert client.actors == [make_actor_id("agent")]
        reqs = client.reset()
        assert reqs[0]["actor"] == make_actor_id("agent")
        assert reqs[0]["episode_id"] == 0
        assert reqs[0]["episode_start"] is True
        out = client.step({make_actor_id("agent"): "hi"})
        assert out[0]["actor"] == make_actor_id("agent")
        assert out[0]["episode_start"] is False
    finally:
        client.close()


def test_subprocess_env_batch_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    clients = [
        subprocess_env(
            env="gyllm.envs.openenv.echo:EchoEnv",
            env_kwargs={},
            pythonpath=str(repo_root / "packages" / "gyllm" / "src"),
        )
        for _ in range(2)
    ]
    try:
        venv = batch_envs(clients)
        assert isinstance(venv, LLMEnv)
        reqs = venv.reset()
        actions = {r["actor"]: "hello" for r in reqs if r["needs_action"]}
        out = venv.step(actions)
        assert {actor_env_id(r["actor"]) for r in out} == {0, 1}
    finally:
        for c in clients:
            c.close()


def _docker_accessible() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "version"], check=False, capture_output=True, text=True)
    except OSError:
        return False
    return proc.returncode == 0


def _docker_mount_args(repo_root: Path) -> list[str]:
    """
    Bind mounts in `docker run -v ...` are resolved on the docker *host*.

    When tests run inside the dev container (common in this repo), we don't know the host path for the
    repo, but we *can* reuse the dev container's volumes via `--volumes-from`.
    """
    if Path("/.dockerenv").exists():
        container_id = Path("/etc/hostname").read_text(encoding="utf-8").strip()
        return [
            "--volumes-from",
            f"{container_id}:ro",
            "-w",
            "/workspace",
            "-e",
            "PYTHONPATH=/workspace/packages/gyllm/src",
        ]

    return [
        "-v",
        f"{repo_root}:/workspace:ro",
        "-w",
        "/workspace",
        "-e",
        "PYTHONPATH=/workspace/packages/gyllm/src",
    ]


@pytest.mark.skipif(not _docker_accessible(), reason="docker not available/accessible")
def test_docker_env_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    preferred = os.environ.get("GYLLM_TEST_DOCKER_IMAGE")
    candidates = [preferred] if preferred else []
    candidates += ["vllm-ariel:0.1", "python:3.12-slim", "python:3.12"]

    image = None
    for cand in candidates:
        if not cand:
            continue
        inspected = subprocess.run(["docker", "image", "inspect", cand], check=False, capture_output=True, text=True)
        if inspected.returncode == 0:
            image = cand
            break
    if image is None:
        pytest.skip(
            "No suitable docker image available locally. "
            "Set GYLLM_TEST_DOCKER_IMAGE or pre-pull an image (e.g. python:3.12-slim)."
        )
    assert image is not None

    client = docker_env(
        image=image,
        env="gyllm.envs.openenv.echo:EchoEnv",
        env_kwargs={},
        docker_args=[
            "--pull=never",
            *_docker_mount_args(repo_root),
        ],
    )
    try:
        assert isinstance(client, LLMEnv)
        assert client.actors == [make_actor_id("agent")]
        reqs = client.reset()
        assert reqs[0]["needs_action"] is True
        assert reqs[0]["episode_id"] == 0
        out = client.step({reqs[0]["actor"]: "hello"})
        assert out[0]["actor"] == make_actor_id("agent")
        assert out[0]["message"]["content"] == "hello"
    finally:
        client.close()
