import json
import os
import subprocess
import sys
from dataclasses import dataclass
from types import TracebackType
from typing import IO, Any

from gyllm.core import ActorId, LLMEnv, Message, Request, make_actor_id


class RemoteEnvError(RuntimeError):
    """Error raised when the remote env reports an exception."""

    def __init__(self, *, error_type: str, message: str, traceback: str | None = None) -> None:
        """Initialize the remote error.

        Args:
            error_type: Remote exception type name.
            message: Error message from the remote env.
            traceback: Optional traceback string.
        """
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message
        self.traceback = traceback


@dataclass(slots=True)
class _RpcProcess:
    """Holds subprocess handles for an RPC env."""

    proc: subprocess.Popen[str]
    stdin: IO[str]
    stdout: IO[str]
    stderr: IO[str]


class _JsonlRpcClient:
    def __init__(self, *, proc: _RpcProcess, name: str = "rpc") -> None:
        """Initialize a JSONL RPC client.

        Args:
            proc: RPC process handles.
            name: Name used in error messages.
        """
        self._proc = proc
        self._name = name
        self._next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSONL RPC request and wait for the response.

        Args:
            method: RPC method name.
            params: Optional parameter mapping.

        Returns:
            RPC result payload.

        Raises:
            EOFError: If the RPC process terminates unexpectedly.
            RuntimeError: If the response id mismatches.
            RemoteEnvError: If the remote env returns an error payload.
        """
        params = params or {}
        msg_id = self._next_id
        self._next_id += 1

        msg = {"id": msg_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.proc.poll()
                if rc is not None:
                    stderr = self._proc.stderr.read().strip()
                    if stderr:
                        if len(stderr) > 4000:
                            stderr = "…" + stderr[-4000:]
                        raise EOFError(
                            f"{self._name} server closed stdout (method={method!r}, rc={rc}). stderr:\n{stderr}"
                        )
                    raise EOFError(f"{self._name} server closed stdout (method={method!r}, rc={rc})")
                raise EOFError(f"{self._name} server closed stdout (method={method!r})")
            line = line.strip()
            if not line:
                continue
            resp = json.loads(line)
            break
        if resp.get("id") != msg_id:
            raise RuntimeError(f"{self._name} response id mismatch: expected {msg_id}, got {resp.get('id')}")
        if not resp.get("ok", False):
            err = resp.get("error") or {}
            raise RemoteEnvError(
                error_type=str(err.get("type", "RemoteError")),
                message=str(err.get("message", "")),
                traceback=err.get("traceback"),
            )
        return resp.get("result")


def _wire_to_requests(wire: list[dict[str, Any]]) -> list[Request]:
    """Convert wire format dicts to Request objects.

    Args:
        wire: Wire payload list from the RPC server.

    Returns:
        Parsed request objects.
    """
    requests: list[Request] = []
    for item in wire:
        agent_id = str(item["agent_id"])
        raw_message = item.get("message") or {}
        message: Message = {
            "role": str(raw_message.get("role", "user")),
            "content": str(raw_message.get("content", "")),
        }
        info: dict[str, object] = dict(item.get("info") or {})
        request: Request = {
            "actor": make_actor_id(agent_id),
            "reward": float(item["reward"]),
            "message": message,
            "needs_action": bool(item["needs_action"]),
            "info": info,
            "episode_id": int(item.get("episode_id", 0)),
            "group_id": item.get("group_id"),
            "episode_start": bool(item.get("episode_start", False)),
            "episode_end": bool(item.get("episode_end", False)),
        }
        if item.get("system_message") is not None:
            raw_system = item["system_message"]
            request["system_message"] = {
                "role": str(raw_system.get("role", "system")),
                "content": str(raw_system.get("content", "")),
            }
        requests.append(request)
    return requests


class RpcEnvClient(LLMEnv):
    """
    Client for an env hosted out-of-process via `gyllm.rpc_server`.

    The hosted env is always treated as a single instance (env_id=0).
    Use `gyllm.batch.BatchedEnv` to combine multiple clients.
    """

    def __init__(self, rpc_process: _RpcProcess, *, name: str = "rpc-env") -> None:
        """Initialize an RPC env client.

        Args:
            rpc_process: RPC subprocess handles.
            name: Client name for error messages.
        """
        super().__init__()
        self._proc = rpc_process
        self._rpc = _JsonlRpcClient(proc=rpc_process, name=name)
        spec = self._rpc.call("spec")
        agent_ids = [str(a) for a in spec.get("agent_ids", [])]
        self.agents = agent_ids

    @property
    def actors(self) -> list[ActorId]:
        """Return actor ids for all remote agent ids.

        Returns:
            List of actor ids.
        """
        return [make_actor_id(agent_id) for agent_id in self.agent_ids]

    def _system_message(self, actor: ActorId) -> Message:
        """Raise because system messages are delivered via Request on reset."""
        del actor
        raise RuntimeError("RpcEnvClient does not expose system messages; read Request['system_message'] instead.")

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the remote env.

        Args:
            options: Optional reset options.

        Returns:
            Requests produced by reset.
        """
        wire = self._rpc.call("reset", {"options": options})
        return _wire_to_requests(wire)

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Step the remote env with actions.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced by the step.
        """
        actions_by_agent_id: dict[str, str] = {}
        for actor_id, completion in actions.items():
            agent_id = self.agent_id(actor_id)
            actions_by_agent_id[agent_id] = completion
        wire = self._rpc.call("step", {"actions": actions_by_agent_id})
        return _wire_to_requests(wire)

    def close(self) -> None:
        """Close the RPC connection and subprocess."""
        self._rpc.call("close")
        self._proc.stdin.close()
        self._proc.stdout.close()
        self._proc.stderr.close()
        self._proc.proc.wait(timeout=2.0)

    def __enter__(self) -> "RpcEnvClient":
        """Enter the context manager.

        Returns:
            Self for context management.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the context manager and close the client.

        Args:
            exc_type: Exception type if raised.
            exc: Exception instance if raised.
            tb: Traceback if raised.
        """
        self.close()


def _start_subprocess_rpc(
    *,
    env: str,
    env_kwargs: dict[str, Any] | None = None,
    python_executable: str | None = None,
    pythonpath: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> _RpcProcess:
    """Start a subprocess-based RPC server.

    Args:
        env: Env entrypoint string.
        env_kwargs: Optional env constructor kwargs.
        python_executable: Optional python executable path.
        pythonpath: Optional PYTHONPATH override.
        extra_env: Extra environment variables.

    Returns:
        RPC process handles.
    """
    env_kwargs = env_kwargs or {}
    python_executable = python_executable or sys.executable

    cmd = [
        python_executable,
        "-m",
        "gyllm.rpc_server",
        "--env",
        env,
        "--env-kwargs-json",
        json.dumps(env_kwargs),
    ]

    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    if pythonpath is not None:
        child_env["PYTHONPATH"] = pythonpath + (
            os.pathsep + child_env["PYTHONPATH"] if "PYTHONPATH" in child_env else ""
        )
    if extra_env:
        child_env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    return _RpcProcess(proc=proc, stdin=proc.stdin, stdout=proc.stdout, stderr=proc.stderr)


class SubprocessEnv(RpcEnvClient):
    def __init__(
        self,
        *,
        env: str,
        env_kwargs: dict[str, Any] | None = None,
        python_executable: str | None = None,
        pythonpath: str | None = None,
        extra_env: dict[str, str] | None = None,
        name: str = "subprocess-env",
    ) -> None:
        """Initialize a subprocess-backed RPC env.

        Args:
            env: Env entrypoint string.
            env_kwargs: Optional env constructor kwargs.
            python_executable: Optional python executable path.
            pythonpath: Optional PYTHONPATH override.
            extra_env: Extra environment variables.
            name: Client name for error messages.
        """
        super().__init__(
            _start_subprocess_rpc(
                env=env,
                env_kwargs=env_kwargs,
                python_executable=python_executable,
                pythonpath=pythonpath,
                extra_env=extra_env,
            ),
            name=name,
        )


def subprocess_env(
    *,
    env: str,
    env_kwargs: dict[str, Any] | None = None,
    python_executable: str | None = None,
    pythonpath: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> SubprocessEnv:
    """Construct a subprocess-backed RPC env.

    Args:
        env: Env entrypoint string.
        env_kwargs: Optional env constructor kwargs.
        python_executable: Optional python executable path.
        pythonpath: Optional PYTHONPATH override.
        extra_env: Extra environment variables.

    Returns:
        Subprocess-backed env client.
    """
    return SubprocessEnv(
        env=env,
        env_kwargs=env_kwargs,
        python_executable=python_executable,
        pythonpath=pythonpath,
        extra_env=extra_env,
    )


def _start_docker_rpc(
    *,
    image: str,
    env: str,
    env_kwargs: dict[str, Any] | None = None,
    docker_executable: str = "docker",
    docker_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> _RpcProcess:
    """Start a docker-based RPC server.

    Args:
        image: Docker image name.
        env: Env entrypoint string.
        env_kwargs: Optional env constructor kwargs.
        docker_executable: Docker binary to use.
        docker_args: Additional docker arguments.
        extra_env: Extra environment variables.

    Returns:
        RPC process handles.
    """
    env_kwargs = env_kwargs or {}
    docker_args = docker_args or []

    has_entrypoint = any(arg == "--entrypoint" or arg.startswith("--entrypoint=") for arg in docker_args)
    if not has_entrypoint:
        docker_args = ["--entrypoint=python", *docker_args]

    cmd = [
        docker_executable,
        "run",
        "--rm",
        "-i",
        *docker_args,
        image,
        "-m",
        "gyllm.rpc_server",
        "--env",
        env,
        "--env-kwargs-json",
        json.dumps(env_kwargs),
    ]

    child_env = os.environ.copy()
    if extra_env:
        child_env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    return _RpcProcess(proc=proc, stdin=proc.stdin, stdout=proc.stdout, stderr=proc.stderr)


class DockerEnv(RpcEnvClient):
    def __init__(
        self,
        *,
        image: str,
        env: str,
        env_kwargs: dict[str, Any] | None = None,
        docker_executable: str = "docker",
        docker_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        name: str = "docker-env",
    ) -> None:
        """Initialize a docker-backed RPC env.

        Args:
            image: Docker image name.
            env: Env entrypoint string.
            env_kwargs: Optional env constructor kwargs.
            docker_executable: Docker binary to use.
            docker_args: Additional docker arguments.
            extra_env: Extra environment variables.
            name: Client name for error messages.
        """
        super().__init__(
            _start_docker_rpc(
                image=image,
                env=env,
                env_kwargs=env_kwargs,
                docker_executable=docker_executable,
                docker_args=docker_args,
                extra_env=extra_env,
            ),
            name=name,
        )


def docker_env(
    *,
    image: str,
    env: str,
    env_kwargs: dict[str, Any] | None = None,
    docker_executable: str = "docker",
    docker_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> DockerEnv:
    """Construct a docker-backed RPC env.

    Args:
        image: Docker image name.
        env: Env entrypoint string.
        env_kwargs: Optional env constructor kwargs.
        docker_executable: Docker binary to use.
        docker_args: Additional docker arguments.
        extra_env: Extra environment variables.

    Returns:
        Docker-backed env client.
    """
    return DockerEnv(
        image=image,
        env=env,
        env_kwargs=env_kwargs,
        docker_executable=docker_executable,
        docker_args=docker_args,
        extra_env=extra_env,
    )
