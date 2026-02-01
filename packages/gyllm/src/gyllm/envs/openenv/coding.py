import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


def _parse_code(text: str) -> str:
    """Parse a code snippet from model output.

    Args:
        text: Raw model completion text.

    Returns:
        Parsed code string or "DONE" sentinel.
    """
    raw = text.strip()
    parsed = maybe_parse_json(raw)
    if isinstance(parsed, dict) and "code" in parsed:
        raw = str(parsed["code"])
    code = raw
    if code.strip().lower() in {"quit", "exit", "done"}:
        return "DONE"
    return code


def _validate_imports(code: str, allowed_imports: set[str] | None) -> None:
    """Validate imports against an allowed set.

    Args:
        code: Code snippet to inspect.
        allowed_imports: Allowed top-level import names, or None.

    Raises:
        ActionError: If disallowed imports are used.
    """
    if allowed_imports is None:
        return
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_imports:
                    raise ActionError(f"Import not allowed: {alias.name!r}")
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ActionError("Relative imports are not allowed.")
            if node.module.split(".")[0] not in allowed_imports:
                raise ActionError(f"Import not allowed: {node.module!r}")


@dataclass(slots=True)
class ExecResult:
    """Holds execution results from running a code snippet."""

    stdout: str
    stderr: str
    ok: bool
    error: str | None = None


class PythonCodeEnv(LLMEnv):
    """
    Interactive Python execution environment (inspired by OpenEnv's coding_env).

    Notes:
    - In-memory mode executes code via `exec()` in a shared namespace.
    - For untrusted code, prefer running this env out-of-process or in a container.
    """

    agents: list[str] = ["coder"]

    def __init__(self, *, allowed_imports: set[str] | None = None) -> None:
        """Initialize the Python code execution environment.

        Args:
            allowed_imports: Optional whitelist for imports.
        """
        super().__init__()
        self.allowed_imports = allowed_imports
        self._globals: dict[str, Any] = {}
        self._step = 0

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the coder.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "coder":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        allow = (
            "any imports" if self.allowed_imports is None else f"imports only from: {sorted(self.allowed_imports)!r}"
        )
        return {
            "role": "system",
            "content": (
                "You are in a Python code execution environment.\n"
                f"You may write Python code ({allow}).\n"
                "Send code as plain text (multi-line is fine), or as JSON like:\n"
                '  {"code": "print(\\"hello\\")"}\n'
                "To end the episode, send `DONE`.\n"
            ),
        }

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return initial request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "allowed_imports" in options:
            if options["allowed_imports"] is None:
                self.allowed_imports = None
            else:
                self.allowed_imports = {str(x) for x in options["allowed_imports"]}  # type: ignore[arg-type]
        welcome = options.get(
            "welcome_message",
            "Python executor ready. Send a snippet of Python code.\nExample: `print('hello')`",
        )
        self._begin_episode()
        self._globals = {}
        self._step = 0
        requests: list[Request] = [
            {
                "actor": make_actor_id("coder"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("coder")),
                "message": {
                    "role": "user",
                    "content": str(welcome),
                },
                "needs_action": True,
                "info": {"step": self._step},
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def _exec(self, code: str) -> ExecResult:
        """Execute code in the shared globals namespace.

        Args:
            code: Code snippet to execute.

        Returns:
            Execution result with stdout/stderr and status.
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            compiled = compile(code, "<python_code_env>", "exec")
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compiled, self._globals, self._globals)
            return ExecResult(stdout=stdout.getvalue(), stderr=stderr.getvalue(), ok=True)
        except BaseException as exc:
            tb = traceback.format_exc()
            return ExecResult(
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue() + tb,
                ok=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply a code action and return the next request.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after execution.
        """
        actions = self._normalize_actions(actions)
        self._step += 1
        code = _parse_code(actions["coder"])
        if code == "DONE":
            requests: list[Request] = [
                {
                    "actor": make_actor_id("coder"),
                    "reward": 0.0,
                    "message": {"role": "user", "content": "Done."},
                    "needs_action": False,
                    "info": {"step": self._step, "done": True},
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = done
            return requests

        _validate_imports(code, self.allowed_imports)
        result = self._exec(code)

        reward = 0.0 if result.ok else -1.0
        out = []
        out.append(f"Step {self._step} result:")
        out.append(f"ok={result.ok} reward={reward}")
        if result.stdout.strip():
            out.append("\n[stdout]\n" + result.stdout.rstrip())
        if result.stderr.strip():
            out.append("\n[stderr]\n" + result.stderr.rstrip())
        if result.error:
            out.append("\n[error]\n" + result.error)

        requests: list[Request] = [
            {
                "actor": make_actor_id("coder"),
                "reward": reward,
                "message": {"role": "user", "content": "\n".join(out).strip()},
                "needs_action": True,
                "info": {"step": self._step, "ok": result.ok},
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
