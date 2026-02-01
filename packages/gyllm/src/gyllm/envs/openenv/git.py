import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


@dataclass(slots=True)
class _Repo:
    tmp: tempfile.TemporaryDirectory[str]
    path: Path

    def cleanup(self) -> None:
        """Clean up the temporary repo directory."""
        self.tmp.cleanup()


class GitEnv(LLMEnv):
    """
    Local Git task environment (inspired by OpenEnv's git_env).

    This version is self-contained and does not require a Gitea server.
    It creates a temporary git repo per episode and lets the agent run git commands.
    """

    agents: list[str] = ["agent"]

    def __init__(self) -> None:
        """Initialize the Git environment."""
        super().__init__()
        self._repo: _Repo | None = None
        self._step = 0

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the agent.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "agent":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                "You are in a Git environment.\n"
                "The environment maintains a local git repository.\n"
                "On each turn, run a git command.\n"
                "You may reply with either:\n"
                "- a plain git command like `status` (you may omit the leading `git`), or\n"
                '- a JSON object like {"action_type": "execute_git_command", "command": "status", "working_dir": ""}.\n'
                "Other OpenEnv-style action types may be accepted on a best-effort basis."
            ),
        }

    def close(self) -> None:
        """Close the environment and clean up the repo."""
        if self._repo is not None:
            self._repo.cleanup()
            self._repo = None

    def _run_git(self, args: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
        """Run a git command in the repo.

        Args:
            args: Git command arguments (without the leading "git").
            cwd: Optional working directory.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        if self._repo is None:
            raise RuntimeError("Repo not initialized.")
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd or self._repo.path),
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _init_repo(self) -> None:
        """Initialize a new temporary git repository.

        Raises:
            RuntimeError: If git is unavailable.
        """
        if shutil.which("git") is None:
            raise RuntimeError("`git` executable not found on PATH.")

        self.close()
        tmp = tempfile.TemporaryDirectory(prefix="gyllm_git_env_")
        repo_path = Path(tmp.name)
        self._repo = _Repo(tmp=tmp, path=repo_path)

        subprocess.run(["git", "init"], cwd=str(repo_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "gyllm"], cwd=str(repo_path), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "gyllm@example.com"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
        )
        (repo_path / "README.md").write_text("# GitEnv\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(repo_path), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
        )

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return the initial request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        self._begin_episode()
        self._step = 0
        self._init_repo()
        assert self._repo is not None
        files = options.get("files")
        if files is not None:
            if not isinstance(files, dict):
                raise TypeError("files must be a dict of path -> content")
            if files:
                for relpath, content in files.items():
                    path = self._repo.path / str(relpath)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(content), encoding="utf-8")
                add_code, add_out, add_err = self._run_git(["add", "."])
                if add_code != 0:
                    raise RuntimeError(f"Failed to stage seed files: {add_err or add_out}")
                msg = str(options.get("seed_commit_message", "seed"))
                commit_code, commit_out, commit_err = self._run_git(["commit", "-m", msg])
                if commit_code != 0:
                    raise RuntimeError(f"Failed to commit seed files: {commit_err or commit_out}")
        prompt = options.get(
            "prompt",
            f"Git repo initialized at {self._repo.path}.\nRun a git command (e.g. `status`).",
        )
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {
                    "role": "user",
                    "content": str(prompt),
                },
                "needs_action": True,
                "info": {"repo_path": str(self._repo.path) if self._repo else ""},
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply a git command and return the next request.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after executing the command.
        """
        actions = self._normalize_actions(actions)
        self._step += 1
        raw = actions["agent"].strip()
        parsed = maybe_parse_json(raw)

        if isinstance(parsed, dict):
            action_type = str(parsed.get("action_type", "execute_git_command"))

            if action_type == "list_repos":
                assert self._repo is not None
                content = f"repos=[{{'name': 'local', 'path': '{self._repo.path}'}}]"
                requests: list[Request] = [
                    {
                        "actor": make_actor_id("agent"),
                        "reward": 0.0,
                        "message": {"role": "user", "content": content},
                        "needs_action": True,
                        "info": {"action_type": "list_repos"},
                    }
                ]
                done = not requests or not any(r["needs_action"] for r in requests)
                for request in requests:
                    request["episode_id"] = self._episode_id
                    request["episode_start"] = False
                    request["episode_end"] = done
                return requests

            if action_type == "clone_repo":
                repo_name = str(parsed.get("repo_name", "") or "")
                target_dir = str(parsed.get("target_dir", "") or "")
                content = (
                    "clone_repo is not supported in this in-memory GitEnv.\n"
                    f"repo_name={repo_name!r} target_dir={target_dir!r}"
                )
                requests: list[Request] = [
                    {
                        "actor": make_actor_id("agent"),
                        "reward": -1.0,
                        "message": {"role": "user", "content": content},
                        "needs_action": True,
                        "info": {"action_type": "clone_repo", "repo_name": repo_name, "target_dir": target_dir},
                    }
                ]
                done = not requests or not any(r["needs_action"] for r in requests)
                for request in requests:
                    request["episode_id"] = self._episode_id
                    request["episode_start"] = False
                    request["episode_end"] = done
                return requests

            if action_type != "execute_git_command":
                raise ActionError(
                    "Unsupported action_type. Expected one of: 'execute_git_command', 'list_repos', 'clone_repo'."
                )

            cmd = str(parsed.get("command", "") or "").strip()
            working_dir = str(parsed.get("working_dir", "") or "").strip()
        else:
            cmd = raw
            working_dir = ""

        parts = cmd.split()
        if not parts:
            raise ActionError("Empty git command.")
        if parts[0] == "git":
            parts = parts[1:]
        if not parts:
            raise ActionError("Empty git command after stripping leading 'git'.")

        assert self._repo is not None
        cwd = self._repo.path
        if working_dir:
            cwd = cwd / working_dir
            if not cwd.is_dir():
                raise ActionError(f"working_dir does not exist: {working_dir!r}")

        code, out, err = self._run_git(parts, cwd=cwd)
        reward = 0.0 if code == 0 else -1.0
        content = [f"Step {self._step}: git {' '.join(parts)}", f"exit_code={code} reward={reward}"]
        if out.strip():
            content.append("\n[stdout]\n" + out.rstrip())
        if err.strip():
            content.append("\n[stderr]\n" + err.rstrip())
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": reward,
                "message": {"role": "user", "content": "\n".join(content).strip()},
                "needs_action": True,
                "info": {"command": "git " + " ".join(parts), "exit_code": code},
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
