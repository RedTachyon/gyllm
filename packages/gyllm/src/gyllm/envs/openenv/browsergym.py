import importlib
from typing import Any

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


class BrowserGymEnv(LLMEnv):
    """
    BrowserGym environment wrapper (inspired by OpenEnv's browsergym_env).

    Requires optional `browsergym` dependencies and a browser runtime, which are not
    included in this repo by default.
    """

    agents: list[str] = ["agent"]

    def __init__(
        self,
        *,
        benchmark: str = "miniwob",
        task_name: str | None = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout_ms: float = 10000.0,
        max_obs_chars: int = 4000,
        **gym_kwargs: Any,
    ) -> None:
        """Initialize the BrowserGym environment wrapper.

        Args:
            benchmark: BrowserGym benchmark name.
            task_name: Optional task name.
            headless: Whether to run the browser headless.
            viewport_width: Browser viewport width.
            viewport_height: Browser viewport height.
            timeout_ms: Task timeout in milliseconds.
            max_obs_chars: Maximum observation text length.
            **gym_kwargs: Additional kwargs passed to gym.make.

        Raises:
            ImportError: If browsergym dependencies are missing.
            RuntimeError: If the environment cannot be created.
        """
        super().__init__()
        try:
            import gymnasium as gym  # type: ignore[import]
        except Exception as exc:  # pragma: no cover
            raise ImportError("BrowserGymEnv requires the optional `browsergym` dependency.") from exc

        self._gym = gym
        self.benchmark = benchmark
        self.task_name = task_name
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout_ms = timeout_ms
        self.max_obs_chars = max_obs_chars
        self.gym_kwargs = dict(gym_kwargs)

        # Force import the benchmark module (mirrors OpenEnv behavior).
        benchmark_modules = {
            "miniwob": "browsergym.miniwob",
            "webarena": "browsergym.webarena",
            "visualwebarena": "browsergym.visualwebarena",
            "workarena": "browsergym.workarena",
        }
        module_path = benchmark_modules.get(benchmark)
        try:
            importlib.import_module(module_path or "browsergym")
        except ModuleNotFoundError as import_error:  # pragma: no cover
            raise ImportError(
                f"Failed to import BrowserGym benchmark {benchmark!r}. "
                "Install the matching package (e.g. `browsergym-miniwob`)."
            ) from import_error

        if task_name:
            self.env_id = f"browsergym/{benchmark}.{task_name}"
        else:
            self.env_id = f"browsergym/{benchmark}"

        try:
            self._env = self._gym.make(
                self.env_id,
                headless=headless,
                viewport={"width": viewport_width, "height": viewport_height},
                timeout=timeout_ms,
                **self.gym_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create BrowserGym env {self.env_id!r}: {exc}") from exc

        self._step = 0
        self._last_obs: Any | None = None
        self._last_info: dict[str, Any] | None = None

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
                f"You are controlling a web browser task ({self.env_id}).\n"
                "On each turn, output a BrowserGym action string.\n"
                "Respond with either:\n"
                "- a plain action string (e.g. `click('Submit')`), or\n"
                '- a JSON object like `{"action_str": "click(\\"Submit\\")"}`.\n'
            ),
        }

    def _obs_text(self, obs: Any, info: dict[str, Any] | None) -> str:
        """Build a text summary of the observation.

        Args:
            obs: Observation payload.
            info: Optional info dict from the env.

        Returns:
            Rendered observation text.
        """
        if isinstance(obs, str):
            text = obs
        elif isinstance(obs, dict):
            text = obs.get("axtree_txt") or obs.get("pruned_html") or obs.get("dom_txt") or obs.get("text") or ""
        else:
            text = str(obs)

        if len(text) > self.max_obs_chars:
            text = text[: self.max_obs_chars] + "\n...<truncated>..."

        goal = ""
        url = ""
        if info:
            goal = str(info.get("goal", "") or "")
            url = str(info.get("url", "") or "")

        parts = [f"url={url}", f"goal={goal}", "", text]
        return "\n".join(p for p in parts if p is not None).strip()

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return the initial request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "max_obs_chars" in options:
            max_obs_chars = int(options["max_obs_chars"])
            if max_obs_chars <= 0:
                raise ValueError(f"max_obs_chars must be > 0; got {max_obs_chars}")
            self.max_obs_chars = max_obs_chars
        self._begin_episode()
        obs, info = self._env.reset()
        self._step = 0
        self._last_obs = obs
        self._last_info = dict(info or {})
        content = self._obs_text(obs, self._last_info)
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": content},
                "needs_action": True,
                "info": dict(self._last_info),
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply an action and return the next request.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after the step.
        """
        actions = self._normalize_actions(actions)
        raw = actions["agent"].strip()
        parsed = maybe_parse_json(raw)
        action_str = str(parsed["action_str"]) if isinstance(parsed, dict) and "action_str" in parsed else raw
        if not action_str:
            raise ActionError("Empty BrowserGym action string.")

        try:
            obs, reward, terminated, truncated, info = self._env.step(action_str)
        except Exception as exc:  # pragma: no cover
            obs = self._last_obs
            reward = 0.0
            terminated = False
            truncated = False
            info = {"error": str(exc)}

        self._step += 1
        self._last_obs = obs
        self._last_info = dict(info or {})
        done = bool(terminated or truncated)

        content = f"step={self._step} reward={float(reward)} done={done}\n\n{self._obs_text(obs, self._last_info)}"
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": float(reward),
                "message": {"role": "user", "content": content},
                "needs_action": not done,
                "info": {
                    "step": self._step,
                    "done": done,
                    **dict(self._last_info),
                },
            }
        ]
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
