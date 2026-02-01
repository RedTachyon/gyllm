import json
from typing import Any

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


class FinRLEnv(LLMEnv):
    """
    FinRL-style trading environment wrapper (inspired by OpenEnv's finrl_env).

    This wrapper does not depend on FinRL directly; instead you pass:
    - `finrl_env_class`: a class with `reset()` and `step(action)` methods
    - `finrl_env_config`: kwargs passed to the env constructor

    Expected API (matching FinRL's StockTradingEnv):
    - reset() -> (state, info)
    - step(action) -> (state, reward, terminal, truncated, info)

    Action format:
    - JSON list of numbers (e.g. `[0.1, -0.2]`), or
    - JSON object `{"actions": [...numbers...]}` (matches OpenEnv's FinRLAction)
    """

    agents: list[str] = ["trader"]

    def __init__(self, *, finrl_env_class: Any, finrl_env_config: dict[str, Any]) -> None:
        """Initialize the FinRL environment wrapper.

        Args:
            finrl_env_class: Env class implementing FinRL-style API.
            finrl_env_config: Env constructor kwargs.
        """
        super().__init__()
        self.finrl_env_class = finrl_env_class
        self.finrl_env_config = finrl_env_config
        self._env: Any | None = None
        self._step = 0
        self._last_reward = 0.0
        self._done = False

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the trader.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "trader":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                "You are trading in a FinRL-style stock trading environment.\n"
                "Each step, output trading actions as JSON, e.g.:\n"
                "  [0.1, -0.2]\n"
                "or:\n"
                '  {"actions": [0.1, -0.2]}\n'
                "The environment will return reward and whether the episode is done."
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
        self._begin_episode()
        env_config = dict(self.finrl_env_config)
        override = options.get("env_config")
        if override is not None:
            if not isinstance(override, dict):
                raise TypeError("env_config must be a dict")
            env_config.update(override)
        self._env = self.finrl_env_class(**env_config)
        self._step = 0
        self._last_reward = 0.0
        self._done = False

        state, _info = self._env.reset()
        prompt = options.get("prompt")
        msg: Message = {
            "role": "user",
            "content": f"{prompt}\nReset complete. Initial state:\n{state}"
            if prompt
            else f"Reset complete. Initial state:\n{state}",
        }
        request: Request = {
            "actor": make_actor_id("trader"),
            "reward": 0.0,
            "system_message": self._system_message(make_actor_id("trader")),
            "message": msg,
            "needs_action": True,
            "info": {},
        }
        requests: list[Request] = [request]
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
        if self._env is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        if self._done:
            return []

        actions = self._normalize_actions(actions)
        raw = actions["trader"].strip()
        parsed = maybe_parse_json(raw)
        if isinstance(parsed, dict) and "actions" in parsed:
            parsed = parsed["actions"]
        if parsed is None:
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                raise ActionError("Action must be JSON (list or {'actions': [...] }).") from exc
        if not isinstance(parsed, list) or not all(isinstance(x, (int, float)) for x in parsed):
            raise ActionError("Action must be a JSON list of numbers (or {'actions': [...] }).")

        state, reward, terminal, truncated, info = self._env.step(parsed)
        self._step += 1
        self._last_reward = float(reward)
        self._done = bool(terminal or truncated)

        content = f"Step {self._step} reward={self._last_reward} done={self._done}\nState:\n{state}"
        if info:
            content += f"\nInfo:\n{info}"
        msg: Message = {"role": "user", "content": content}
        request: Request = {
            "actor": make_actor_id("trader"),
            "reward": self._last_reward,
            "message": msg,
            "needs_action": not self._done,
            "info": dict(info or {}),
        }
        requests: list[Request] = [request]
        done = bool(self._done)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
