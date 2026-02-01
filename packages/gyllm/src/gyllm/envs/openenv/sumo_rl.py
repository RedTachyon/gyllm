import os
from typing import Any

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json, parse_int


class SumoRLEnv(LLMEnv):
    """
    SUMO-RL environment wrapper (inspired by OpenEnv's sumo_rl_env).

    Requires optional SUMO + sumo-rl dependencies and system binaries.
    """

    agents: list[str] = ["agent"]

    def __init__(
        self,
        *,
        net_file: str,
        route_file: str,
        num_seconds: int = 20000,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        reward_fn: str = "diff-waiting-time",
        sumo_seed: int = 42,
    ) -> None:
        """Initialize the SUMO-RL environment wrapper.

        Args:
            net_file: SUMO network file path.
            route_file: SUMO route file path.
            num_seconds: Episode duration in seconds.
            delta_time: Simulation delta time.
            yellow_time: Yellow light duration.
            min_green: Minimum green light duration.
            max_green: Maximum green light duration.
            reward_fn: Reward function name.
            sumo_seed: SUMO RNG seed.

        Raises:
            ImportError: If SUMO-RL dependencies are missing.
        """
        super().__init__()
        os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
        try:
            from sumo_rl import SumoEnvironment as BaseSumoEnv  # type: ignore[import]
        except Exception as exc:  # pragma: no cover
            raise ImportError("SumoRLEnv requires the optional `sumo-rl` dependency and SUMO binaries.") from exc

        self.net_file = net_file
        self.route_file = route_file
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed

        self._env = BaseSumoEnv(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            single_agent=True,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            reward_fn=reward_fn,
            sumo_seed=sumo_seed,
            sumo_warnings=False,
            out_csv_name=None,
            add_system_info=True,
            add_per_agent_info=False,
        )
        self._step = 0
        self._last_info: dict[str, Any] = {}

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
                "You are controlling a traffic signal in a SUMO-RL environment.\n"
                "On each turn, choose the next traffic light phase by its integer ID.\n"
                "Respond with either:\n"
                "- an integer phase_id (e.g. `0`), or\n"
                '- a JSON object like `{"phase_id": 0, "ts_id": "0"}`.'
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
        obs, info = self._env.reset()
        self._step = 0
        self._last_info = dict(info or {})
        action_n = int(self._env.action_space.n)
        prompt = options.get(
            "prompt_prefix",
            "",
        )
        content = (
            f"{(str(prompt) + '\n' if prompt else '')}"
            f"Reset complete. observation_dim={len(getattr(obs, 'tolist', lambda: list(obs))()) if obs is not None else 0}\n"
            f"num_phases={action_n}\n"
            f"system_info_keys={[k for k in self._last_info if str(k).startswith('system_')]}"
        )
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": content},
                "needs_action": True,
                "info": {"action_count": action_n},
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
        if isinstance(parsed, dict) and "phase_id" in parsed:
            raw = str(parsed["phase_id"])
        try:
            phase_id = parse_int(raw)
        except Exception as exc:
            raise ActionError("Phase id must be an integer phase_id (or JSON like {'phase_id': 0}).") from exc

        _obs, reward, terminated, truncated, info = self._env.step(int(phase_id))
        self._step += 1
        self._last_info = dict(info or {})
        done = bool(terminated or truncated)

        action_n = int(self._env.action_space.n)
        if phase_id < 0 or phase_id >= action_n:
            raise ActionError(f"Invalid phase_id={phase_id}. Valid range: 0..{action_n - 1}")

        sys_info = {k: v for k, v in self._last_info.items() if str(k).startswith("system_")}
        content = f"Step {self._step} phase_id={phase_id} reward={float(reward)} done={done}\nsystem_info={sys_info}"
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": float(reward),
                "message": {"role": "user", "content": content},
                "needs_action": not done,
                "info": {
                    "phase_id": phase_id,
                    "done": done,
                    "system_info": sys_info,
                },
            }
        ]
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
