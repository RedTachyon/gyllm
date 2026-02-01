import random
from typing import Any

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json, parse_int


class OpenSpielEnv(LLMEnv):
    """
    OpenSpiel environment wrapper (inspired by OpenEnv's openspiel_env).

    This is a single-agent interface: the agent controls `agent_player` and
    all other players use a fixed policy (currently: random).

    Requires optional `open_spiel` dependencies.
    """

    agents: list[str] = ["player"]

    def __init__(
        self,
        *,
        game_name: str = "catch",
        agent_player: int = 0,
        opponent_policy: str = "random",
        game_params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the OpenSpiel environment wrapper.

        Args:
            game_name: OpenSpiel game name.
            agent_player: Player index controlled by the agent.
            opponent_policy: Opponent policy name (currently "random").
            game_params: Optional game parameters.
            seed: Optional RNG seed.

        Raises:
            ImportError: If open_spiel is unavailable.
            ValueError: If configuration values are invalid.
        """
        super().__init__()
        try:
            from open_spiel.python import rl_environment  # type: ignore[import]
        except Exception as exc:  # pragma: no cover
            raise ImportError("OpenSpielEnv requires the optional `open_spiel` dependency.") from exc

        self.game_name = game_name
        self.agent_player = agent_player
        self.opponent_policy = opponent_policy
        self.game_params = game_params or {}
        self._rng = random.Random(seed)

        try:
            self._env = rl_environment.Environment(game_name, **self.game_params)
        except Exception as exc:
            raise ValueError(f"Failed to create OpenSpiel game {game_name!r}: {exc}") from exc

        self.num_players: int = int(self._env.num_players)
        self.is_turn_based: bool = bool(self._env.is_turn_based)
        if self.agent_player < 0 or self.agent_player >= self.num_players:
            raise ValueError(f"agent_player must be in [0, {self.num_players - 1}]")
        if opponent_policy != "random":
            raise ValueError("Only opponent_policy='random' is supported currently.")

        self._last_opponent_action: int | None = None
        self._time_step: Any | None = None

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the player.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "player":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                f"You are playing an OpenSpiel game: {self.game_name}.\n"
                "On your turn, pick an action_id from the provided legal actions.\n"
                "Respond with either:\n"
                "- an integer action_id (e.g. `2`), or\n"
                '- a JSON object like `{"action_id": 2}`.'
            ),
        }

    def _format_obs(self, time_step: Any, *, reward: float) -> str:
        """Format an observation into a text prompt.

        Args:
            time_step: OpenSpiel time step object.
            reward: Step reward.

        Returns:
            Formatted observation string.
        """
        obs = time_step.observations
        info_state = obs.get("info_state", [])[self.agent_player]
        legal_actions = obs.get("legal_actions", [])[self.agent_player]
        current_player = obs.get("current_player")
        done = bool(time_step.last())
        lines = [
            f"game={self.game_name} step_reward={reward} done={done}",
            f"current_player={current_player} agent_player={self.agent_player}",
        ]
        if self._last_opponent_action is not None:
            lines.append(f"opponent_last_action={self._last_opponent_action}")
        lines.append(f"legal_actions={legal_actions}")
        lines.append(f"info_state={list(info_state) if hasattr(info_state, '__iter__') else info_state}")
        return "\n".join(lines)

    def _auto_play_opponents(self, time_step: Any) -> Any:
        """Auto-play opponents until it's the agent's turn.

        Args:
            time_step: Current OpenSpiel time step.

        Returns:
            Updated time step after opponent actions.
        """
        if self.num_players == 1:
            return time_step
        while (not time_step.last()) and (time_step.observations["current_player"] != self.agent_player):
            current_player = int(time_step.observations["current_player"])
            legal = time_step.observations["legal_actions"][current_player]
            if not legal:
                break
            action = self._rng.choice(list(legal))
            self._last_opponent_action = int(action)
            time_step = self._env.step([int(action)])
        return time_step

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return initial request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "seed" in options:
            self._rng = random.Random(int(options["seed"]))
        self._begin_episode()
        time_step = self._env.reset()
        self._last_opponent_action = None
        time_step = self._auto_play_opponents(time_step)
        self._time_step = time_step
        content = self._format_obs(time_step, reward=0.0)
        prefix = options.get("prompt_prefix")
        if prefix:
            content = f"{prefix}\n{content}"
        msg: Message = {"role": "user", "content": content}
        legal = time_step.observations["legal_actions"][self.agent_player]
        request: Request = {
            "actor": make_actor_id("player"),
            "reward": 0.0,
            "system_message": self._system_message(make_actor_id("player")),
            "message": msg,
            "needs_action": not time_step.last(),
            "info": {"legal_actions": legal},
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
        if self._time_step is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        if self._time_step.last():
            return []

        actions = self._normalize_actions(actions)
        raw = actions["player"].strip()
        parsed = maybe_parse_json(raw)
        if isinstance(parsed, dict) and "action_id" in parsed:
            raw = str(parsed["action_id"])
        try:
            action_id = parse_int(raw)
        except Exception as exc:
            raise ActionError("Action must be an integer action_id (or JSON like {'action_id': 2}).") from exc

        legal = self._time_step.observations["legal_actions"][self.agent_player]
        if legal and action_id not in legal:
            raise ActionError(f"Illegal action_id={action_id}. Legal actions: {legal}")

        if not self.is_turn_based:
            raise NotImplementedError("Simultaneous-move OpenSpiel games are not supported yet.")

        time_step = self._env.step([int(action_id)])
        time_step = self._auto_play_opponents(time_step)
        self._time_step = time_step

        reward = 0.0
        if time_step.rewards is not None:
            reward = float(time_step.rewards[self.agent_player])
        msg: Message = {"role": "user", "content": self._format_obs(time_step, reward=reward)}
        legal = time_step.observations["legal_actions"][self.agent_player]
        request: Request = {
            "actor": make_actor_id("player"),
            "reward": reward,
            "message": msg,
            "needs_action": not time_step.last(),
            "info": {"legal_actions": legal},
        }
        requests: list[Request] = [request]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
