from typing import Any

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


class TextArenaEnv(LLMEnv):
    """
    Wrapper for TextArena games (inspired by OpenEnv's textarena_env).

    Requires the optional `textarena` dependency.
    """

    def __init__(
        self,
        *,
        env_id: str = "Wordle-v0",
        num_players: int = 1,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the TextArena environment wrapper.

        Args:
            env_id: TextArena environment id.
            num_players: Number of players.
            env_kwargs: Optional kwargs passed to TextArena.

        Raises:
            ImportError: If textarena is unavailable.
            ValueError: If num_players is invalid.
        """
        super().__init__()
        try:
            import textarena as ta  # type: ignore[import]
        except Exception as exc:  # pragma: no cover
            raise ImportError("TextArenaEnv requires `textarena` (optional dependency).") from exc

        if num_players <= 0:
            raise ValueError("num_players must be > 0")
        self.env_id = env_id
        self.num_players = num_players
        self.agents = [f"player_{i}" for i in range(num_players)]
        self._ta = ta
        self._env_kwargs = env_kwargs or {}
        self._env: Any = self._ta.make(env_id=env_id, **self._env_kwargs)
        self._current_player: int = 0
        self._done = False

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the actor.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id not in set(self.agent_ids):
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                f"You are playing a TextArena game: {self.env_id}.\n"
                "When it is your turn, respond with your move as plain text.\n"
                'You may also respond with JSON like `{"message": "..."}`.\n'
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
        self._env.reset(num_players=self.num_players)
        self._done = False
        self._current_player = 0
        player_id, messages = self._env.get_observation()
        self._current_player = int(player_id)
        content = "\n".join(str(m) for m in messages)
        prefix = options.get("prompt_prefix")
        if prefix:
            content = f"{prefix}\n{content}"
        requests: list[Request] = [
            {
                "actor": make_actor_id(f"player_{self._current_player}"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id(f"player_{self._current_player}")),
                "message": {"role": "user", "content": content},
                "needs_action": True,
                "info": {"player_id": self._current_player},
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
        if self._done:
            return []

        actions = self._normalize_actions(actions)
        actor = f"player_{self._current_player}"
        if actor not in actions:
            raise ActionError(f"Missing action for {actor!r}")
        raw = actions[actor].strip()
        parsed = maybe_parse_json(raw)
        move = str(parsed["message"]) if isinstance(parsed, dict) and "message" in parsed else raw

        done, _info = self._env.step(move)
        self._done = bool(done)
        player_id, messages = self._env.get_observation()
        self._current_player = int(player_id)
        content = "\n".join(str(m) for m in messages)
        requests: list[Request] = [
            {
                "actor": make_actor_id(f"player_{self._current_player}"),
                "reward": 0.0,
                "message": {"role": "user", "content": content},
                "needs_action": not self._done,
                "info": {"player_id": self._current_player, "done": self._done},
            }
        ]
        done = bool(self._done)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
