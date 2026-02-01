from gyllm.core import ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


class EchoEnv(LLMEnv):
    """
    A minimal echo environment (inspired by OpenEnv's EchoEnvironment).

    Single-agent loop:
    - Env asks for any message.
    - Agent responds with any text.
    - Env echoes it back and rewards proportionally to message length.
    """

    agents: list[str] = ["agent"]

    def __init__(self) -> None:
        """Initialize the echo environment."""
        super().__init__()
        self._step = 0
        self._reward_scale = 0.1

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
                "You are in an Echo environment.\n"
                "On each turn, send any text as your action.\n"
                "The environment will echo it back.\n"
                "There is no special action format required.\n"
                'Optional: you may also send JSON like `{"message": "hello"}`.'
            ),
        }

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request for the new episode.
        """
        options = options or {}
        self._begin_episode()
        self._step = 0
        if "reward_scale" in options:
            self._reward_scale = float(options["reward_scale"])
        prompt = options.get("prompt")
        if prompt is None:
            prompt = "Echo environment ready. Send a message."
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": str(prompt)},
                "needs_action": True,
                "info": {},
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply an action and return the echo response.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after the step.
        """
        actions = self._normalize_actions(actions)
        self._step += 1
        raw = actions["agent"].strip()
        parsed = maybe_parse_json(raw)
        text = str(parsed["message"]) if isinstance(parsed, dict) and "message" in parsed else raw
        reward = self._reward_scale * len(text)
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": float(reward),
                "message": {"role": "user", "content": text},
                "needs_action": True,
                "info": {"step": self._step},
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
