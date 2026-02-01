from gyllm.core import ActorId, LLMEnv, Message, Request, make_actor_id


class ChatEnv(LLMEnv):
    """
    Minimal chat environment (inspired by OpenEnv's chat_env).

    This is intentionally lightweight: it provides a conversational loop with
    no external tools. The environment responds with an acknowledgement and
    does not attempt to simulate another agent.
    """

    agents: list[str] = ["assistant"]

    def __init__(self, *, system_prompt: str | None = None, max_turns: int = 8) -> None:
        """Initialize the chat environment.

        Args:
            system_prompt: Optional system prompt override.
            max_turns: Maximum turns before termination.
        """
        super().__init__()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_turns = max_turns
        self._turn = 0

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the assistant.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "assistant":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {"role": "system", "content": self.system_prompt}

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return initial request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "system_prompt" in options:
            self.system_prompt = str(options["system_prompt"])
        if "max_turns" in options:
            max_turns = int(options["max_turns"])
            if max_turns <= 0:
                raise ValueError(f"max_turns must be > 0; got {max_turns}")
            self.max_turns = max_turns
        prompt = options.get("start_message", "Start a conversation.")
        self._begin_episode()
        self._turn = 0
        requests: list[Request] = [
            {
                "actor": make_actor_id("assistant"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("assistant")),
                "message": {"role": "user", "content": str(prompt)},
                "needs_action": True,
                "info": {"turn": self._turn, "max_turns": self.max_turns},
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
        self._turn += 1
        done = self._turn >= self.max_turns
        content = "Acknowledged." if not done else "Conversation ended."
        requests: list[Request] = [
            {
                "actor": make_actor_id("assistant"),
                "reward": 0.0,
                "message": {"role": "user", "content": content},
                "needs_action": not done,
                "info": {"turn": self._turn, "max_turns": self.max_turns, "done": done},
            }
        ]
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
