import random
import uuid
from typing import Literal

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id

MessageKind = Literal["word", "number", "uuid"]

_DEFAULT_WORDS: list[str] = [
    "apple",
    "banana",
    "cherry",
    "delta",
    "echo",
    "falcon",
    "garden",
    "harbor",
    "island",
    "jungle",
    "kitten",
    "lemon",
    "monkey",
    "nebula",
    "oasis",
    "panda",
    "quartz",
    "rocket",
    "sunset",
    "turtle",
]


class ReverseEcho(LLMEnv):
    """
    ReverseEcho environment: the env presents a message, the agent repeats it.
    """

    agents: list[str] = ["agent"]

    def __init__(
        self,
        *,
        num_turns: int = 5,
        message_kind: MessageKind = "word",
        seed: int | None = None,
    ) -> None:
        """Initialize the ReverseEcho env.

        Args:
            num_turns: Number of repeat turns per episode.
            message_kind: Kind of message to repeat ("word", "number", "uuid").
            seed: Optional RNG seed for reproducibility.

        Raises:
            ValueError: If num_turns or message_kind is invalid.
        """
        super().__init__()
        if num_turns <= 0:
            raise ValueError(f"num_turns must be > 0; got {num_turns}")
        if message_kind not in ("word", "number", "uuid"):
            raise ValueError(f"message_kind must be one of: 'word', 'number', 'uuid'; got {message_kind!r}")
        self.num_turns = int(num_turns)
        self.message_kind: MessageKind = message_kind
        self._seed = seed
        self._rng = random.Random(seed)

        self._turns_completed = 0
        self._done = False
        self._current: str | None = None

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
                "You are in ReverseEcho.\n"
                "Each turn, the environment will send you a message.\n"
                "Your task is to reply with the exact same message.\n"
                "Do not add extra words.\n"
                "Leading/trailing whitespace is ignored.\n"
            ),
        }

    def _next_message(self) -> str:
        """Generate the next message for the agent.

        Returns:
            Generated message string.
        """
        if self.message_kind == "word":
            return str(self._rng.choice(_DEFAULT_WORDS))
        if self.message_kind == "number":
            return str(self._rng.randint(0, 9999))
        if self.message_kind == "uuid":
            if self._seed is None:
                return str(uuid.uuid4())
            return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))
        raise ActionError(f"Unknown message_kind: {self.message_kind!r}")

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and start a new episode.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request for the new episode.
        """
        options = options or {}
        if "num_turns" in options:
            num_turns = int(options["num_turns"])
            if num_turns <= 0:
                raise ValueError(f"num_turns must be > 0; got {num_turns}")
            self.num_turns = num_turns
        if "message_kind" in options:
            message_kind = str(options["message_kind"])
            if message_kind not in ("word", "number", "uuid"):
                raise ValueError(f"message_kind must be one of: 'word', 'number', 'uuid'; got {message_kind!r}")
            self.message_kind = message_kind  # type: ignore[assignment]
        if "seed" in options:
            self._seed = options["seed"] if options["seed"] is None else int(options["seed"])
            self._rng = random.Random(self._seed)
        self._begin_episode()
        self._turns_completed = 0
        self._done = False
        if "message" in options:
            self._current = str(options["message"])
        else:
            self._current = self._next_message()
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": self._current},
                "needs_action": True,
                "info": {"turn": self._turns_completed, "num_turns": self.num_turns},
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
        if self._current is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        actions = self._normalize_actions(actions)
        expected = self._current.strip()
        got = actions["agent"].strip()

        correct = got == expected
        if not correct:
            self._done = True
            requests: list[Request] = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": -1.0,
                    "message": {
                        "role": "user",
                        "content": f"Incorrect. expected={expected!r} got={got!r}",
                    },
                    "needs_action": False,
                    "info": {"expected": expected, "got": got, "turn": self._turns_completed},
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = done
            return requests

        self._turns_completed += 1
        if self._turns_completed >= self.num_turns:
            self._done = True
            requests: list[Request] = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": 1.0,
                    "message": {"role": "user", "content": "Done."},
                    "needs_action": False,
                    "info": {"turn": self._turns_completed, "num_turns": self.num_turns},
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = done
            return requests

        self._current = self._next_message()
        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 1.0,
                "message": {"role": "user", "content": self._current},
                "needs_action": True,
                "info": {"turn": self._turns_completed, "num_turns": self.num_turns},
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
