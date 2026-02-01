from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict


class Message(TypedDict):
    role: str
    content: str


type ActorId = str
type ActorMeta = dict[str, str]

ACTOR_ID_SEPARATOR = "::"


def make_actor_id(
    agent_id: str,
    *,
    env_id: int | None = None,
    episode_id: str | None = None,
) -> ActorId:
    """Build a structured actor id string.

    Examples:
        >>> make_actor_id("agent")
        'agent'
        >>> make_actor_id("agent", env_id=2, episode_id="e5")
        'agent::env=2::episode=e5'

    Args:
        agent_id: Agent identifier.
        env_id: Optional environment index.
        episode_id: Optional episode identifier.

    Returns:
        Actor id string.

    Raises:
        ValueError: If the agent id is empty.
    """
    if not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    parts = [agent_id]
    if env_id is not None:
        parts.append(f"env={env_id}")
    if episode_id is not None:
        parts.append(f"episode={episode_id}")
    return ACTOR_ID_SEPARATOR.join(parts)


def parse_actor_id(actor: ActorId) -> tuple[str, ActorMeta]:
    """Parse an actor id into agent id and metadata.

    Examples:
        >>> parse_actor_id("agent::env=2::episode=e5")
        ('agent', {'env': '2', 'episode': 'e5'})
        >>> parse_actor_id("agent")
        ('agent', {})

    Args:
        actor: Actor id string to parse.

    Returns:
        Tuple of agent identifier and metadata mapping.

    Raises:
        TypeError: If the actor id is not a string.
    """
    if not isinstance(actor, str):
        raise TypeError(f"actor must be a string ActorId; got {type(actor).__name__}")
    parts = actor.split(ACTOR_ID_SEPARATOR)
    agent_id = parts[0]
    meta: ActorMeta = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            if key:
                meta[key] = value
        else:
            meta[part] = ""
    return agent_id, meta


def actor_agent_id(actor: ActorId) -> str:
    """Extract the agent identifier from an actor id.

    Args:
        actor: Actor id string.

    Returns:
        Agent identifier.

    Raises:
        ValueError: If the actor id has no agent id.
    """
    agent_id, _meta = parse_actor_id(actor)
    if not agent_id:
        raise ValueError(f"actor must include a non-empty agent id; got {actor!r}")
    return agent_id


def actor_env_id(actor: ActorId) -> int | None:
    """Extract the env id from an actor id.

    Args:
        actor: Actor id string.

    Returns:
        Environment id or None if absent.

    Raises:
        ValueError: If the env id is present but invalid.
    """
    _agent_id, meta = parse_actor_id(actor)
    if "env" not in meta:
        return None
    try:
        return int(meta["env"])
    except ValueError as exc:
        raise ValueError(f"Invalid env id in actor {actor!r}") from exc


def actor_episode_id(actor: ActorId) -> str | None:
    """Extract the episode id from an actor id.

    Args:
        actor: Actor id string.

    Returns:
        Episode id or None if absent.
    """
    _agent_id, meta = parse_actor_id(actor)
    return meta.get("episode")


class Request(TypedDict):
    """
    Text-based request for a single actor (agent in a specific env instance).

    Roles are intentionally restricted at the API boundary:
    - System prompt: role="system" (via `system_message` on episode start)
    - Env responses/observations: role="user" (this `message`)
    - Agent actions/completions: role="assistant" (provided back to `step`)

    Episode metadata (populated by envs/wrappers, may be omitted pre-finalization):
    - `episode_id`: per-env integer episode index
    - `group_id`: optional grouping key for batch-level algorithms (e.g., GRPO)
    - `episode_start`: True on the first request of an episode
    - `episode_end`: True when the episode has ended
    - `repeat_count`: number of consecutive repeats due to action errors
    - `info`: Extra metadata dict for debugging or logging
    - `system_message`: System message for the actor, present iff `episode_start`
    """

    actor: ActorId
    reward: float
    message: Message
    needs_action: bool
    info: dict[str, Any]
    repeat_count: NotRequired[int]
    episode_id: NotRequired[int]
    group_id: NotRequired[str | int]
    episode_start: NotRequired[bool]
    episode_end: NotRequired[bool]
    system_message: NotRequired[Message]


class LLMEnv(ABC):
    """
    Base class for a single environment instance.

    Vectorization is handled by wrappers that combine multiple `LLMEnv`
    instances and attach env metadata to `ActorId` (e.g. `agent::env=3`).
    """

    agents: list[str]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize episode tracking state.

        Args:
            *args: Positional args (unused, for compatibility).
            **kwargs: Keyword args (unused, for compatibility).
        """
        self._episode_id = -1

    @property
    def agent_ids(self) -> list[str]:
        """Return agent identifiers for the env."""
        return self.agents

    @property
    def actors(self) -> list[ActorId]:
        """Return actor ids for all agent identifiers in the env.

        Returns:
        List of actor ids for each agent id.
        """
        return [make_actor_id(agent_id) for agent_id in self.agent_ids]

    def agent_id(self, actor: ActorId) -> str:
        """Validate and return the agent id for a given actor.

        Args:
            actor: Actor id string.

        Returns:
            Agent id.

        Raises:
            KeyError: If the actor references an unknown agent id or env id.
        """
        agent_id = actor_agent_id(actor)
        env_id = actor_env_id(actor)
        if env_id not in (None, 0):
            raise KeyError(f"{self.__class__.__name__} only supports env_id=0; got {actor!r}.")
        if agent_id not in self.agent_ids:
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return agent_id

    @abstractmethod
    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the specified actor.

        Env implementations should use this internally and include the
        result in `Request["system_message"]` for episode start requests.

        Args:
            actor: Actor id string.

        Returns:
            System message for the actor.
        """
        raise NotImplementedError

    def _begin_episode(self) -> int:
        """Advance and return the next episode id.

        Returns:
            New episode id.
        """
        self._episode_id += 1
        return self._episode_id

    def _normalize_actions(self, actions: dict[ActorId, str]) -> dict[str, str]:
        """Normalize actor actions into agent-id keyed actions.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Mapping of agent ids to action strings.
        """
        agent_actions: dict[str, str] = {}
        for actor_id, completion in actions.items():
            agent_id = self.agent_id(actor_id)
            agent_actions[agent_id] = completion
        return agent_actions

    @abstractmethod
    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the env and return initial requests for the new episode.

        Args:
            options: Optional env-specific reset options.

        Returns:
            Initial requests for the new episode.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply actions and return resulting requests for the current episode.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests emitted after applying the actions.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close any resources held by the environment."""
        return


class ActionError(Exception):
    """Custom exception for invalid actions in the environment."""

    def __init__(self, message: str) -> None:
        """Initialize the error with a message.

        Args:
            message: Error description.
        """
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """Format the error message.

        Returns:
            Formatted error string.
        """
        return f"ActionError: {self.message}"
