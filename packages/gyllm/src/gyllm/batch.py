from collections.abc import Callable, Sequence

from gyllm.core import (
    ActionError,
    ActorId,
    LLMEnv,
    Message,
    Request,
    actor_agent_id,
    actor_env_id,
    actor_episode_id,
    make_actor_id,
)


class BatchedEnv(LLMEnv):
    """
    Combine multiple single-instance envs into one unified (vectorized) env.

    Each underlying env is treated as env_id=0 internally; this wrapper
    rewrites actor IDs to include env metadata (e.g. `agent::env=2`).
    When autoreset is enabled, completed envs are reset in place.
    """

    def __init__(
        self,
        envs: Sequence[LLMEnv],
        *,
        validate_actions: bool = True,
        autoreset: bool = False,
    ) -> None:
        """Initialize a batched environment wrapper.

        Args:
            envs: Sequence of single-instance environments.
            validate_actions: Whether to enforce per-step action coverage.
            autoreset: Whether to reset envs automatically when done.
        """
        super().__init__()
        self._envs = list(envs)
        self._validate_actions = validate_actions
        self._autoreset = autoreset
        self._pending: set[ActorId] = set()
        self._last_requests_by_env: dict[int, list[Request]] = {}

    def _stash_last_requests(self, env_index: int, requests: Sequence[Request]) -> None:
        self._last_requests_by_env[env_index] = list(requests)

    def _repeat_last_requests(self, env_index: int) -> list[Request]:
        last_requests = self._last_requests_by_env.get(env_index, [])
        if not last_requests:
            return []
        for request in last_requests:
            request["episode_start"] = False
            request.pop("system_message", None)
            request["repeat_count"] = int(request.get("repeat_count", 0)) + 1
        self._last_requests_by_env[env_index] = last_requests
        return list(last_requests)

    @property
    def envs(self) -> list[LLMEnv]:
        """Return the underlying env list.

        Returns:
            List of wrapped env instances.
        """
        return self._envs

    @property
    def validate_actions(self) -> bool:
        """Return whether action validation is enabled.

        Returns:
            True if action validation is enabled.
        """
        return self._validate_actions

    @property
    def autoreset(self) -> bool:
        """Return whether autoreset is enabled.

        Returns:
            True if autoreset is enabled.
        """
        return self._autoreset

    @property
    def actors(self) -> list[ActorId]:
        """Return actor ids for all env instances.

        Returns:
            List of actor ids including env metadata.
        """
        actors: list[ActorId] = []
        for env_index, env in enumerate(self._envs):
            for actor in env.actors:
                agent_id = actor_agent_id(actor)
                episode_id = actor_episode_id(actor)
                actors.append(make_actor_id(agent_id, env_id=env_index, episode_id=episode_id))
        return actors

    def agent_id(self, actor: ActorId) -> str:
        """Validate and return the agent id for a given actor.

        Args:
            actor: Actor id string.

        Returns:
            Agent identifier.

        Raises:
            KeyError: If the env index or agent id is unknown.
        """
        env_index = actor_env_id(actor)
        if env_index is None:
            raise KeyError(f"{self.__class__.__name__} requires actor IDs with env metadata; got {actor!r}.")
        try:
            env = self._envs[env_index]
        except IndexError as exc:
            raise KeyError(f"Unknown env_index: {env_index}") from exc
        agent_id = actor_agent_id(actor)
        if agent_id not in env.agent_ids:
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return agent_id

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for a given actor (internal)."""
        env_index = actor_env_id(actor)
        if env_index is None:
            raise KeyError(f"{self.__class__.__name__} requires actor IDs with env metadata; got {actor!r}.")
        agent_id = actor_agent_id(actor)
        episode_id = actor_episode_id(actor)
        try:
            env = self._envs[env_index]
        except IndexError as exc:
            raise KeyError(f"Unknown env_index: {env_index}") from exc
        return env._system_message(make_actor_id(agent_id, episode_id=episode_id))

    def update_pending(self, requests: Sequence[Request]) -> None:
        """Update the pending actor set from a request batch.

        Args:
            requests: Requests to scan for actors needing actions.
        """
        self._pending = {r["actor"] for r in requests if r["needs_action"]}

    def reset_at(
        self,
        env_index: int,
        options: dict[str, object] | None = None,
        *,
        update_pending: bool = True,
    ) -> list[Request]:
        """Reset a single env inside the batch.

        Args:
            env_index: Index of the env to reset.
            options: Optional reset options for the env.
            update_pending: Whether to refresh the pending actor set.

        Returns:
            Requests produced by the reset.

        Raises:
            KeyError: If the env index is invalid.
        """
        try:
            env = self._envs[env_index]
        except IndexError as exc:
            raise KeyError(f"Unknown env_index: {env_index}") from exc

        local_requests = env.reset(options)
        requests: list[Request] = []
        for request in local_requests:
            agent_id = actor_agent_id(request["actor"])
            local_env_id = actor_env_id(request["actor"])
            if local_env_id not in (None, 0):
                raise ValueError(
                    "BatchedEnv expects underlying envs to use env_id=0; "
                    f"got {request['actor']!r} from {env.__class__.__name__}."
                )
            episode_id = actor_episode_id(request["actor"])
            env_request: Request = {
                "actor": make_actor_id(agent_id, env_id=env_index, episode_id=episode_id),
                "reward": request["reward"],
                "message": request["message"],
                "needs_action": request["needs_action"],
                "info": request.get("info", {}),
                "repeat_count": int(request.get("repeat_count", 0)),
                "episode_id": request.get("episode_id", 0),
                "group_id": request.get("group_id"),
                "episode_start": True,
                "episode_end": False,
            }
            if "system_message" in request:
                env_request["system_message"] = request["system_message"]
            requests.append(env_request)

        if update_pending:
            self._pending = {actor for actor in self._pending if actor_env_id(actor) != env_index}
            self._pending.update({r["actor"] for r in requests if r["needs_action"]})
        self._stash_last_requests(env_index, requests)
        return requests

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset all envs and return their initial requests.

        Args:
            options: Optional shared or per-env reset options.

        Returns:
            Requests produced across all envs.

        Raises:
            TypeError: If env_options is not a list when provided.
            ValueError: If env_options length does not match env count.
        """
        requests: list[Request] = []
        options = options or {}
        env_options = options.get("env_options")
        shared_options = {k: v for k, v in options.items() if k != "env_options"}

        if env_options is None:
            per_env_options: list[dict[str, object] | None] = [
                dict(shared_options) if shared_options else None for _ in self._envs
            ]
        else:
            if not isinstance(env_options, list):
                raise TypeError("env_options must be a list of dicts or None")
            if len(env_options) != len(self._envs):
                raise ValueError(f"env_options must have {len(self._envs)} entries; got {len(env_options)}")
            per_env_options = []
            for entry in env_options:
                if entry is None:
                    per_env_options.append(dict(shared_options) if shared_options else None)
                    continue
                if not isinstance(entry, dict):
                    raise TypeError("env_options entries must be dicts or None")
                merged = dict(shared_options) if shared_options else {}
                merged.update(entry)
                per_env_options.append(merged)

        for env_index in range(len(self._envs)):
            requests.extend(self.reset_at(env_index, per_env_options[env_index], update_pending=False))

        self.update_pending(requests)
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Step envs with a batch of actions.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced by stepping the envs.

        Raises:
            KeyError: If action validation fails or actor ids are invalid.
        """
        if self._validate_actions:
            missing = self._pending - set(actions.keys())
            if missing:
                raise KeyError(f"Missing actions for actors: {sorted(missing)!r}")

        actions_by_env: dict[int, dict[ActorId, str]] = {}
        for actor, completion in actions.items():
            env_index = actor_env_id(actor)
            if env_index is None:
                raise KeyError(f"Missing env metadata for actor: {actor!r}")
            if env_index < 0 or env_index >= len(self._envs):
                raise KeyError(f"Unknown env_index: {env_index}")
            agent_id = actor_agent_id(actor)
            episode_id = actor_episode_id(actor)
            actions_by_env.setdefault(env_index, {})[make_actor_id(agent_id, episode_id=episode_id)] = completion

        requests: list[Request] = []
        for env_index, env in enumerate(self._envs):
            local_actions = actions_by_env.get(env_index)
            if not local_actions:
                continue

            try:
                local_requests = env.step(local_actions)
            except ActionError:
                repeated = self._repeat_last_requests(env_index)
                if not repeated:
                    raise
                requests.extend(repeated)
                continue
            done = not local_requests or not any(r["needs_action"] for r in local_requests)
            env_requests: list[Request] = []
            for request in local_requests:
                agent_id = actor_agent_id(request["actor"])
                local_env_id = actor_env_id(request["actor"])
                if local_env_id not in (None, 0):
                    raise ValueError(
                        "BatchedEnv expects underlying envs to use env_id=0; "
                        f"got {request['actor']!r} from {env.__class__.__name__}."
                    )
                episode_id = actor_episode_id(request["actor"])
                env_request: Request = {
                    "actor": make_actor_id(agent_id, env_id=env_index, episode_id=episode_id),
                    "reward": request["reward"],
                    "message": request["message"],
                    "needs_action": request["needs_action"],
                    "info": request.get("info", {}),
                    "repeat_count": int(request.get("repeat_count", 0)),
                    "episode_id": request.get("episode_id", 0),
                    "group_id": request.get("group_id"),
                    "episode_start": request.get("episode_start", False),
                    "episode_end": bool(done),
                }
                if "system_message" in request:
                    env_request["system_message"] = request["system_message"]
                env_requests.append(env_request)

            if env_requests:
                requests.extend(env_requests)

            if self._autoreset and done:
                requests.extend(self.reset_at(env_index, update_pending=False))
            else:
                self._stash_last_requests(env_index, env_requests)

        self.update_pending(requests)
        return requests

    def close(self) -> None:
        """Close all underlying envs."""
        for env in self._envs:
            env.close()


def batch_envs(
    envs: Sequence[LLMEnv],
    *,
    validate_actions: bool = True,
    autoreset: bool = False,
) -> BatchedEnv:
    """Create a batched env from explicit env instances.

    Args:
        envs: Env instances to batch.
        validate_actions: Whether to enforce per-step action coverage.
        autoreset: Whether to reset envs automatically when done.

    Returns:
        Batched environment instance.
    """
    return BatchedEnv(envs, validate_actions=validate_actions, autoreset=autoreset)


def vectorize(
    make_env: Callable[[], LLMEnv],
    num_envs: int,
    *,
    validate_actions: bool = True,
    autoreset: bool = False,
) -> BatchedEnv:
    """Create a batched env from a factory and environment count.

    Args:
        make_env: Factory that creates a single env.
        num_envs: Number of env instances to create.
        validate_actions: Whether to enforce per-step action coverage.
        autoreset: Whether to reset envs automatically when done.

    Returns:
        Batched environment instance.

    Raises:
        ValueError: If num_envs is not positive.
    """
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0; got {num_envs}")
    return BatchedEnv(
        [make_env() for _ in range(num_envs)],
        validate_actions=validate_actions,
        autoreset=autoreset,
    )
