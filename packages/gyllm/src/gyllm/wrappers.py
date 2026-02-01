from collections.abc import Callable, Sequence
from typing import Any, Literal

from gyllm.batch import BatchedEnv, batch_envs
from gyllm.core import ActorId, LLMEnv, Message, Request, actor_env_id

type BatchMode = Literal["env", "per_env"]


class EnvWrapper(LLMEnv):
    """Base wrapper for LLMEnv-style environments (single or batched)."""

    def __init__(self, env: LLMEnv) -> None:
        """Initialize the wrapper with a base env.

        Args:
            env: Environment instance to wrap.
        """
        super().__init__()
        self.env = env

    @property
    def agents(self) -> list[str]:
        """Return the wrapped env's agent id list.

        Returns:
            List of agent ids.
        """
        return self.env.agents

    @property
    def agent_ids(self) -> list[str]:
        """Return the wrapped env's agent id list."""
        return self.env.agent_ids

    @property
    def actors(self) -> list[ActorId]:
        """Return the wrapped env's actor list.

        Returns:
            List of actor ids.
        """
        return self.env.actors

    def agent_id(self, actor: ActorId) -> str:
        """Validate and return the agent id for a given actor.

        Args:
            actor: Actor id string.

        Returns:
            Agent id.
        """
        return self.env.agent_id(actor)

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for a given actor (internal)."""
        return self.env._system_message(actor)

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the env and return initial requests.

        Args:
            options: Optional reset options.

        Returns:
            Requests produced by reset.
        """
        return self.env.reset(options)

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply actions and return resulting requests.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after stepping the env.
        """
        return self.env.step(actions)

    def close(self) -> None:
        """Close the wrapped env."""
        self.env.close()

    @property
    def unwrapped(self) -> LLMEnv:
        """Return the base env beneath wrapper layers.

        Returns:
            Base env instance.
        """
        env = self.env
        while isinstance(env, EnvWrapper):
            env = env.env
        return env

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped env.

        Args:
            name: Attribute name to resolve.

        Returns:
            Attribute value from the wrapped env.
        """
        return getattr(self.env, name)


def unwrap_env(env: LLMEnv) -> LLMEnv:
    """Return the base env beneath any EnvWrapper layers.

    Args:
        env: Possibly wrapped environment.

    Returns:
        Base env instance.
    """
    while isinstance(env, EnvWrapper):
        env = env.env
    return env


def _find_batched_env(env: LLMEnv) -> BatchedEnv | None:
    """Return the base batched env if present.

    Args:
        env: Possibly wrapped environment.

    Returns:
        BatchedEnv instance if present, otherwise None.
    """
    base = unwrap_env(env)
    if isinstance(base, BatchedEnv):
        return base
    return None


def _sync_batched_pending(env: LLMEnv, requests: Sequence[Request]) -> None:
    """Synchronize pending actors for a batched env if present.

    Args:
        env: Possibly wrapped environment.
        requests: Latest request batch.
    """
    batched = _find_batched_env(env)
    if batched is not None:
        batched.update_pending(requests)


def _group_by_env(requests: Sequence[Request]) -> dict[int | None, list[Request]]:
    """Group requests by env id metadata.

    Args:
        requests: Requests to group.

    Returns:
        Mapping of env id to request list.
    """
    grouped: dict[int | None, list[Request]] = {}
    for request in requests:
        env_id = actor_env_id(request["actor"])
        grouped.setdefault(env_id, []).append(request)
    return grouped


def wrap_env(
    env: LLMEnv,
    wrapper: Callable[[LLMEnv], LLMEnv],
    *,
    batch_mode: BatchMode = "env",
) -> LLMEnv:
    """Apply a wrapper to an env, optionally per-env for BatchedEnv instances.

    Args:
        env: Environment to wrap.
        wrapper: Callable that produces a wrapped env.
        batch_mode: Whether to wrap the batched env or per-env instances.

    Returns:
        Wrapped environment.

    Raises:
        ValueError: If batch_mode is invalid.
    """
    if isinstance(env, BatchedEnv):
        if batch_mode == "env":
            return wrapper(env)
        if batch_mode != "per_env":
            raise ValueError(f"Unknown batch_mode: {batch_mode!r}")
        wrapped_envs = [wrapper(inner_env) for inner_env in env.envs]
        return batch_envs(
            wrapped_envs,
            validate_actions=env.validate_actions,
            autoreset=env.autoreset,
        )
    return wrapper(env)


class ActionParsingWrapper(EnvWrapper):
    """Preprocess actions with a parsing function before stepping the env."""

    def __init__(
        self,
        env: LLMEnv,
        parse_action: Callable[..., str],
        *,
        pass_actor: bool = False,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
            parse_action: Callable that transforms raw actions.
            pass_actor: Whether to pass actor ids to the parser.
        """
        super().__init__(env)
        self._action_parser = parse_action
        self._pass_actor = pass_actor

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Parse actions then step the env.

        Args:
            actions: Mapping of actor ids to raw action strings.

        Returns:
            Requests produced by the wrapped env.
        """
        if self._pass_actor:
            parsed_actions = {actor: self._action_parser(actor, action) for actor, action in actions.items()}
        else:
            parsed_actions = {actor: self._action_parser(action) for actor, action in actions.items()}
        return self.env.step(parsed_actions)


class MaxStepsWrapper(EnvWrapper):
    """Terminate episodes after a fixed number of steps."""

    def __init__(self, env: LLMEnv, max_steps: int) -> None:
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
            max_steps: Maximum steps per episode.

        Raises:
            ValueError: If max_steps is not positive.
        """
        super().__init__(env)
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0; got {max_steps}")
        self.max_steps = int(max_steps)
        self._steps: dict[int | None, int] = {}
        self._reset_steps()

    def _reset_steps(self) -> None:
        """Reset step counters for all envs."""
        batched = _find_batched_env(self.env)
        if batched is None:
            self._steps = {None: 0}
        else:
            self._steps = dict.fromkeys(range(len(batched.envs)), 0)

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the env and step counters.

        Args:
            options: Optional reset options.

        Returns:
            Requests produced by reset.
        """
        requests = self.env.reset(options)
        self._reset_steps()
        return requests

    def reset_at(
        self,
        env_index: int,
        options: dict[str, object] | None = None,
        *,
        update_pending: bool = True,
    ) -> list[Request]:
        """Reset a single env in a batched wrapper.

        Args:
            env_index: Index of env to reset.
            options: Optional reset options.
            update_pending: Whether to update pending actors.

        Returns:
            Requests produced by reset.

        Raises:
            AttributeError: If the wrapped env is not batched.
        """
        reset_at = getattr(self.env, "reset_at", None)
        if reset_at is None:
            raise AttributeError("reset_at is only available for batched environments.")
        requests = reset_at(env_index, options, update_pending=update_pending)
        self._steps[env_index] = 0
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Step the env and terminate when max steps are reached.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced by the wrapped env.
        """
        requests = self.env.step(actions)

        env_ids = {actor_env_id(actor) for actor in actions}
        for env_id in env_ids:
            self._steps[env_id] = self._steps.get(env_id, 0) + 1

        if not requests:
            return requests

        grouped = _group_by_env(requests)
        updated = False
        for env_id, env_requests in grouped.items():
            if not env_requests:
                continue
            steps = self._steps.get(env_id, 0)
            done = not any(req["needs_action"] for req in env_requests)
            if steps >= self.max_steps and not done:
                for request in env_requests:
                    request["needs_action"] = False
                    request["episode_end"] = True
                updated = True

        if updated:
            _sync_batched_pending(self.env, requests)
        return requests


class AutoResetWrapper(EnvWrapper):
    """Automatically reset envs that finish an episode."""

    def __init__(
        self,
        env: LLMEnv,
        *,
        reset_options: dict[str, object] | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
            reset_options: Optional reset options for autoreset.
        """
        super().__init__(env)
        self._reset_options = reset_options

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the env and sync pending actors.

        Args:
            options: Optional reset options.

        Returns:
            Requests produced by reset.
        """
        requests = self.env.reset(options)
        _sync_batched_pending(self.env, requests)
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Step the env and autoreset finished episodes.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced by stepping and any autoresets.
        """
        requests = self.env.step(actions)
        batched = _find_batched_env(self.env)

        if batched is None:
            done = not requests or not any(req["needs_action"] for req in requests)
            if done:
                reset_requests = self.env.reset(self._reset_options)
                requests = list(requests) + reset_requests
            _sync_batched_pending(self.env, requests)
            return requests

        reset_at = getattr(self.env, "reset_at", None)
        if reset_at is None:
            raise RuntimeError("AutoResetWrapper requires a batched env with reset_at().")

        appended: list[Request] = []
        env_ids = {actor_env_id(actor) for actor in actions}
        if not env_ids:
            _sync_batched_pending(self.env, requests)
            return requests

        grouped = _group_by_env(requests)
        for env_id in env_ids:
            if env_id is None:
                raise ValueError("AutoResetWrapper expected env_id metadata in batched actor IDs.")
            env_requests = grouped.get(env_id, [])
            if env_requests and any(req["needs_action"] for req in env_requests):
                continue
            appended.extend(reset_at(env_id, self._reset_options, update_pending=False))

        if appended:
            requests = list(requests) + appended
        _sync_batched_pending(self.env, requests)
        return requests


__all__ = [
    "ActionParsingWrapper",
    "AutoResetWrapper",
    "BatchMode",
    "EnvWrapper",
    "MaxStepsWrapper",
    "unwrap_env",
    "wrap_env",
]
