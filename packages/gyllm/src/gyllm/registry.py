"""Global, lazy environment registry."""

import importlib
from dataclasses import dataclass
from typing import Any

from gyllm.batch import batch_envs
from gyllm.core import LLMEnv
from gyllm.rpc import docker_env, subprocess_env


@dataclass(frozen=True)
class EnvSpec:
    """Registry entry describing how to load an environment."""

    name: str
    entrypoint: str
    description: str | None = None

    def load(self) -> type[LLMEnv]:
        """Import the entrypoint and return the environment class.

        Returns:
            Environment class referenced by the entrypoint.

        Raises:
            ValueError: If the entrypoint is malformed.
            AttributeError: If the module lacks the expected attribute.
            TypeError: If the entrypoint does not resolve to an LLMEnv class.
        """
        if ":" not in self.entrypoint:
            raise ValueError(f"Invalid entrypoint {self.entrypoint!r}; expected 'module:Class'.")
        module_name, class_name = self.entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        entry = getattr(module, class_name, None)
        if entry is None:
            raise AttributeError(f"Module {module_name!r} has no attribute {class_name!r}")
        if not isinstance(entry, type) or not issubclass(entry, LLMEnv):
            raise TypeError(f"{self.entrypoint!r} did not resolve to an LLMEnv class")
        return entry


class EnvRegistry:
    """Global registry for named environments."""

    def __init__(self) -> None:
        """Initialize an empty environment registry."""
        self._specs: dict[str, EnvSpec] = {}

    def register(
        self,
        name: str,
        entrypoint: str,
        *,
        description: str | None = None,
        override: bool = False,
    ) -> None:
        """Register an env name to a lazy entrypoint string.

        Args:
            name: Environment name.
            entrypoint: Module and class path as "module:Class".
            description: Optional human-readable description.
            override: Whether to overwrite existing registration.

        Raises:
            ValueError: If name or entrypoint are invalid.
            KeyError: If the env is already registered without override.
        """
        if not name:
            raise ValueError("Env name must be a non-empty string")
        if ":" not in entrypoint:
            raise ValueError("entrypoint must be of the form 'module:Class'")
        if name in self._specs and not override:
            existing = self._specs[name]
            if existing.entrypoint == entrypoint:
                return
            raise KeyError(f"Env {name!r} already registered")
        spec = EnvSpec(name=name, entrypoint=entrypoint, description=description)
        self._specs[name] = spec

    def get(self, name: str, *, ensure_builtin: bool = True) -> EnvSpec:
        """Fetch a registry entry by name.

        Args:
            name: Environment name.
            ensure_builtin: Whether to register builtins before lookup.

        Returns:
            Registered environment spec.

        Raises:
            KeyError: If the env name is unknown.
        """
        if ensure_builtin:
            ensure_builtin_envs_registered()
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown env name: {name!r}") from exc

    def names(self, *, ensure_builtin: bool = True) -> list[str]:
        """List registered env names.

        Args:
            ensure_builtin: Whether to register builtins before listing.

        Returns:
            Sorted list of registered env names.
        """
        if ensure_builtin:
            ensure_builtin_envs_registered()
        return sorted(self._specs.keys())

    def make(
        self,
        name: str,
        *,
        mode: str = "local",
        env_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        num_envs: int | None = None,
        validate_actions: bool = True,
        autoreset: bool = False,
        python_executable: str | None = None,
        pythonpath: str | None = None,
        image: str | None = None,
        docker_executable: str = "docker",
        docker_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> LLMEnv:
        """Instantiate an env by name with local/subprocess/docker modes.

        Args:
            name: Environment name.
            mode: Execution mode ("local", "subprocess", or "docker").
            env_kwargs: Env init kwargs or list of per-env kwargs.
            num_envs: Optional number of env instances to create.
            validate_actions: Whether to validate action coverage in batches.
            autoreset: Whether to autoreset batched envs on completion.
            python_executable: Optional python executable for subprocess mode.
            pythonpath: Optional PYTHONPATH for subprocess mode.
            image: Docker image for docker mode.
            docker_executable: Docker binary to use.
            docker_args: Additional docker args.
            extra_env: Extra environment variables for subprocess/docker.

        Returns:
            Instantiated environment or batched environment.

        Raises:
            ValueError: If mode is invalid or required parameters are missing.
        """
        spec = self.get(name, ensure_builtin=True)

        env_kwargs_list, resolved_num_envs = _expand_env_kwargs(env_kwargs, num_envs)

        if mode == "local":
            env_cls = spec.load()
            envs = [env_cls(**kwargs) for kwargs in env_kwargs_list]
        elif mode == "subprocess":
            envs = [
                subprocess_env(
                    env=spec.entrypoint,
                    env_kwargs=kwargs,
                    python_executable=python_executable,
                    pythonpath=pythonpath,
                    extra_env=extra_env,
                )
                for kwargs in env_kwargs_list
            ]
        elif mode == "docker":
            if not image:
                raise ValueError("docker mode requires `image`")
            envs = [
                docker_env(
                    image=image,
                    env=spec.entrypoint,
                    env_kwargs=kwargs,
                    docker_executable=docker_executable,
                    docker_args=docker_args,
                    extra_env=extra_env,
                )
                for kwargs in env_kwargs_list
            ]
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        if resolved_num_envs > 1:
            return batch_envs(envs, validate_actions=validate_actions, autoreset=autoreset)
        return envs[0]


ENV_REGISTRY = EnvRegistry()

_BUILTIN_ENVS: list[tuple[str, str]] = [
    ("simple/tft_ipd", "gyllm.envs.simple.iterated_games:TftIpdEnv"),
    ("simple/ipd", "gyllm.envs.simple.iterated_games:IpdEnv"),
    ("simple/matrix_game", "gyllm.envs.simple.iterated_games:MatrixGameEnv"),
    ("simple/reverse", "gyllm.envs.simple.reverse:ReverseEnv"),
    ("simple/reverse_echo", "gyllm.envs.simple.reverse_echo:ReverseEcho"),
    ("simple/tic_tac_toe", "gyllm.envs.simple.tic_tac_toe:TicTacToeEnv"),
    ("openenv/atari", "gyllm.envs.openenv.atari:AtariEnv"),
    ("openenv/browsergym", "gyllm.envs.openenv.browsergym:BrowserGymEnv"),
    ("openenv/chat", "gyllm.envs.openenv.chat:ChatEnv"),
    ("openenv/coding", "gyllm.envs.openenv.coding:PythonCodeEnv"),
    ("openenv/connect4", "gyllm.envs.openenv.connect4:Connect4Env"),
    ("openenv/dipg", "gyllm.envs.openenv.dipg:DipgSafetyEnv"),
    ("openenv/echo", "gyllm.envs.openenv.echo:EchoEnv"),
    ("openenv/finrl", "gyllm.envs.openenv.finrl:FinRLEnv"),
    ("openenv/git", "gyllm.envs.openenv.git:GitEnv"),
    ("openenv/openspiel", "gyllm.envs.openenv.openspiel:OpenSpielEnv"),
    ("openenv/snake", "gyllm.envs.openenv.snake:SnakeEnv"),
    ("openenv/sumo_rl", "gyllm.envs.openenv.sumo_rl:SumoRLEnv"),
    ("openenv/textarena", "gyllm.envs.openenv.textarena:TextArenaEnv"),
    ("openenv/websearch", "gyllm.envs.openenv.websearch:WebSearchEnv"),
    ("misalignment/rvl", "gyllm.envs.misalignment.rescue_vs_loot:RescueVsLootEnv"),
    ("misalignment/fs", "gyllm.envs.misalignment.fragile_shortcut:FragileShortcutEnv"),
    ("misalignment/rrs", "gyllm.envs.misalignment.renewable_resource:RenewableResourceEnv"),
    ("reasoning/qa", "gyllm.envs.reasoning.qa:ReasoningQAEnv"),
]

_BUILTINS_LOADED = False


def ensure_builtin_envs_registered() -> None:
    """Register built-in env names without importing their modules."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for name, entrypoint in _BUILTIN_ENVS:
        ENV_REGISTRY.register(name, entrypoint)
    _BUILTINS_LOADED = True


def register_env(
    name: str,
    entrypoint: str,
    *,
    description: str | None = None,
    override: bool = False,
) -> None:
    """Register a custom env name to an entrypoint string.

    Args:
        name: Environment name.
        entrypoint: Module and class path as "module:Class".
        description: Optional human-readable description.
        override: Whether to overwrite existing registration.
    """
    ENV_REGISTRY.register(name, entrypoint, description=description, override=override)


def list_envs() -> list[str]:
    """Return all registered env names (builtins included).

    Returns:
        Sorted list of environment names.
    """
    return ENV_REGISTRY.names(ensure_builtin=True)


def make(
    name: str,
    *,
    mode: str = "local",
    env_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    num_envs: int | None = None,
    validate_actions: bool = True,
    autoreset: bool = False,
    python_executable: str | None = None,
    pythonpath: str | None = None,
    image: str | None = None,
    docker_executable: str = "docker",
    docker_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> LLMEnv:
    """Instantiate an environment by name from the global registry.

    Args:
        name: Environment name.
        mode: Execution mode ("local", "subprocess", or "docker").
        env_kwargs: Env init kwargs or list of per-env kwargs.
        num_envs: Optional number of env instances to create.
        validate_actions: Whether to validate action coverage in batches.
        autoreset: Whether to autoreset batched envs on completion.
        python_executable: Optional python executable for subprocess mode.
        pythonpath: Optional PYTHONPATH for subprocess mode.
        image: Docker image for docker mode.
        docker_executable: Docker binary to use.
        docker_args: Additional docker args.
        extra_env: Extra environment variables for subprocess/docker.

    Returns:
        Instantiated environment or batched environment.
    """
    return ENV_REGISTRY.make(
        name,
        mode=mode,
        env_kwargs=env_kwargs,
        num_envs=num_envs,
        validate_actions=validate_actions,
        autoreset=autoreset,
        python_executable=python_executable,
        pythonpath=pythonpath,
        image=image,
        docker_executable=docker_executable,
        docker_args=docker_args,
        extra_env=extra_env,
    )


def _expand_env_kwargs(
    env_kwargs: dict[str, Any] | list[dict[str, Any]] | None,
    num_envs: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Normalize env_kwargs into a list aligned with num_envs.

    Args:
        env_kwargs: Env kwargs mapping or list of mappings.
        num_envs: Optional number of envs requested.

    Returns:
        Tuple of per-env kwargs list and resolved env count.

    Raises:
        ValueError: If num_envs is invalid or list length mismatches.
    """
    if env_kwargs is None:
        resolved = num_envs or 1
        if resolved <= 0:
            raise ValueError(f"num_envs must be > 0; got {resolved}")
        return [{} for _ in range(resolved)], resolved

    if isinstance(env_kwargs, list):
        if num_envs is None:
            resolved = len(env_kwargs)
        else:
            resolved = num_envs
            if len(env_kwargs) != resolved:
                raise ValueError("env_kwargs list length must match num_envs")
        if resolved <= 0:
            raise ValueError(f"num_envs must be > 0; got {resolved}")
        return [dict(kwargs) for kwargs in env_kwargs], resolved

    resolved = num_envs or 1
    if resolved <= 0:
        raise ValueError(f"num_envs must be > 0; got {resolved}")
    return [dict(env_kwargs) for _ in range(resolved)], resolved
