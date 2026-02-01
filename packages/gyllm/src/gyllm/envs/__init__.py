"""Environment API and helpers for GYLLM."""

from gyllm.batch import BatchedEnv, batch_envs, vectorize
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
    parse_actor_id,
)
from gyllm.registry import ENV_REGISTRY, EnvRegistry, list_envs, make, register_env
from gyllm.rpc import DockerEnv, RpcEnvClient, SubprocessEnv, docker_env, subprocess_env
from gyllm.wrappers import (
    ActionParsingWrapper,
    AutoResetWrapper,
    BatchMode,
    EnvWrapper,
    MaxStepsWrapper,
    unwrap_env,
    wrap_env,
)

__all__ = [
    "ENV_REGISTRY",
    "ActionError",
    "ActionParsingWrapper",
    "ActorId",
    "AutoResetWrapper",
    "BatchMode",
    "BatchedEnv",
    "DockerEnv",
    "EnvRegistry",
    "EnvWrapper",
    "LLMEnv",
    "MaxStepsWrapper",
    "Message",
    "Request",
    "RpcEnvClient",
    "SubprocessEnv",
    "actor_agent_id",
    "actor_env_id",
    "actor_episode_id",
    "batch_envs",
    "docker_env",
    "list_envs",
    "make",
    "make_actor_id",
    "parse_actor_id",
    "register_env",
    "subprocess_env",
    "unwrap_env",
    "vectorize",
    "wrap_env",
]
