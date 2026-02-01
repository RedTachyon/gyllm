import pytest

from gyllm.core import make_actor_id
from gyllm.envs.openenv.echo import EchoEnv


def test_reset_includes_system_message() -> None:
    env = EchoEnv()
    req = env.reset()[0]
    msg = req.get("system_message")
    assert msg is not None
    assert msg["role"] == "system"


def test_reset_returns_user_message() -> None:
    env = EchoEnv()
    req = env.reset()[0]
    assert req["actor"] == make_actor_id("agent")
    assert req["system_message"]["role"] == "system"
    assert req["message"]["role"] == "user"
    assert req["needs_action"] is True
    assert req["episode_id"] == 0
    assert req["episode_start"] is True
    assert req["episode_end"] is False


def test_step_requires_actor_ids() -> None:
    env = EchoEnv()
    env.reset()
    out = env.step({make_actor_id("agent"): "hello"})[0]
    assert out["actor"] == make_actor_id("agent")
    assert "system_message" not in out
    assert out["message"]["role"] == "user"
    assert out["needs_action"] is True
    assert out["reward"] > 0.0
    assert out["episode_id"] == 0
    assert out["episode_start"] is False
    assert out["episode_end"] is False


def test_step_rejects_nonzero_env_id() -> None:
    env = EchoEnv()
    env.reset()
    with pytest.raises(KeyError):
        env.step({make_actor_id("agent", env_id=1): "hello"})
