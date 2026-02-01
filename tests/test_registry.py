from gyllm import list_envs, make
from gyllm.core import actor_env_id, make_actor_id


def test_registry_lists_builtins() -> None:
    names = list_envs()
    assert "openenv/echo" in names
    assert "reasoning/qa" in names


def test_make_local_env() -> None:
    env = make("openenv/echo")
    req = env.reset()[0]
    assert req["actor"] == make_actor_id("agent")
    env.close()


def test_make_batched_env() -> None:
    env = make("openenv/echo", num_envs=2)
    reqs = env.reset()
    assert {actor_env_id(r["actor"]) for r in reqs} == {0, 1}
    env.close()
