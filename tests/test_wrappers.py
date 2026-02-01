from gyllm.batch import BatchedEnv, batch_envs
from gyllm.core import actor_env_id
from gyllm.envs.openenv.echo import EchoEnv
from gyllm.envs.simple.reverse_echo import ReverseEcho
from gyllm.wrappers import (
    ActionParsingWrapper,
    AutoResetWrapper,
    MaxStepsWrapper,
    wrap_env,
)


def test_action_parsing_wrapper_preprocesses_action() -> None:
    env = ActionParsingWrapper(EchoEnv(), lambda action: action.upper())
    req = env.reset()[0]
    out = env.step({req["actor"]: "hello"})[0]
    assert out["message"]["content"] == "HELLO"


def test_max_steps_wrapper_terminates_episode() -> None:
    env = MaxStepsWrapper(
        ReverseEcho(num_turns=3, message_kind="word", seed=0),
        max_steps=1,
    )
    req = env.reset()[0]
    out = env.step({req["actor"]: req["message"]["content"]})
    assert all(not r["needs_action"] for r in out)
    assert all(r["episode_end"] for r in out)


def test_autoreset_wrapper_resets_single_env() -> None:
    env = AutoResetWrapper(ReverseEcho(num_turns=1, message_kind="word", seed=0))
    req = env.reset()[0]
    out = env.step({req["actor"]: req["message"]["content"]})
    terminal = [r for r in out if r["episode_end"]]
    starts = [r for r in out if r["episode_start"]]
    assert terminal
    assert starts
    assert {r["episode_id"] for r in terminal} == {0}
    assert {r["episode_id"] for r in starts} == {1}


def test_wrap_env_per_env_batches_wrapped_envs() -> None:
    envs = [
        ReverseEcho(num_turns=2, message_kind="word", seed=0),
        ReverseEcho(num_turns=2, message_kind="word", seed=1),
    ]
    batched = batch_envs(envs, validate_actions=False)
    wrapped = wrap_env(
        batched,
        lambda inner: MaxStepsWrapper(inner, max_steps=1),
        batch_mode="per_env",
    )
    assert isinstance(wrapped, BatchedEnv)
    assert isinstance(wrapped.envs[0], MaxStepsWrapper)
    assert isinstance(wrapped.envs[1], MaxStepsWrapper)

    reqs = wrapped.reset()
    actions = {r["actor"]: r["message"]["content"] for r in reqs}
    out = wrapped.step(actions)
    done_envs = {actor_env_id(r["actor"]) for r in out if r["episode_end"]}
    assert done_envs == {0, 1}
