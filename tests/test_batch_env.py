from gyllm.batch import batch_envs, vectorize
from gyllm.core import ActionError, LLMEnv, Message, Request, actor_env_id, make_actor_id
from gyllm.envs.openenv.echo import EchoEnv
from gyllm.envs.simple.reverse_echo import ReverseEcho


class _ActionErrorEnv(LLMEnv):
    agents: list[str] = ["agent"]

    def __init__(self, expected_action: str) -> None:
        super().__init__()
        self._expected_action = expected_action
        self._done = False

    def _system_message(self, actor: str) -> Message:
        self.agent_id(actor)
        return {"role": "system", "content": "system"}

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        del options
        self._done = False
        self._begin_episode()
        return [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": "start"},
                "needs_action": True,
                "info": {},
                "episode_id": self._episode_id,
                "episode_start": True,
                "episode_end": False,
            }
        ]

    def step(self, actions: dict[str, str]) -> list[Request]:
        actions = self._normalize_actions(actions)
        if "agent" not in actions:
            raise ActionError("Missing action for 'agent'")
        if actions["agent"] != self._expected_action:
            raise ActionError(f"Expected {self._expected_action!r}")
        if self._done:
            return []
        self._done = True
        return [
            {
                "actor": make_actor_id("agent"),
                "reward": 1.0,
                "message": {"role": "user", "content": "done"},
                "needs_action": False,
                "info": {},
                "episode_id": self._episode_id,
                "episode_start": False,
                "episode_end": True,
            }
        ]


def test_vectorize_rewrites_actor_ids() -> None:
    env = vectorize(lambda: EchoEnv(), num_envs=3)
    reqs = env.reset()
    actors = {r["actor"] for r in reqs}
    assert actors == {make_actor_id("agent", env_id=i) for i in range(3)}


def test_vectorized_step() -> None:
    env = vectorize(lambda: EchoEnv(), num_envs=2)
    reqs = env.reset()
    actions = {r["actor"]: "hi" for r in reqs if r["needs_action"]}
    out = env.step(actions)
    assert {r["actor"] for r in out} == {make_actor_id("agent", env_id=i) for i in range(2)}


def test_batch_autoreset_restarts_done_envs() -> None:
    envs = [
        ReverseEcho(num_turns=1, message_kind="word", seed=0),
        ReverseEcho(num_turns=1, message_kind="word", seed=1),
    ]
    env = batch_envs(envs, autoreset=True)
    reqs = env.reset()
    actions = {r["actor"]: r["message"]["content"] for r in reqs}
    out = env.step(actions)

    terminal = [r for r in out if r["episode_end"]]
    starts = [r for r in out if r["episode_start"]]
    assert terminal
    assert starts
    assert {actor_env_id(r["actor"]) for r in terminal} == {0, 1}
    assert {actor_env_id(r["actor"]) for r in starts} == {0, 1}
    assert {r["episode_id"] for r in terminal} == {0}
    assert {r["episode_id"] for r in starts} == {1}


def test_batched_env_repeats_request_after_action_error() -> None:
    env = batch_envs([_ActionErrorEnv("ok"), _ActionErrorEnv("go")])
    reqs = env.reset()
    actions: dict[str, str] = {}
    for req in reqs:
        env_id = actor_env_id(req["actor"])
        if env_id == 0:
            actions[req["actor"]] = "bad"
        else:
            actions[req["actor"]] = "go"
    out = env.step(actions)

    by_env = {actor_env_id(req["actor"]): req for req in out}
    assert by_env[0]["message"]["content"] == "start"
    assert by_env[0]["needs_action"] is True
    assert by_env[0]["episode_start"] is False
    assert by_env[1]["message"]["content"] == "done"
    assert by_env[1]["needs_action"] is False
