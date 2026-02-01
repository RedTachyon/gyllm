import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from gyllm.core import make_actor_id
from gyllm.envs.openenv.chat import ChatEnv
from gyllm.envs.openenv.coding import PythonCodeEnv
from gyllm.envs.openenv.connect4 import Connect4Env
from gyllm.envs.openenv.dipg import DipgSafetyEnv
from gyllm.envs.openenv.echo import EchoEnv
from gyllm.envs.openenv.finrl import FinRLEnv
from gyllm.envs.openenv.git import GitEnv
from gyllm.envs.openenv.snake import SnakeEnv
from gyllm.envs.openenv.websearch import WebSearchEnv


def test_echo_env() -> None:
    env = EchoEnv()
    req = env.reset()[0]
    out = env.step({req["actor"]: "hello"})[0]
    assert out["message"]["content"] == "hello"
    assert out["reward"] > 0.0


def test_chat_env_basic_loop() -> None:
    env = ChatEnv(max_turns=2)
    req = env.reset()[0]
    assert req["needs_action"] is True
    out = env.step({req["actor"]: "hello"})[0]
    assert out["message"]["content"]
    assert out["needs_action"] is True
    out2 = env.step({out["actor"]: "bye"})[0]
    assert out2["needs_action"] is False


def test_connect4_two_player() -> None:
    env = Connect4Env(opponent=None)
    reqs = env.reset()
    assert len(reqs) == 1
    assert reqs[0]["actor"] == make_actor_id("player_a")
    assert reqs[0]["needs_action"] is True

    out = env.step({reqs[0]["actor"]: json.dumps({"column": 3})})
    assert len(out) == 1
    assert out[0]["actor"] == make_actor_id("player_b")
    assert out[0]["needs_action"] is True

    out2 = env.step({out[0]["actor"]: json.dumps({"column": 3})})
    assert len(out2) == 1
    assert out2[0]["actor"] == make_actor_id("player_a")
    assert out2[0]["needs_action"] is True


def test_connect4_single_agent() -> None:
    env = Connect4Env(opponent="random", seed=0)
    req = env.reset()[0]
    out = env.step({req["actor"]: json.dumps({"column": 3})})[0]
    assert out["actor"] == make_actor_id("player")


def test_snake_env() -> None:
    env = SnakeEnv(height=6, width=6, snake_length=3, seed=0)
    req = env.reset()[0]
    out = env.step({req["actor"]: json.dumps({"action": 0})})[0]
    assert out["actor"] == make_actor_id("player")


def test_python_code_env_exec() -> None:
    env = PythonCodeEnv()
    req = env.reset()[0]
    out = env.step({req["actor"]: json.dumps({"code": "print('hi')"})})[0]
    assert "stdout" in out["message"]["content"].lower()
    assert out["reward"] == 0.0


def test_git_env_requires_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    env = GitEnv()
    try:
        req = env.reset()[0]
        out = env.step({req["actor"]: json.dumps({"action_type": "execute_git_command", "command": "status"})})[0]
        assert out["reward"] == 0.0
    finally:
        env.close()


def test_websearch_env_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    env = WebSearchEnv()
    req = env.reset()[0]
    assert req["needs_action"] is False
    assert req["reward"] < 0.0


def test_dipg_env_dataset(tmp_path: Path) -> None:
    path = tmp_path / "dipg.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": "unused"},
            {"role": "user", "content": "context here\n\nquestion here"},
            {"role": "assistant", "content": json.dumps({"final": "42", "proof": "context here"})},
        ]
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    env = DipgSafetyEnv(dataset_path=str(path), seed=0)
    req = env.reset()[0]
    response = (
        "<|channel|>analysis<|message|>x<|end|>"
        "<|channel|>proof<|message|>context here<|end|>"
        "<|channel|>final<|message|>42<|end|>"
    )
    out = env.step({req["actor"]: response})[0]
    assert out["needs_action"] is False


def test_finrl_env_dummy() -> None:
    class DummyFinRLEnv:
        def __init__(self, **kwargs):
            self._t = 0

        def reset(self):
            self._t = 0
            return [0.0, 1.0], {}

        def step(self, action):
            self._t += 1
            return [float(self._t)], 0.5, self._t >= 2, False, {"a": action}

    env = FinRLEnv(finrl_env_class=DummyFinRLEnv, finrl_env_config={})
    req = env.reset()[0]
    out = env.step({req["actor"]: json.dumps({"actions": [0.1]})})[0]
    assert "reward=" in out["message"]["content"]


@pytest.mark.parametrize(
    ("module", "ctor", "kwargs"),
    [
        ("ale_py", "gyllm.envs.openenv.atari:AtariEnv", {}),
        ("open_spiel", "gyllm.envs.openenv.openspiel:OpenSpielEnv", {}),
        ("sumo_rl", "gyllm.envs.openenv.sumo_rl:SumoRLEnv", {"net_file": "net.net.xml", "route_file": "route.rou.xml"}),
        ("browsergym", "gyllm.envs.openenv.browsergym:BrowserGymEnv", {}),
        ("textarena", "gyllm.envs.openenv.textarena:TextArenaEnv", {}),
    ],
)
def test_optional_deps_missing_raise(module: str, ctor: str, kwargs: dict) -> None:
    """
    If a dependency is missing, the corresponding env constructor should raise ImportError.
    If present, skip (these envs are heavy/integration-style).
    """
    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"optional dependency {module!r} present; skipping integration env smoke test")

    mod, name = ctor.split(":")
    cls = getattr(__import__(mod, fromlist=[name]), name)
    with pytest.raises(ImportError):
        cls(**kwargs)
