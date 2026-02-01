import json

import pytest

from gyllm.core import ActionError, make_actor_id
from gyllm.envs.simple.tic_tac_toe import TicTacToeEnv


def test_tictactoe_two_player_orchestration_and_win() -> None:
    env = TicTacToeEnv(opponent=None)

    reqs = env.reset()
    assert len(reqs) == 1
    assert reqs[0]["actor"] == make_actor_id("player_a")
    assert reqs[0]["needs_action"] is True

    # A: top-left (0)
    reqs = env.step({make_actor_id("player_a"): "0"})
    assert len(reqs) == 1
    assert reqs[0]["actor"] == make_actor_id("player_b")
    assert reqs[0]["needs_action"] is True

    # B: middle-left (3)
    reqs = env.step({make_actor_id("player_b"): json.dumps({"row": 1, "col": 0})})
    assert len(reqs) == 1
    assert reqs[0]["actor"] == make_actor_id("player_a")
    assert reqs[0]["needs_action"] is True

    # A: top-middle (1)
    reqs = env.step({make_actor_id("player_a"): json.dumps({"index": 1})})
    assert len(reqs) == 1
    assert reqs[0]["actor"] == make_actor_id("player_b")
    assert reqs[0]["needs_action"] is True

    # B: center (4)
    reqs = env.step({make_actor_id("player_b"): "4"})
    assert len(reqs) == 1
    assert reqs[0]["actor"] == make_actor_id("player_a")
    assert reqs[0]["needs_action"] is True

    # A: top-right (2) -> win for player_a
    out = env.step({make_actor_id("player_a"): "2"})
    assert len(out) == 2
    assert {r["actor"] for r in out} == {make_actor_id("player_a"), make_actor_id("player_b")}
    assert all(r["needs_action"] is False for r in out)
    rewards = {r["actor"]: r["reward"] for r in out}
    assert rewards[make_actor_id("player_a")] == 1.0
    assert rewards[make_actor_id("player_b")] == -1.0


def test_tictactoe_two_player_missing_current_action_raises() -> None:
    env = TicTacToeEnv(opponent=None)
    env.reset()
    with pytest.raises(ActionError):
        env.step({make_actor_id("player_b"): "0"})


def test_tictactoe_two_player_occupied_cell_ends_game() -> None:
    env = TicTacToeEnv(opponent=None)
    env.reset()

    env.step({make_actor_id("player_a"): "0"})
    env.step({make_actor_id("player_b"): "3"})

    # A tries to play into B's occupied cell -> invalid -> B wins.
    out = env.step({make_actor_id("player_a"): "3"})
    assert len(out) == 2
    assert all(r["needs_action"] is False for r in out)
    rewards = {r["actor"]: r["reward"] for r in out}
    assert rewards[make_actor_id("player_a")] == -1.0
    assert rewards[make_actor_id("player_b")] == 1.0


def test_tictactoe_single_agent_random_opponent() -> None:
    env = TicTacToeEnv(opponent="random", seed=0)
    req = env.reset()[0]
    assert req["actor"] == make_actor_id("player")

    out = env.step({req["actor"]: "0"})
    assert len(out) == 1
    assert out[0]["actor"] == make_actor_id("player")

    flat = [cell for row in env.board for cell in row]
    assert flat.count(1) == 1
    assert flat.count(-1) == 1


def test_tictactoe_single_agent_invalid_action_loses() -> None:
    env = TicTacToeEnv(opponent="random", seed=0)
    req = env.reset()[0]
    out = env.step({req["actor"]: "9"})
    assert len(out) == 1
    assert out[0]["needs_action"] is False
    assert out[0]["reward"] == -1.0


def test_tictactoe_single_agent_invalid_action_can_repeat() -> None:
    env = TicTacToeEnv(opponent="random", seed=0, repeat_invalid_action=True)
    req = env.reset()[0]
    with pytest.raises(ActionError):
        env.step({req["actor"]: "9"})
