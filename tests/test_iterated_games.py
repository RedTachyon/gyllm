from gyllm.envs.simple.iterated_games import IpdEnv, TftIpdEnv


def test_tft_ipd_single_round() -> None:
    env = TftIpdEnv(num_turns=1)
    req = env.reset()[0]
    out = env.step({req["actor"]: "A"})[0]
    assert out["needs_action"] is False
    assert out["episode_end"] is True
    assert out["reward"] > 0.0


def test_ipd_two_player_single_round() -> None:
    env = IpdEnv(num_turns=1)
    reqs = env.reset()
    actions = {r["actor"]: "A" for r in reqs}
    out = env.step(actions)
    assert len(out) == 2
    assert all(r["needs_action"] is False for r in out)
    assert all(r["episode_end"] is True for r in out)
