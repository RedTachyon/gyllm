import pytest

from gyllm.envs.simple.reverse_echo import MessageKind, ReverseEcho


@pytest.mark.parametrize("message_kind", ["word", "number", "uuid"])
def test_reverse_echo_happy_path(message_kind: MessageKind) -> None:
    env = ReverseEcho(num_turns=3, message_kind=message_kind, seed=0)
    req = env.reset()[0]
    actor = req["actor"]

    for i in range(3):
        expected = req["message"]["content"]
        out = env.step({actor: expected})[0]
        assert out["reward"] == 1.0
        req = out

        if i < 2:
            assert out["needs_action"] is True
        else:
            assert out["needs_action"] is False
            assert out["message"]["content"] == "Done."

    assert env.step({actor: "anything"}) == []


def test_reverse_echo_incorrect_ends_episode() -> None:
    env = ReverseEcho(num_turns=5, message_kind="word", seed=0)
    req = env.reset()[0]
    actor = req["actor"]

    out = env.step({actor: "wrong"})[0]
    assert out["reward"] == -1.0
    assert out["needs_action"] is False
    assert "Incorrect." in out["message"]["content"]
    assert env.step({actor: "still wrong"}) == []


def test_reverse_echo_invalid_message_kind() -> None:
    with pytest.raises(ValueError):
        ReverseEcho(num_turns=1, message_kind="nope")  # type: ignore[arg-type]
