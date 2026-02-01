import pytest

from gyllm.envs.simple.reverse import ReverseEnv


def test_reverse_env_words_mode() -> None:
    env = ReverseEnv(num_turns=2, mode="words", word_pool=["apple", "banana"], seed=0)
    req = env.reset()[0]
    actor = req["actor"]
    word = req["message"]["content"]
    assert word in {"apple", "banana"}

    out = env.step({actor: word[::-1]})[0]
    assert out["reward"] == 1.0
    if out["needs_action"]:
        next_word = out["message"]["content"]
        out = env.step({actor: next_word[::-1]})[0]
    assert out["needs_action"] is False
    assert out["message"]["content"] == "Done."


def test_reverse_env_random_mode_length() -> None:
    env = ReverseEnv(num_turns=1, mode="random", min_length=4, max_length=4, seed=1)
    req = env.reset()[0]
    actor = req["actor"]
    word = req["message"]["content"]
    assert len(word) == 4
    assert word.islower()
    assert word.isalpha()

    out = env.step({actor: word[::-1]})[0]
    assert out["reward"] == 1.0
    assert out["needs_action"] is False


def test_reverse_env_incorrect_ends_episode() -> None:
    env = ReverseEnv(num_turns=3, mode="random", min_length=3, max_length=3, seed=2)
    req = env.reset()[0]
    actor = req["actor"]

    out = env.step({actor: "wrong"})[0]
    assert out["reward"] == -1.0
    assert out["needs_action"] is False
    assert "Incorrect." in out["message"]["content"]
    assert env.step({actor: "still wrong"}) == []


def test_reverse_env_invalid_mode() -> None:
    with pytest.raises(ValueError):
        ReverseEnv(mode="nope")  # type: ignore[arg-type]
