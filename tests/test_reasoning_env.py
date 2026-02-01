from datasets import Dataset

from gyllm.envs.reasoning import Gsm8kNormalizer, SimpleQANormalizer
from gyllm.envs.reasoning.qa import ReasoningQAEnv


def test_reasoning_env_batching_and_repeats() -> None:
    dataset = Dataset.from_list(
        [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
            {"question": "q3", "answer": "a3"},
        ]
    )
    env = ReasoningQAEnv(
        dataset=dataset,
        normalizer=SimpleQANormalizer(),
        problems_per_step=2,
        repetitions_per_problem=3,
    )
    reqs = env.reset()
    assert len(reqs) == 6
    assert [req["message"]["content"] for req in reqs[:3]] == ["q1", "q1", "q1"]
    assert [req["message"]["content"] for req in reqs[3:]] == ["q2", "q2", "q2"]
    assert reqs[0]["info"]["question_id"] == 0
    assert reqs[3]["info"]["question_id"] == 1

    actions = {pending.actor: f"<answer>{pending.problem['answer']}</answer>" for pending in env._pending}
    out = env.step(actions)
    assert len(out) == 6
    assert all(req["reward"] == 1.0 for req in out)
    assert all(req["needs_action"] is False for req in out)
    assert all(req["episode_end"] is True for req in out)

    reqs2 = env.reset()
    assert [req["message"]["content"] for req in reqs2[:3]] == ["q3", "q3", "q3"]
    assert [req["message"]["content"] for req in reqs2[3:]] == ["q1", "q1", "q1"]
    assert reqs2[0]["info"]["question_id"] == 2
    assert reqs2[3]["info"]["question_id"] == 0
    env.close()


def test_reasoning_env_gsm8k_dataset() -> None:
    env = ReasoningQAEnv(
        dataset_path="openai/gsm8k",
        dataset_kwargs={"name": "main", "split": "train"},
        normalizer=Gsm8kNormalizer(),
        problems_per_step=1,
        repetitions_per_problem=1,
    )
    req = env.reset()[0]
    assert "question_id" in req["info"]
    action = f"<answer>{env._pending[0].problem['answer']}</answer>"
    out = env.step({req["actor"]: action})[0]
    assert out["reward"] == 1.0
    assert out["needs_action"] is False
    assert out["episode_end"] is True
    env.close()
