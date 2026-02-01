"""Dataset-backed reasoning environment."""

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.reasoning.utils import Gsm8kNormalizer, MathNormalizer, Normalizer, ReasoningProblem, verify_answer

_MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass
class _PendingProblem:
    actor: ActorId
    problem: ReasoningProblem
    problem_index: int
    episode_id: int


class ReasoningQAEnv(LLMEnv):
    """Single-turn QA env backed by a reasoning dataset."""

    agents: list[str] = ["agent"]

    def __init__(
        self,
        *,
        dataset_id: str | None = None,
        dataset_path: str | None = None,
        dataset_config: str | None = None,
        dataset_split: str = "train",
        dataset_kwargs: dict[str, Any] | None = None,
        dataset: Any | None = None,
        normalizer: Normalizer | None = None,
        problems_per_step: int = 1,
        repetitions_per_problem: int = 1,
        use_math_verify: bool = False,
        reveal_answer: bool = False,
    ) -> None:
        """Create a single-turn QA env from a dataset or in-memory samples.

        Args:
            dataset_id: HuggingFace dataset id or local path for load_dataset.
            dataset_path: Deprecated alias for dataset_id.
            dataset_config: Optional dataset config/name for load_dataset.
            dataset_split: Dataset split for load_dataset.
            dataset_kwargs: Extra kwargs forwarded to load_dataset.
            dataset: Pre-loaded dataset to use instead of load_dataset.
            normalizer: Optional dataset row normalizer.
            problems_per_step: Number of distinct problems per reset.
            repetitions_per_problem: Number of repetitions per problem.
            use_math_verify: Whether to use math-verify for answers.
            reveal_answer: Whether to reveal the correct answer on errors.
        """
        super().__init__()
        if use_math_verify:
            try:
                import math_verify  # noqa: F401  # type: ignore[import]
            except Exception as exc:  # pragma: no cover
                raise ImportError("ReasoningQAEnv requires math-verify when use_math_verify=True") from exc

        if dataset_id is not None and dataset_path is not None:
            raise ValueError("Use either dataset_id or dataset_path, not both.")
        if dataset_id is None and dataset_path is not None:
            dataset_id = dataset_path

        if dataset is not None and dataset_id is not None:
            raise ValueError("Provide either dataset or dataset_id, not both.")

        if dataset is None:
            if dataset_id is None:
                raise ValueError("dataset_id must be provided when dataset is None.")
            from datasets import load_dataset

            dataset_kwargs = dict(dataset_kwargs or {})
            if "name" in dataset_kwargs and dataset_config is None:
                dataset_config = dataset_kwargs.pop("name")
            else:
                dataset_kwargs.pop("name", None)
            if "split" in dataset_kwargs and dataset_split == "train":
                dataset_split = dataset_kwargs.pop("split")
            else:
                dataset_kwargs.pop("split", None)

            if _is_math_dataset_id(dataset_id):
                dataset = _load_math_dataset(
                    dataset_id,
                    dataset_config=dataset_config,
                    dataset_split=dataset_split,
                    dataset_kwargs=dataset_kwargs,
                )
            else:
                dataset = load_dataset(
                    dataset_id,
                    name=dataset_config,
                    split=dataset_split,
                    **dataset_kwargs,
                )

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            if dataset_split not in dataset:
                raise ValueError(f"Dataset split {dataset_split!r} not found.")
            dataset = dataset[dataset_split]

        if normalizer is None:
            normalizer = _infer_normalizer(dataset_id)
        if normalizer is None:
            raise ValueError("normalizer must be provided unless dataset_id is a supported preset.")

        rows = list(dataset)
        self._problems = [normalizer.normalize(row) for row in rows]
        self._index = 0
        self._problems_per_step = problems_per_step
        self._repetitions_per_problem = repetitions_per_problem
        self._use_math_verify = use_math_verify
        self._reveal_answer = reveal_answer
        self._pending: list[_PendingProblem] = []

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the reasoning agent.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id not in self.agent_ids:
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                "Solve the user's question. Provide your final answer inside <answer></answer> tags. "
                "Inside the answer tag, include only the final answer."
            ),
        }

    def _next_problem(self) -> tuple[int, ReasoningProblem]:
        """Return the next problem, cycling through the dataset.

        Returns:
            Tuple of dataset index and reasoning problem.
        """
        index = self._index
        problem = self._problems[index]
        self._index = (self._index + 1) % len(self._problems)
        return index, problem

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Start a new single-turn episode with the next question.

        Args:
            options: Optional reset overrides for question selection.

        Returns:
            Initial request for the new episode.
        """
        del options
        self._pending = []
        requests: list[Request] = []
        for _ in range(self._problems_per_step):
            problem_index, problem = self._next_problem()
            for _ in range(self._repetitions_per_problem):
                episode_id = self._begin_episode()
                actor = make_actor_id("agent", episode_id=str(episode_id))
                self._pending.append(
                    _PendingProblem(
                        actor=actor,
                        problem=problem,
                        problem_index=problem_index,
                        episode_id=episode_id,
                    )
                )
                requests.append(
                    {
                        "actor": actor,
                        "reward": 0.0,
                        "system_message": self._system_message(actor),
                        "message": {"role": "user", "content": problem["question"]},
                        "needs_action": True,
                        "info": {
                            "question_id": problem_index,
                            "question": problem["question"],
                        },
                        "episode_id": episode_id,
                        "group_id": problem_index,
                        "episode_start": True,
                        "episode_end": False,
                    }
                )
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Score the agent response and end the episode.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Terminal request with reward and feedback.
        """
        if not self._pending:
            return []
        expected = {pending.actor for pending in self._pending}
        missing = expected - set(actions)
        if missing:
            raise ActionError(f"Missing actions for: {sorted(missing)!r}")
        unexpected = set(actions) - expected
        if unexpected:
            raise ActionError(f"Unexpected actions for: {sorted(unexpected)!r}")

        responses: list[Request] = []
        for pending in self._pending:
            completion = actions[pending.actor]
            correct = verify_answer(
                pending.problem["answer"],
                completion,
                use_math_verify=self._use_math_verify,
            )
            reward = 1.0 if correct else 0.0
            if correct:
                content = "Correct."
            else:
                content = "Incorrect."
                if self._reveal_answer:
                    content = f"{content} Expected: {pending.problem['answer']}"

            responses.append(
                {
                    "actor": pending.actor,
                    "reward": reward,
                    "message": {"role": "user", "content": content},
                    "needs_action": False,
                    "info": {
                        "question_id": pending.problem_index,
                        "question": pending.problem["question"],
                        "answer": pending.problem["answer"],
                        "correct": correct,
                    },
                    "episode_id": pending.episode_id,
                    "group_id": pending.problem_index,
                    "episode_start": False,
                    "episode_end": True,
                }
            )

        self._pending = []
        return responses


def _infer_normalizer(dataset_id: str | None) -> Normalizer | None:
    if dataset_id is None:
        return None
    dataset_key = dataset_id.lower()
    if dataset_key in {"gsm8k", "openai/gsm8k"} or dataset_key.endswith("/gsm8k"):
        return Gsm8kNormalizer()
    if _is_math_dataset_id(dataset_key):
        return MathNormalizer()
    return None


def _is_math_dataset_id(dataset_id: str) -> bool:
    dataset_key = dataset_id.lower()
    return dataset_key in {"eleutherai/hendrycks_math", "hendrycks_math"} or dataset_key.endswith("/hendrycks_math")


def _load_math_dataset(
    dataset_id: str,
    *,
    dataset_config: str | None,
    dataset_split: str,
    dataset_kwargs: dict[str, Any],
) -> Any:
    from datasets import concatenate_datasets, load_dataset

    if dataset_config:
        dataset = load_dataset(
            dataset_id,
            name=dataset_config,
            split=dataset_split,
            **dataset_kwargs,
        )
        if not isinstance(dataset, (Dataset, IterableDataset)):
            raise TypeError("Expected a Dataset for MATH with split specified.")
        return dataset

    first = load_dataset(
        dataset_id,
        name=_MATH_SUBSETS[0],
        split=dataset_split,
        **dataset_kwargs,
    )
    if isinstance(first, IterableDataset):
        subsets_iter: list[IterableDataset] = [first]
        for subset in _MATH_SUBSETS[1:]:
            dataset = load_dataset(
                dataset_id,
                name=subset,
                split=dataset_split,
                **dataset_kwargs,
            )
            if not isinstance(dataset, IterableDataset):
                raise TypeError("Expected an IterableDataset for MATH with split specified.")
            subsets_iter.append(dataset)
        return concatenate_datasets(subsets_iter)

    if not isinstance(first, Dataset):
        raise TypeError("Expected a Dataset for MATH with split specified.")
    subsets_ds: list[Dataset] = [first]
    for subset in _MATH_SUBSETS[1:]:
        dataset = load_dataset(
            dataset_id,
            name=subset,
            split=dataset_split,
            **dataset_kwargs,
        )
        if not isinstance(dataset, Dataset):
            raise TypeError("Expected a Dataset for MATH with split specified.")
        subsets_ds.append(dataset)
    return concatenate_datasets(subsets_ds)
