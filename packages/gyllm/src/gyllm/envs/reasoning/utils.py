"""Utilities for reasoning-style QA datasets and verification."""

import json
import re
from pathlib import Path
from typing import Any, Protocol, TypedDict


class ReasoningProblem(TypedDict):
    """Normalized reasoning sample with question/answer text."""

    question: str
    answer: str
    cot: str


class Normalizer(Protocol):
    """Protocol for dataset-specific normalization."""

    def normalize(self, data: dict[str, Any]) -> ReasoningProblem:
        """Convert a raw record into a ReasoningProblem.

        Args:
            data: Raw dataset record.

        Returns:
            Normalized reasoning problem.
        """


class Gsm8kNormalizer(Normalizer):
    """Normalize GSM8K-style samples with 'question' and 'answer'."""

    def normalize(self, data: dict[str, Any]) -> ReasoningProblem:
        """Normalize a GSM8K record into question/answer/cot fields.

        Args:
            data: Raw GSM8K record.

        Returns:
            Normalized reasoning problem.
        """
        question = str(data["question"])
        solution = str(data["answer"])
        if "#### " in solution:
            cot, answer = solution.split("#### ", 1)
        else:
            cot, answer = "", solution
        return {"question": question, "answer": answer.strip(), "cot": cot.strip()}


class SimpleQANormalizer(Normalizer):
    """Normalize generic QA samples with 'question' and 'answer'."""

    def normalize(self, data: dict[str, Any]) -> ReasoningProblem:
        """Normalize a QA record with optional chain-of-thought.

        Args:
            data: Raw QA record.

        Returns:
            Normalized reasoning problem.
        """
        return {
            "question": str(data["question"]),
            "answer": str(data["answer"]),
            "cot": str(data.get("cot", "")),
        }


class MathNormalizer(Normalizer):
    """
    Normalize the MATH dataset.

    TODO: verify the parsing logic.
    """

    def normalize(self, data: dict[str, str]) -> ReasoningProblem:
        """Normalize a MATH sample into question/answer/cot fields."""
        import regex  # type: ignore[import]

        pattern = r"\\boxed{((?:[^{}]+|\{(?1)\})*)}"
        question = str(data["problem"])
        solution = str(data["solution"])
        matches = regex.findall(pattern, solution)
        answer = matches[-1] if matches else ""
        cot = solution.split(r"\boxed")[0].strip() if r"\boxed" in solution else ""
        return {"question": question, "answer": answer, "cot": cot}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON into a list of dicts.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of JSON objects.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_answer_tags(text: str) -> list[str]:
    """Return all <answer>...</answer> contents from a completion.

    Args:
        text: Completion text to parse.

    Returns:
        List of extracted answer strings.
    """
    return [m.strip() for m in re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)]


def normalize_answer(text: str) -> str:
    """Normalize whitespace for answer comparison.

    Args:
        text: Raw answer string.

    Returns:
        Normalized answer string.
    """
    return re.sub(r"\s+", " ", text.strip())


def verify_answer(gold: str, completion: str, *, use_math_verify: bool = False) -> bool:
    """Verify the completion against gold using <answer> tags.

    Args:
        gold: Expected answer text.
        completion: Model completion containing answer tags.
        use_math_verify: Whether to use math_verify for equivalence.

    Returns:
        True if the completion matches the gold answer.
    """
    answers = extract_answer_tags(completion)
    if not answers:
        return False
    candidate = answers[-1]
    if use_math_verify:
        from math_verify import parse, verify  # type: ignore[import]

        return bool(verify(parse(gold), parse(candidate)))
    return normalize_answer(candidate) == normalize_answer(gold)
