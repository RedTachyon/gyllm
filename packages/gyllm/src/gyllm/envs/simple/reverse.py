import random
from collections.abc import Sequence
from typing import Literal

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id

Mode = Literal["words", "random"]

_FALLBACK_WORDS: list[str] = [
    "about",
    "after",
    "again",
    "almost",
    "always",
    "animal",
    "answer",
    "around",
    "before",
    "better",
    "building",
    "camera",
    "change",
    "circle",
    "coffee",
    "common",
    "country",
    "create",
    "daughter",
    "decimal",
    "doctor",
    "double",
    "dragon",
    "dream",
    "energy",
    "family",
    "forest",
    "future",
    "garden",
    "golden",
    "hammer",
    "island",
    "jungle",
    "little",
    "machine",
    "market",
    "memory",
    "mirror",
    "moment",
    "music",
    "object",
    "orange",
    "paper",
    "people",
    "planet",
    "purple",
    "quick",
    "random",
    "river",
    "school",
    "simple",
    "snow",
    "spring",
    "summer",
    "system",
    "turtle",
    "window",
    "winter",
    "yellow",
]

_WORD_LIST_CACHE: list[str] | None = None


def _normalize_mode(mode: str) -> Mode:
    key = str(mode).lower()
    if key == "word":
        key = "words"
    if key not in ("words", "random"):
        raise ValueError(f"mode must be 'words' or 'random'; got {mode!r}")
    return key  # type: ignore[return-value]


def _sanitize_word_list(words: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for word in words:
        value = str(word).strip().lower()
        if not value or not value.isascii() or not value.isalpha():
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _load_word_list(max_words: int) -> list[str]:
    global _WORD_LIST_CACHE
    if _WORD_LIST_CACHE is not None:
        return list(_WORD_LIST_CACHE)

    words: list[str] = []
    try:
        import wordfreq  # type: ignore[import]

        top_n_list = getattr(wordfreq, "top_n_list", None)
        if callable(top_n_list):
            words = list(top_n_list("en", max_words))
    except Exception:
        words = []

    if not words:
        for path in ("/usr/share/dict/words", "/usr/dict/words"):
            try:
                with open(path, encoding="utf-8") as handle:
                    words = [line.strip() for line in handle if line.strip()]
                if words:
                    break
            except OSError:
                continue

    if not words:
        words = list(_FALLBACK_WORDS)

    _WORD_LIST_CACHE = _sanitize_word_list(words)
    return list(_WORD_LIST_CACHE)


def _normalize_alphabet(alphabet: str) -> str:
    value = str(alphabet).lower()
    cleaned: list[str] = []
    seen: set[str] = set()
    for ch in value:
        if not ch.isascii() or not ch.isalpha():
            continue
        if ch in seen:
            continue
        seen.add(ch)
        cleaned.append(ch)
    if not cleaned:
        raise ValueError("alphabet must include at least one lowercase letter")
    return "".join(cleaned)


class ReverseEnv(LLMEnv):
    """Reverse environment: the env provides a word, the agent reverses it."""

    agents: list[str] = ["agent"]

    def __init__(
        self,
        *,
        num_turns: int = 5,
        mode: Mode = "words",
        min_length: int = 3,
        max_length: int = 10,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
        seed: int | None = None,
        word_pool: Sequence[str] | None = None,
        max_words: int = 50000,
    ) -> None:
        """Initialize the Reverse env.

        Args:
            num_turns: Number of reverse turns per episode.
            mode: "words" for English word pool, "random" for random strings.
            min_length: Minimum word length (inclusive).
            max_length: Maximum word length (inclusive).
            alphabet: Allowed characters for random mode (letters only).
            seed: Optional RNG seed for reproducibility.
            word_pool: Optional override list of words for word mode.
            max_words: Maximum words to load when using the library/system pool.
        """
        super().__init__()
        if num_turns <= 0:
            raise ValueError(f"num_turns must be > 0; got {num_turns}")
        self.num_turns = int(num_turns)
        self.mode: Mode = _normalize_mode(mode)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        if self.min_length <= 0 or self.max_length <= 0 or self.max_length < self.min_length:
            raise ValueError(
                "min_length and max_length must be positive with max_length >= min_length; "
                f"got min_length={min_length} max_length={max_length}"
            )
        self._alphabet = _normalize_alphabet(alphabet)
        self._seed = seed
        self._rng = random.Random(seed)
        if word_pool is None:
            self._word_pool_source = None
        elif isinstance(word_pool, (str, bytes)):
            raise TypeError("word_pool must be a sequence of words, not a single string")
        else:
            self._word_pool_source = list(word_pool)
        self._word_pool: list[str] | None = None
        self._eligible_words: list[str] = []
        self._max_words = int(max_words)
        if self._max_words <= 0:
            raise ValueError(f"max_words must be > 0; got {max_words}")

        self._turns_completed = 0
        self._done = False
        self._current: str | None = None

        if self.mode == "words":
            self._ensure_word_pool()

    def _ensure_word_pool(self) -> None:
        if self._word_pool is None:
            if self._word_pool_source is not None:
                self._word_pool = _sanitize_word_list(self._word_pool_source)
            else:
                self._word_pool = _load_word_list(self._max_words)
        if not self._word_pool:
            raise ValueError("word_pool must contain at least one lowercase word")
        self._eligible_words = [word for word in self._word_pool if self.min_length <= len(word) <= self.max_length]
        if not self._eligible_words:
            raise ValueError(
                f"No words available in word_pool for the specified length range [{self.min_length}, {self.max_length}]"
            )

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the agent."""
        agent_id = self.agent_id(actor)
        if agent_id != "agent":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                "You are in Reverse.\n"
                "Each turn, the environment will send a lowercase word.\n"
                "Your task is to reply with the word reversed.\n"
                "Do not add extra words.\n"
                "Leading/trailing whitespace is ignored.\n"
            ),
        }

    def _next_message(self) -> str:
        """Generate the next word for the agent."""
        if self.mode == "random":
            length = self._rng.randint(self.min_length, self.max_length)
            return "".join(self._rng.choice(self._alphabet) for _ in range(length))
        if self.mode == "words":
            self._ensure_word_pool()
            return str(self._rng.choice(self._eligible_words))
        raise ActionError(f"Unknown mode: {self.mode!r}")

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and start a new episode."""
        options = options or {}
        if "num_turns" in options:
            num_turns = int(options["num_turns"])
            if num_turns <= 0:
                raise ValueError(f"num_turns must be > 0; got {num_turns}")
            self.num_turns = num_turns
        if "mode" in options:
            self.mode = _normalize_mode(str(options["mode"]))
        if "min_length" in options:
            self.min_length = int(options["min_length"])
        if "max_length" in options:
            self.max_length = int(options["max_length"])
        if self.min_length <= 0 or self.max_length <= 0 or self.max_length < self.min_length:
            raise ValueError(
                "min_length and max_length must be positive with max_length >= min_length; "
                f"got min_length={self.min_length} max_length={self.max_length}"
            )
        if "alphabet" in options:
            self._alphabet = _normalize_alphabet(str(options["alphabet"]))
        if "seed" in options:
            self._seed = options["seed"] if options["seed"] is None else int(options["seed"])
            self._rng = random.Random(self._seed)
        if "word_pool" in options:
            word_pool = options["word_pool"]
            if word_pool is None:
                self._word_pool_source = None
            elif isinstance(word_pool, (str, bytes)):
                raise TypeError("word_pool must be a sequence of words, not a single string")
            elif isinstance(word_pool, Sequence):
                self._word_pool_source = list(word_pool)
            else:
                raise TypeError("word_pool must be a sequence of strings or None")
            self._word_pool = None

        self._begin_episode()
        self._turns_completed = 0
        self._done = False

        if "message" in options:
            message = str(options["message"]).strip().lower()
            if not message or not message.isascii() or not message.isalpha():
                raise ValueError("message must be a lowercase alphabetic word")
            if not (self.min_length <= len(message) <= self.max_length):
                raise ValueError(
                    f"message length must fall within the configured range [{self.min_length}, {self.max_length}]"
                )
            self._current = message
        else:
            self._current = self._next_message()

        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("agent")),
                "message": {"role": "user", "content": self._current},
                "needs_action": True,
                "info": {
                    "turn": self._turns_completed,
                    "num_turns": self.num_turns,
                    "mode": self.mode,
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                },
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply an action and return the next request."""
        if self._done:
            return []
        if self._current is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        actions = self._normalize_actions(actions)
        expected = self._current[::-1]
        got = actions["agent"].strip()

        if got != expected:
            self._done = True
            requests: list[Request] = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": -1.0,
                    "message": {
                        "role": "user",
                        "content": f"Incorrect. expected={expected!r} got={got!r}",
                    },
                    "needs_action": False,
                    "info": {
                        "expected": expected,
                        "got": got,
                        "turn": self._turns_completed,
                    },
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = done
            return requests

        self._turns_completed += 1
        if self._turns_completed >= self.num_turns:
            self._done = True
            requests = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": 1.0,
                    "message": {"role": "user", "content": "Done."},
                    "needs_action": False,
                    "info": {"turn": self._turns_completed, "num_turns": self.num_turns},
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = done
            return requests

        self._current = self._next_message()
        requests = [
            {
                "actor": make_actor_id("agent"),
                "reward": 1.0,
                "message": {"role": "user", "content": self._current},
                "needs_action": True,
                "info": {
                    "turn": self._turns_completed,
                    "num_turns": self.num_turns,
                    "mode": self.mode,
                },
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
