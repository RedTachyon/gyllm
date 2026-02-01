import random
from typing import Any, ClassVar

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id

IPD_SYSTEM_PROMPT = (
    """You are participating in a repeated two-player game-theoretic scenario.\n"""
    "Each round you must choose action A or action B. The payoff matrix (your score shown first) is:\n\n"
    "                 Opponent\n"
    "               A          B\n"
    "  You A     (3, 3)     (0, 5)\n"
    "  You B     (5, 0)     (1, 1)\n\n"
    "Your objective is to maximise your cumulative score across rounds.\n"
    "Before selecting an action, reason step by step about the history of play so far to decide optimally.\n"
    "After that, respond with exactly A or B, nothing else."
)


class TftIpdEnv(LLMEnv):
    """Iterated Prisoner's Dilemma where the built‑in opponent plays tit‑for‑tat."""

    agents: list[str] = ["player"]

    # Payoff matrix for (player_action, opponent_action)
    PAYOFFS: ClassVar[dict[tuple[str, str], tuple[int, int]]] = {
        ("A", "A"): (3, 3),
        ("A", "B"): (0, 5),
        ("B", "A"): (5, 0),
        ("B", "B"): (1, 1),
    }

    def __init__(self, num_turns: int = 5) -> None:
        """Initialize the environment.

        Args:
            num_turns: Number of rounds per episode.
        """
        super().__init__()
        self.num_turns = num_turns
        self.history: list[tuple[str, str]] = []
        self.messages: list[Message] = []
        self.player_score: int = 0
        self.opponent_score: int = 0
        self.system_prompt = IPD_SYSTEM_PROMPT

        # self.system_prompt = (
        #     """You are participating in a repeated two-player game-theoretic scenario.\n"""
        #     "Each round you must choose action A or action B. The payoff matrix (your score shown first) is:\n\n"
        #     "                 Opponent\n"
        #     "               A          B\n"
        #     "  You A     (3, 3)     (0, 5)\n"
        #     "  You B     (5, 0)     (1, 1)\n\n"
        #     "Your objective is to maximise your cumulative score across rounds.\n"
        #     "Before selecting an action, reason step by step about the history of play so far to decide optimally.\n"
        #     "After that, finish your response with either 'ACTION: A' or 'ACTION: B'."
        # )

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _tit_for_tat(self) -> str:
        """Return the opponent's tit-for-tat action.

        Returns:
            Opponent action ("A" or "B").
        """
        return "A" if len(self.history) == 0 else self.history[-1][0]

    # ------------------------------------------------------------------
    # Public API methods required by LLMEnv
    # ------------------------------------------------------------------
    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and produce the first request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request for the new episode.
        """
        options = options or {}
        if "num_turns" in options:
            num_turns = int(options["num_turns"])
            if num_turns <= 0:
                raise ValueError(f"num_turns must be > 0; got {num_turns}")
            self.num_turns = num_turns
        if "system_prompt" in options:
            self.system_prompt = str(options["system_prompt"])
        self._begin_episode()
        self.history.clear()
        self.player_score = 0
        self.opponent_score = 0
        start_msg: Message = {
            "role": "user",
            "content": "Starting round 1. You have 0 points. Your opponent has 0 points.",
        }
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            start_msg,
        ]
        requests: list[Request] = [
            {
                "actor": make_actor_id("player"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("player")),
                "message": start_msg,
                "needs_action": True,
                "info": {"round": 1, "player_score": 0, "opponent_score": 0},
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the player.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "player":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {"role": "system", "content": self.system_prompt}

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Advance one round given the agent's proposed action.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after the round.
        """
        if len(self.history) >= self.num_turns:
            return []

        actions = self._normalize_actions(actions)
        player_action = actions["player"]
        if player_action not in {"A", "B"}:
            raise ActionError("Action must be 'A' or 'B'")

        # Opponent responds according to tit-for-tat.
        opponent_action = self._tit_for_tat()

        # Determine payoffs
        player_reward, opponent_reward = self.PAYOFFS[(player_action, opponent_action)]
        self.player_score += player_reward
        self.opponent_score += opponent_reward

        # Log the round
        self.history.append((player_action, opponent_action))
        round_idx = len(self.history)  # 1‑based index

        game_over = round_idx >= self.num_turns
        if game_over:
            env_message = (
                f"End of round {round_idx}. You played {player_action}, opponent played {opponent_action}. "
                f"Round payoff: {player_reward}. Final scores – You: {self.player_score}, "
                f"Opponent: {self.opponent_score}. Game over."
            )
        else:
            env_message = (
                f"End of round {round_idx}. You played {player_action}, opponent played {opponent_action}. "
                f"Round payoff: {player_reward}. Total scores – You: {self.player_score}, "
                f"Opponent: {self.opponent_score}. Starting round {round_idx + 1}."
            )

        self.messages.append({"role": "assistant", "content": actions["player"]})
        env_msg: Message = {"role": "user", "content": env_message}
        self.messages.append(env_msg)

        # Emit the next request for the agent id.
        requests: list[Request] = [
            {
                "actor": make_actor_id("player"),
                "reward": float(player_reward),
                "message": env_msg,
                "needs_action": not game_over,
                "info": {
                    "round": round_idx,
                    "player_action": player_action,
                    "opponent_action": opponent_action,
                    "player_score": self.player_score,
                    "opponent_score": self.opponent_score,
                },
            }
        ]
        done = bool(game_over)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests

    # ------------------------------------------------------------------
    # Convenience methods (optional)
    # ------------------------------------------------------------------
    def render_history(self) -> list[Message]:
        """Return a copy of the current conversation log.

        Returns:
            Message history list.
        """
        return self.messages.copy()


class IpdEnv(LLMEnv):
    """Two player Iterated Prisoner's Dilemma."""

    agents: list[str] = ["player_a", "player_b"]

    PAYOFFS: ClassVar[dict[tuple[str, str], tuple[int, int]]] = TftIpdEnv.PAYOFFS

    def __init__(self, num_turns: int = 5) -> None:
        """Initialize the environment.

        Args:
            num_turns: Number of rounds per episode.
        """
        super().__init__()
        self.num_turns = num_turns
        self.history: list[tuple[str, str]] = []
        self.messages: dict[str, list[Message]] = {"player_a": [], "player_b": []}
        self.scores: dict[str, int] = {"player_a": 0, "player_b": 0}
        self.system_prompt = IPD_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and produce initial requests.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial requests for both players.
        """
        options = options or {}
        if "num_turns" in options:
            num_turns = int(options["num_turns"])
            if num_turns <= 0:
                raise ValueError(f"num_turns must be > 0; got {num_turns}")
            self.num_turns = num_turns
        if "system_prompt" in options:
            self.system_prompt = str(options["system_prompt"])
        self._begin_episode()
        self.history.clear()
        self.scores = {"player_a": 0, "player_b": 0}
        start_msg: Message = {
            "role": "user",
            "content": "Starting round 1. You have 0 points. Your opponent has 0 points.",
        }
        for agent_id in self.agent_ids:
            system_msg: Message = {"role": "system", "content": self.system_prompt}
            self.messages[agent_id] = [
                system_msg,
                start_msg,
            ]
        requests: list[Request] = []
        for agent_id in self.agent_ids:
            request: Request = {
                "actor": make_actor_id(agent_id),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id(agent_id)),
                "message": start_msg,
                "needs_action": True,
                "info": {"round": 1, "scores": dict(self.scores)},
            }
            requests.append(request)
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for a player.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id not in set(self.agent_ids):
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {"role": "system", "content": self.system_prompt}

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Advance one round given the players' actions.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after the round.
        """
        if len(self.history) >= self.num_turns:
            return []

        actions = self._normalize_actions(actions)
        a_action = actions["player_a"]
        b_action = actions["player_b"]
        if a_action not in {"A", "B"} or b_action not in {"A", "B"}:
            raise ActionError("Action must be 'A' or 'B'")

        a_reward, b_reward = self.PAYOFFS[(a_action, b_action)]
        self.scores["player_a"] += a_reward
        self.scores["player_b"] += b_reward

        self.history.append((a_action, b_action))
        round_idx = len(self.history)

        game_over = round_idx >= self.num_turns
        env_text_by_agent: dict[str, str] = {}
        for name, my_action, opp_action, reward in [
            ("player_a", a_action, b_action, a_reward),
            ("player_b", b_action, a_action, b_reward),
        ]:
            if game_over:
                env_msg = (
                    f"End of round {round_idx}. You played {my_action}, opponent played {opp_action}. "
                    f"Round payoff: {reward}. Final scores – You: {self.scores[name]}, "
                    f"Opponent: {self.scores['player_b' if name == 'player_a' else 'player_a']}. Game over."
                )
            else:
                env_msg = (
                    f"End of round {round_idx}. You played {my_action}, opponent played {opp_action}. "
                    f"Round payoff: {reward}. Total scores – You: {self.scores[name]}, "
                    f"Opponent: {self.scores['player_b' if name == 'player_a' else 'player_a']}. Starting round {round_idx + 1}."
                )
            env_text_by_agent[name] = env_msg

        for agent_id in self.agent_ids:
            self.messages[agent_id].append({"role": "assistant", "content": actions[agent_id]})
            self.messages[agent_id].append({"role": "user", "content": env_text_by_agent[agent_id]})

        requests: list[Request] = []
        for name, reward in [
            ("player_a", a_reward),
            ("player_b", b_reward),
        ]:
            msg: Message = {"role": "user", "content": env_text_by_agent[name]}
            request: Request = {
                "actor": make_actor_id(name),
                "reward": float(reward),
                "message": msg,
                "needs_action": not game_over,
                "info": {
                    "round": round_idx,
                    "player_a_action": a_action,
                    "player_b_action": b_action,
                    "scores": dict(self.scores),
                },
            }
            requests.append(request)
        done = bool(game_over)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests

    def render_history(self) -> dict[str, list[Message]]:
        """Return copies of each player's conversation log.

        Returns:
            Mapping of agent id to message history list.
        """
        return {name: msgs.copy() for name, msgs in self.messages.items()}


class MatrixGameEnv(LLMEnv):
    """Configurable 2x2 matrix game with flexible termination."""

    agents: list[str] = ["player_a", "player_b"]

    # Action labels for the two-player 2x2 game.
    ACTIONS: ClassVar[tuple[str, str]] = ("A", "B")
    # Default to the IPD payoff matrix when no custom payoffs are provided.
    DEFAULT_PAYOFFS: ClassVar[dict[tuple[str, str], tuple[int, int]]] = TftIpdEnv.PAYOFFS

    def __init__(
        self,
        *,
        payoff_matrix: dict[tuple[str, str], tuple[float, float]] | list[list[list[float]]] | None = None,
        termination: str = "fixed",
        num_turns: int = 5,
        termination_prob: float = 0.1,
        max_turns: int = 25,
        seed: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize a matrix game environment.

        Args:
            payoff_matrix: Payoffs for (player_a_action, player_b_action).
            termination: "single", "fixed", or "random".
            num_turns: Number of rounds for fixed termination.
            termination_prob: Per-round termination probability for random termination.
            max_turns: Hard cap on rounds for random termination.
            seed: Optional RNG seed.
            system_prompt: Optional override for the system prompt.
        """
        super().__init__()
        self._rng = random.Random(seed)
        self._seed = seed
        self.payoffs = self._normalize_payoffs(payoff_matrix)
        self.termination = str(termination)
        self.num_turns = int(num_turns)
        self.termination_prob = float(termination_prob)
        self.max_turns = int(max_turns)
        self._system_prompt_override = system_prompt
        self._validate_termination()
        self._refresh_system_prompt()

        self.history: list[tuple[str, str]] = []
        self.messages: dict[str, list[Message]] = {"player_a": [], "player_b": []}
        self.scores: dict[str, float] = {"player_a": 0.0, "player_b": 0.0}
        self._done = False

    def _normalize_payoffs(
        self,
        payoff_matrix: dict[tuple[str, str], tuple[float, float]] | list[list[list[float]]] | None,
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """Normalize payoff configuration to a dict keyed by action pairs."""
        if payoff_matrix is None:
            return {k: (float(v[0]), float(v[1])) for k, v in self.DEFAULT_PAYOFFS.items()}
        if isinstance(payoff_matrix, dict):
            expected = {(a, b) for a in self.ACTIONS for b in self.ACTIONS}
            if set(payoff_matrix.keys()) != expected:
                raise ValueError(f"payoff_matrix must include keys {sorted(expected)}")
            out: dict[tuple[str, str], tuple[float, float]] = {}
            for key, value in payoff_matrix.items():
                out[key] = self._parse_payoff(value, f"payoff_matrix[{key!r}]")
            return out
        if isinstance(payoff_matrix, list):
            if len(payoff_matrix) != len(self.ACTIONS):
                raise ValueError("payoff_matrix must be 2x2 for actions A/B.")
            out = {}
            for row_idx, row in enumerate(payoff_matrix):
                if not isinstance(row, list) or len(row) != len(self.ACTIONS):
                    raise ValueError("payoff_matrix must be 2x2 for actions A/B.")
                for col_idx, cell in enumerate(row):
                    key = (self.ACTIONS[row_idx], self.ACTIONS[col_idx])
                    out[key] = self._parse_payoff(cell, f"payoff_matrix[{row_idx}][{col_idx}]")
            return out
        raise ValueError("payoff_matrix must be a dict or 2x2 list.")

    @staticmethod
    def _parse_payoff(value: Any, label: str) -> tuple[float, float]:
        """Parse a payoff cell into a numeric pair."""
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} must be a pair of numbers.")
        return (float(value[0]), float(value[1]))

    def _validate_termination(self) -> None:
        """Validate termination settings."""
        if self.termination not in {"single", "fixed", "random"}:
            raise ValueError(f"termination must be 'single', 'fixed', or 'random'; got {self.termination!r}")
        if self.termination == "fixed" and self.num_turns <= 0:
            raise ValueError(f"num_turns must be > 0; got {self.num_turns}")
        if self.termination == "random":
            if self.termination_prob < 0 or self.termination_prob > 1:
                raise ValueError("termination_prob must be between 0 and 1.")
            if self.max_turns <= 0:
                raise ValueError(f"max_turns must be > 0; got {self.max_turns}")

    @staticmethod
    def _format_number(value: float) -> str:
        """Format numbers without trailing .0 when they are integers."""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _format_payoff(self, payoff: tuple[float, float]) -> str:
        """Format a payoff tuple for prompt display."""
        left, right = payoff
        return f"({self._format_number(left)}, {self._format_number(right)})"

    def _build_matrix_text(self) -> str:
        """Render the payoff matrix as a text table."""
        cells = {(a, b): self._format_payoff(self.payoffs[(a, b)]) for a in self.ACTIONS for b in self.ACTIONS}
        col_width = max(10, max(len(cell) for cell in cells.values()) + 2)
        a_action, b_action = self.ACTIONS
        lines = [
            "                 Opponent",
            f"               {a_action:<{col_width}}{b_action:<{col_width}}",
            f"  You {a_action:<3} {cells[(a_action, a_action)]:<{col_width}}{cells[(a_action, b_action)]:<{col_width}}",
            f"  You {b_action:<3} {cells[(b_action, a_action)]:<{col_width}}{cells[(b_action, b_action)]:<{col_width}}",
        ]
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the default system prompt based on current settings."""
        matrix_text = self._build_matrix_text()
        if self.termination == "single":
            horizon = "The game lasts for a single round."
        elif self.termination == "fixed":
            horizon = f"The game lasts for {self.num_turns} rounds."
        else:
            horizon = (
                "The game ends after each round with probability "
                f"{self.termination_prob:.2f} (max {self.max_turns} rounds)."
            )
        return (
            "You are participating in a repeated two-player game-theoretic scenario.\n"
            "Each round you must choose action A or action B. The payoff matrix (your score shown first) is:\n\n"
            f"{matrix_text}\n\n"
            f"{horizon}\n"
            "Your objective is to maximise your cumulative score across rounds.\n"
            "Before selecting an action, reason step by step about the history of play so far to decide optimally.\n"
            "After that, respond with exactly A or B, nothing else."
        )

    def _refresh_system_prompt(self) -> None:
        """Refresh the system prompt from overrides or defaults."""
        if self._system_prompt_override is None:
            self.system_prompt = self._build_system_prompt()
        else:
            self.system_prompt = self._system_prompt_override

    def _should_terminate(self, round_idx: int) -> bool:
        """Return True if the episode ends after the given round index."""
        if self.termination == "single":
            return True
        if self.termination == "fixed":
            return round_idx >= self.num_turns
        if round_idx >= self.max_turns:
            return True
        return self._rng.random() < self.termination_prob

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for a player."""
        agent_id = self.agent_id(actor)
        if agent_id not in set(self.agent_ids):
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {"role": "system", "content": self.system_prompt}

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and produce initial requests."""
        options = options or {}
        if "seed" in options:
            self._seed = options["seed"] if options["seed"] is None else int(options["seed"])
            self._rng = random.Random(self._seed)
        if "payoff_matrix" in options:
            self.payoffs = self._normalize_payoffs(options["payoff_matrix"])
        if "termination" in options:
            self.termination = str(options["termination"])
        if "num_turns" in options:
            self.num_turns = int(options["num_turns"])
        if "termination_prob" in options:
            self.termination_prob = float(options["termination_prob"])
        if "max_turns" in options:
            self.max_turns = int(options["max_turns"])
        if "system_prompt" in options:
            override = options["system_prompt"]
            self._system_prompt_override = None if override is None else str(override)
        self._validate_termination()
        self._refresh_system_prompt()

        self._begin_episode()
        self.history.clear()
        self.scores = {"player_a": 0.0, "player_b": 0.0}
        self._done = False
        start_msg: Message = {
            "role": "user",
            "content": "Starting round 1. You have 0 points. Your opponent has 0 points.",
        }
        for agent_id in self.agent_ids:
            system_msg: Message = {"role": "system", "content": self.system_prompt}
            self.messages[agent_id] = [
                system_msg,
                start_msg,
            ]
        requests: list[Request] = []
        for agent_id in self.agent_ids:
            request: Request = {
                "actor": make_actor_id(agent_id),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id(agent_id)),
                "message": start_msg,
                "needs_action": True,
                "info": {"round": 1, "scores": dict(self.scores)},
            }
            requests.append(request)
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Advance one round given the players' actions."""
        if self._done:
            return []

        actions = self._normalize_actions(actions)
        a_action = actions["player_a"]
        b_action = actions["player_b"]
        if a_action not in {"A", "B"} or b_action not in {"A", "B"}:
            raise ActionError("Action must be 'A' or 'B'")

        a_reward, b_reward = self.payoffs[(a_action, b_action)]
        self.scores["player_a"] += a_reward
        self.scores["player_b"] += b_reward

        self.history.append((a_action, b_action))
        round_idx = len(self.history)

        game_over = self._should_terminate(round_idx)
        self._done = game_over
        env_text_by_agent: dict[str, str] = {}
        for name, my_action, opp_action, reward in [
            ("player_a", a_action, b_action, a_reward),
            ("player_b", b_action, a_action, b_reward),
        ]:
            opponent_score = self.scores["player_b" if name == "player_a" else "player_a"]
            if game_over:
                env_msg = (
                    f"End of round {round_idx}. You played {my_action}, opponent played {opp_action}. "
                    f"Round payoff: {reward}. Final scores – You: {self.scores[name]}, "
                    f"Opponent: {opponent_score}. Game over."
                )
            else:
                env_msg = (
                    f"End of round {round_idx}. You played {my_action}, opponent played {opp_action}. "
                    f"Round payoff: {reward}. Total scores – You: {self.scores[name]}, "
                    f"Opponent: {opponent_score}. Starting round {round_idx + 1}."
                )
            env_text_by_agent[name] = env_msg

        for agent_id in self.agent_ids:
            self.messages[agent_id].append({"role": "assistant", "content": actions[agent_id]})
            self.messages[agent_id].append({"role": "user", "content": env_text_by_agent[agent_id]})

        requests: list[Request] = []
        for name, reward in [
            ("player_a", a_reward),
            ("player_b", b_reward),
        ]:
            msg: Message = {"role": "user", "content": env_text_by_agent[name]}
            request: Request = {
                "actor": make_actor_id(name),
                "reward": float(reward),
                "message": msg,
                "needs_action": not game_over,
                "info": {
                    "round": round_idx,
                    "player_a_action": a_action,
                    "player_b_action": b_action,
                    "scores": dict(self.scores),
                },
            }
            requests.append(request)
        done = bool(game_over)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests

    def render_history(self) -> dict[str, list[Message]]:
        """Return copies of each player's conversation log."""
        return {name: msgs.copy() for name, msgs in self.messages.items()}
