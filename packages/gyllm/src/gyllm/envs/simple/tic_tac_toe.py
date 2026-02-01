import random
from dataclasses import dataclass

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json, parse_int


def _render_board(board: list[list[int]]) -> str:
    """Render the board as a human-readable string.

    Args:
        board: 3x3 board with -1, 0, 1 values.

    Returns:
        Rendered board string.
    """
    symbol = {0: ".", 1: "X", -1: "O"}
    lines = ["  0 1 2"]
    for r in range(3):
        lines.append(f"{r} " + " ".join(symbol[board[r][c]] for c in range(3)))
    return "\n".join(lines)


def _legal_actions(board: list[list[int]]) -> list[int]:
    """Return the list of legal action indices.

    Args:
        board: 3x3 board with -1, 0, 1 values.

    Returns:
        List of legal move indices.
    """
    out: list[int] = []
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                out.append(r * 3 + c)
    return out


def _parse_action(text: str) -> tuple[int, int]:
    """Parse a Tic-Tac-Toe move from text.

    Args:
        text: Action text containing an index or row/col JSON.

    Returns:
        Tuple of (row, col) coordinates.

    Raises:
        ValueError: If the action cannot be parsed or is out of range.
    """
    raw = text.strip()
    parsed = maybe_parse_json(raw)
    if isinstance(parsed, dict):
        if "index" in parsed:
            idx = parse_int(str(parsed["index"]), minimum=0, maximum=8)
            return idx // 3, idx % 3
        if "row" in parsed and "col" in parsed:
            row = parse_int(str(parsed["row"]), minimum=0, maximum=2)
            col = parse_int(str(parsed["col"]), minimum=0, maximum=2)
            return row, col
        raise ValueError('Expected {"index": n} or {"row": r, "col": c}.')

    idx = parse_int(raw, minimum=0, maximum=8)
    return idx // 3, idx % 3


def _check_winner(board: list[list[int]]) -> int | None:
    """Check if the board has a winner.

    Args:
        board: 3x3 board with -1, 0, 1 values.

    Returns:
        Winning player value (1 or -1), or None if no winner.
    """
    lines = []
    lines.extend(board)
    lines.extend([[board[r][c] for r in range(3)] for c in range(3)])
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])
    for line in lines:
        s = sum(line)
        if s == 3:
            return 1
        if s == -3:
            return -1
    return None


@dataclass(slots=True)
class _MoveResult:
    played: bool
    row: int
    col: int
    done: bool
    winner: int | None
    reason: str


class TicTacToeEnv(LLMEnv):
    """
    Tic-Tac-Toe (3x3).

    Supports:
    - `opponent=None` -> two-agent turn-based game (player_a vs player_b)
    - `opponent="random"` -> single-agent (player) vs random opponent
    """

    def __init__(
        self,
        *,
        opponent: str | None = None,
        seed: int | None = None,
        repeat_invalid_action: bool = False,
    ) -> None:
        """Initialize the Tic-Tac-Toe environment.

        Args:
            opponent: Opponent type ("random") or None for two-player.
            seed: Optional RNG seed.
            repeat_invalid_action: Whether to reject invalid actions and repeat the request.

        Raises:
            ValueError: If the opponent type is unsupported.
        """
        super().__init__()
        self._rng = random.Random(seed)
        self.opponent = opponent
        self.repeat_invalid_action = repeat_invalid_action

        if opponent is None:
            self.agents = ["player_a", "player_b"]
            self._agent_for_piece = {1: "player_a", -1: "player_b"}
            self._piece_for_agent = {"player_a": 1, "player_b": -1}
        elif opponent == "random":
            self.agents = ["player"]
            self._agent_for_piece = {1: "player"}
            self._piece_for_agent = {"player": 1}
        else:
            raise ValueError(f"Unknown opponent: {opponent!r} (supported: None, 'random')")

        self.board: list[list[int]] = []
        self.next_player: int = 1
        self._done: bool = False
        self._last_summary: str = ""

    def _reset_game(self) -> None:
        """Reset internal board state for a new game."""
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.next_player = 1
        self._done = False
        self._last_summary = "New game."

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the actor.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if self.opponent is None:
            if agent_id not in {"player_a", "player_b"}:
                raise KeyError(f"Unknown agent_id: {agent_id!r}")
            you = "X" if agent_id == "player_a" else "O"
        else:
            if agent_id != "player":
                raise KeyError(f"Unknown agent_id: {agent_id!r}")
            you = "X"

        return {
            "role": "system",
            "content": (
                "You are playing Tic-Tac-Toe on a 3x3 grid.\n"
                "Win by getting 3 in a row (horizontal/vertical/diagonal).\n"
                f"You are {you}.\n"
                "X goes first, O goes second.\n\n"
                "Choose an empty cell by responding with either:\n"
                "- an integer index 0..8 (row-major), or\n"
                '- a JSON object like {"index": 4} or {"row": 1, "col": 1}.\n'
                + (
                    "Invalid actions lose the game."
                    if not self.repeat_invalid_action
                    else "Invalid actions are rejected."
                )
            ),
        }

    def _current_actor(self) -> str:
        """Return the current player's agent id.

        Returns:
            Agent id for the current turn.
        """
        return self._agent_for_piece[self.next_player]

    def _play(self, row: int, col: int, player: int) -> _MoveResult:
        """Apply a move to the board and return the result.

        Args:
            row: Row index.
            col: Column index.
            player: Player value (1 or -1).

        Returns:
            Move result describing the outcome.
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            return _MoveResult(
                played=False,
                row=row,
                col=col,
                done=True,
                winner=-player,
                reason=f"Invalid move: ({row}, {col}) is out of bounds.",
            )
        if self.board[row][col] != 0:
            return _MoveResult(
                played=False,
                row=row,
                col=col,
                done=True,
                winner=-player,
                reason=f"Invalid move: ({row}, {col}) is already occupied.",
            )

        self.board[row][col] = player
        winner = _check_winner(self.board)
        if winner is not None:
            return _MoveResult(played=True, row=row, col=col, done=True, winner=winner, reason="Win.")
        if not _legal_actions(self.board):
            return _MoveResult(played=True, row=row, col=col, done=True, winner=None, reason="Draw.")
        return _MoveResult(played=True, row=row, col=col, done=False, winner=None, reason="Move accepted.")

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return initial requests.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "seed" in options:
            self._rng = random.Random(int(options["seed"]))
        self._begin_episode()
        self._reset_game()
        starting_player = options.get("starting_player")
        if self.opponent is None:
            if starting_player is not None:
                starting_player = str(starting_player)
                if starting_player == "player_a":
                    self.next_player = 1
                elif starting_player == "player_b":
                    self.next_player = -1
                elif starting_player == "random":
                    self.next_player = self._rng.choice([1, -1])
                else:
                    raise ValueError("starting_player must be 'player_a', 'player_b', or 'random' for two-player.")
        else:
            if starting_player is not None:
                starting_player = str(starting_player)
                if starting_player == "player":
                    self.next_player = 1
                elif starting_player in {"opponent", "random"}:
                    legal = _legal_actions(self.board)
                    opp_idx = self._rng.choice(legal)
                    opp_row, opp_col = opp_idx // 3, opp_idx % 3
                    move = self._play(opp_row, opp_col, player=-1)
                    self._done = move.done
                    self._last_summary = f"Opponent played ({opp_row}, {opp_col}). {move.reason}"
                    self.next_player = 1
                else:
                    raise ValueError("starting_player must be 'player' or 'opponent' for single-agent.")
        board = _render_board(self.board)
        legal = _legal_actions(self.board)

        if self.opponent is None:
            current = self._current_actor()
            requests: list[Request] = [
                {
                    "actor": make_actor_id(current),
                    "reward": 0.0,
                    "system_message": self._system_message(make_actor_id(current)),
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nLegal actions: {legal}\nYour turn.",
                    },
                    "needs_action": True,
                    "info": {"legal_actions": legal},
                }
            ]
        else:
            requests: list[Request] = [
                {
                    "actor": make_actor_id("player"),
                    "reward": 0.0,
                    "system_message": self._system_message(make_actor_id("player")),
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nLegal actions: {legal}\nYour turn.",
                    },
                    "needs_action": True,
                    "info": {"legal_actions": legal},
                }
            ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def _act_two_player(self, actions: dict[str, str]) -> list[Request]:
        """Apply a two-player action and return requests.

        Args:
            actions: Mapping of agent ids to actions.

        Returns:
            Requests produced after the move.
        """
        if self._done:
            return []

        current = self._current_actor()
        if current not in actions:
            raise ActionError(f"Missing action for {current!r}")
        unexpected = set(actions) - {current}
        if unexpected:
            raise ActionError(f"Unexpected actions for: {sorted(unexpected)!r}")

        try:
            row, col = _parse_action(actions[current])
        except Exception as exc:
            if self.repeat_invalid_action:
                raise ActionError(
                    'Invalid action. Expected 0..8 or {"index": n} or {"row": r, "col": c}. Got '
                    + actions[current]
                    + " instead."
                ) from exc
            self._done = True
            winner = "player_b" if current == "player_a" else "player_a"
            reward_by_agent = {winner: 1.0, current: -1.0}
            board = _render_board(self.board)
            legal = _legal_actions(self.board)
            self._last_summary = f"{current} provided an invalid action. {winner} wins."
            content = f"{self._last_summary}\n\nBoard:\n{board}\n\nLegal actions: {legal}\nGame over."
            return [
                {
                    "actor": make_actor_id("player_a"),
                    "reward": float(reward_by_agent["player_a"]),
                    "message": {"role": "user", "content": content},
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                },
                {
                    "actor": make_actor_id("player_b"),
                    "reward": float(reward_by_agent["player_b"]),
                    "message": {"role": "user", "content": content},
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                },
            ]

        player = self._piece_for_agent[current]
        move = self._play(row, col, player)
        self._done = move.done

        board = _render_board(self.board)
        legal = _legal_actions(self.board)

        if not move.played:
            loser = current
            winner = "player_b" if current == "player_a" else "player_a"
            reward_by_agent = {winner: 1.0, loser: -1.0}
            self._last_summary = f"{current} attempted ({row}, {col}). {move.reason} {winner} wins."
        elif move.winner == 1:
            reward_by_agent = {"player_a": 1.0, "player_b": -1.0}
            self._last_summary = f"{current} played ({row}, {col}). {move.reason} player_a wins."
        elif move.winner == -1:
            reward_by_agent = {"player_a": -1.0, "player_b": 1.0}
            self._last_summary = f"{current} played ({row}, {col}). {move.reason} player_b wins."
        elif move.done:
            reward_by_agent = {"player_a": 0.0, "player_b": 0.0}
            self._last_summary = f"{current} played ({row}, {col}). {move.reason}"
        else:
            reward_by_agent = {"player_a": 0.0, "player_b": 0.0}
            self._last_summary = f"{current} played ({row}, {col}). {move.reason}"
            self.next_player *= -1

        content = f"{self._last_summary}\n\nBoard:\n{board}\n\nLegal actions: {legal}\n" + (
            "Game over." if self._done else "Your turn."
        )

        if self._done:
            return [
                {
                    "actor": make_actor_id("player_a"),
                    "reward": float(reward_by_agent["player_a"]),
                    "message": {"role": "user", "content": content},
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                },
                {
                    "actor": make_actor_id("player_b"),
                    "reward": float(reward_by_agent["player_b"]),
                    "message": {"role": "user", "content": content},
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                },
            ]

        next_actor = self._current_actor()
        return [
            {
                "actor": make_actor_id(next_actor),
                "reward": float(reward_by_agent[next_actor]),
                "message": {"role": "user", "content": content},
                "needs_action": True,
                "info": {"legal_actions": legal},
                "episode_id": self._episode_id,
                "episode_start": False,
                "episode_end": False,
            }
        ]

    def _act_single_agent(self, actions: dict[str, str]) -> list[Request]:
        """Apply a single-agent action and return requests.

        Args:
            actions: Mapping of agent ids to actions.

        Returns:
            Requests produced after the move.
        """
        if self._done:
            return []

        if "player" not in actions:
            raise ActionError("Missing action for 'player'")
        unexpected = set(actions) - {"player"}
        if unexpected:
            raise ActionError(f"Unexpected actions for: {sorted(unexpected)!r}")

        try:
            row, col = _parse_action(actions["player"])
        except Exception as exc:
            if self.repeat_invalid_action:
                raise ActionError(
                    'Invalid action. Expected 0..8 or {"index": n} or {"row": r, "col": c}. Got '
                    + actions["player"]
                    + " instead."
                ) from exc
            self._done = True
            board = _render_board(self.board)
            legal = _legal_actions(self.board)
            self._last_summary = "Invalid action. Opponent wins."
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": -1.0,
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nLegal actions: {legal}\nGame over.",
                    },
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        player_move = self._play(row, col, player=1)
        board = _render_board(self.board)
        legal = _legal_actions(self.board)

        if not player_move.played:
            self._done = True
            self._last_summary = f"You attempted ({row}, {col}). {player_move.reason} Opponent wins."
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": -1.0,
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nGame over.",
                    },
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        if player_move.winner == 1:
            self._done = True
            self._last_summary = f"You played ({row}, {col}). {player_move.reason}"
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": 1.0,
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nGame over.",
                    },
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        if player_move.done:
            self._done = True
            self._last_summary = f"You played ({row}, {col}). {player_move.reason}"
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": 0.0,
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nGame over.",
                    },
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        opp_idx = self._rng.choice(legal)
        opp_row, opp_col = opp_idx // 3, opp_idx % 3
        opp_move = self._play(opp_row, opp_col, player=-1)
        board = _render_board(self.board)
        legal = _legal_actions(self.board)

        if opp_move.winner == -1:
            self._done = True
            self._last_summary = f"You played ({row}, {col}). Opponent played ({opp_row}, {opp_col}). Opponent wins."
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": -1.0,
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nGame over.",
                    },
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        if opp_move.done:
            self._done = True
            self._last_summary = f"You played ({row}, {col}). Opponent played ({opp_row}, {opp_col}). Draw."
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": 0.0,
                    "message": {
                        "role": "user",
                        "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nGame over.",
                    },
                    "needs_action": False,
                    "info": {"legal_actions": legal},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        self._last_summary = f"You played ({row}, {col}). Opponent played ({opp_row}, {opp_col})."
        return [
            {
                "actor": make_actor_id("player"),
                "reward": 0.0,
                "message": {
                    "role": "user",
                    "content": f"{self._last_summary}\n\nBoard:\n{board}\n\nLegal actions: {legal}\nYour turn.",
                },
                "needs_action": True,
                "info": {"legal_actions": legal},
                "episode_id": self._episode_id,
                "episode_start": False,
                "episode_end": False,
            }
        ]

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply actions and return resulting requests.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after the step.
        """
        actions = self._normalize_actions(actions)
        requests = self._act_two_player(actions) if self.opponent is None else self._act_single_agent(actions)
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
