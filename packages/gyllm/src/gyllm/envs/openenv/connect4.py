import random
from dataclasses import dataclass

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json, parse_int


def _render_board(board: list[list[int]]) -> str:
    """Render the board as a string.

    Args:
        board: 2D board with -1, 0, 1 values.

    Returns:
        Rendered board string.
    """
    rows = len(board)
    cols = len(board[0]) if rows else 0
    symbol = {0: ".", 1: "X", -1: "O"}
    lines = [" ".join(str(c) for c in range(cols))]
    for r in range(rows):
        lines.append(" ".join(symbol[board[r][c]] for c in range(cols)))
    return "\n".join(lines)


def _legal_actions(board: list[list[int]]) -> list[int]:
    """Return available column indices.

    Args:
        board: 2D board with -1, 0, 1 values.

    Returns:
        List of playable column indices.
    """
    cols = len(board[0])
    return [c for c in range(cols) if board[0][c] == 0]


def _parse_column_action(text: str, *, maximum: int) -> int:
    """Parse a Connect4 column action from text.

    Args:
        text: Action text containing a column index.
        maximum: Maximum allowed column index.

    Returns:
        Parsed column index.

    Raises:
        ValueError: If parsing fails or index is out of range.
    """
    raw = text.strip()
    parsed = maybe_parse_json(raw)
    if isinstance(parsed, dict) and "column" in parsed:
        raw = str(parsed["column"])
    return parse_int(raw, minimum=0, maximum=maximum)


def _check_win_or_draw(board: list[list[int]], row: int, col: int) -> tuple[float, bool]:
    """Check for a win or draw after a move.

    Args:
        board: Board state.
        row: Row index of the last move.
        col: Column index of the last move.

    Returns:
        Tuple of reward and done flag.
    """
    player = board[row][col]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    rows = len(board)
    cols = len(board[0])
    for dr, dc in directions:
        count = 0
        for step in range(-3, 4):
            r = row + step * dr
            c = col + step * dc
            if 0 <= r < rows and 0 <= c < cols and board[r][c] == player:
                count += 1
                if count >= 4:
                    return 1.0, True
            else:
                count = 0
    if all(cell != 0 for r in board for cell in r):
        return 0.0, True
    return 0.0, False


@dataclass(slots=True)
class _MoveResult:
    played: bool
    row: int | None
    col: int | None
    reward: float
    done: bool
    reason: str


class Connect4Env(LLMEnv):
    """
    Connect4 environment (inspired by OpenEnv's Connect4Environment).

    Supports:
    - `opponent=None`  -> two-agent turn-based game (player_a vs player_b)
    - `opponent="random"` -> single-agent (player) vs random opponent
    """

    ROWS = 6
    COLUMNS = 7

    def __init__(self, *, opponent: str | None = None, seed: int | None = None) -> None:
        """Initialize the Connect4 environment.

        Args:
            opponent: Opponent type ("random") or None for two-player.
            seed: Optional RNG seed.

        Raises:
            ValueError: If the opponent type is unsupported.
        """
        super().__init__()
        self._rng = random.Random(seed)
        self.opponent = opponent

        if opponent is None:
            self.agents = ["player_a", "player_b"]
            self._agent_for_piece = {1: "player_a", -1: "player_b"}
        elif opponent == "random":
            self.agents = ["player"]
            self._agent_for_piece = {1: "player"}
        else:
            raise ValueError(f"Unknown opponent: {opponent!r} (supported: None, 'random')")

        self.board: list[list[int]] = []
        self.next_player: int = 1
        self._done: bool = False
        self._last_move_summary: str = ""

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the actor.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id not in set(self.agent_ids):
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                "You are playing Connect4.\n"
                "Goal: get 4 in a row (horizontal/vertical/diagonal).\n"
                f"Choose a column index 0..{self.COLUMNS - 1}.\n"
                "Respond with either:\n"
                f"- an integer column (e.g. `3`), or\n"
                f'- a JSON object like `{{"column": 3}}`.'
            ),
        }

    def _reset_game(self) -> None:
        """Reset internal game state."""
        self.board = [[0 for _ in range(self.COLUMNS)] for _ in range(self.ROWS)]
        self.next_player = 1
        self._done = False
        self._last_move_summary = "New game."

    def _current_actor(self) -> str:
        """Return the current actor name.

        Returns:
            Agent name for the current turn.
        """
        return self._agent_for_piece[self.next_player]

    def _drop_piece(self, col: int) -> _MoveResult:
        """Drop a piece in a column.

        Args:
            col: Column index.

        Returns:
            Move result describing the outcome.
        """
        if col < 0 or col >= self.COLUMNS or self.board[0][col] != 0:
            return _MoveResult(
                played=False,
                row=None,
                col=col,
                reward=-1.0,
                done=True,
                reason=f"Invalid move: column {col} is not playable.",
            )

        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.next_player
                reward, done = _check_win_or_draw(self.board, row, col)
                reason = ("Win." if reward == 1.0 else "Draw.") if done else "Move accepted."
                return _MoveResult(
                    played=True,
                    row=row,
                    col=col,
                    reward=reward,
                    done=done,
                    reason=reason,
                )

        return _MoveResult(
            played=False,
            row=None,
            col=col,
            reward=-1.0,
            done=True,
            reason=f"Invalid move: column {col} is full.",
        )

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
                    self.next_player = -1
                    legal = _legal_actions(self.board)
                    opp_col = self._rng.choice(legal)
                    move = self._drop_piece(opp_col)
                    self._done = move.done
                    self._last_move_summary = f"Opponent played column {opp_col}. {move.reason}"
                    self.next_player = 1
                else:
                    raise ValueError("starting_player must be 'player' or 'opponent' for single-agent.")
        board_text = _render_board(self.board)
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
                        "content": (
                            f"{self._last_move_summary}\n\n"
                            f"Board:\n{board_text}\n\n"
                            f"Legal actions: {legal}\n"
                            f"Turn: {current} (X=player_a, O=player_b)."
                        ),
                    },
                    "needs_action": True,
                    "info": {"legal_actions": legal},
                }
            ]
        else:
            # Single-agent vs random opponent
            requests: list[Request] = [
                {
                    "actor": make_actor_id("player"),
                    "reward": 0.0,
                    "system_message": self._system_message(make_actor_id("player")),
                    "message": {
                        "role": "user",
                        "content": (
                            f"{self._last_move_summary}\n\n"
                            f"Board:\n{board_text}\n\n"
                            f"Legal actions: {legal}\n"
                            "You are X. Choose a column."
                        ),
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
        """Apply two-player actions and return requests.

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
            col = _parse_column_action(actions[current], maximum=self.COLUMNS - 1)
        except Exception as exc:
            raise ActionError(
                f'Invalid action. Expected integer column 0..{self.COLUMNS - 1} or JSON like {{"column": 3}}.'
            ) from exc

        move = self._drop_piece(col)
        self._last_move_summary = f"{current} played column {col}. {move.reason}"
        self._done = move.done

        # Reward shaping (zero-sum): current player gets move.reward, other gets -move.reward.
        reward_by_agent: dict[str, float] = {"player_a": 0.0, "player_b": 0.0}
        other = "player_b" if current == "player_a" else "player_a"

        if not move.played:
            reward_by_agent[current] = -1.0
            reward_by_agent[other] = 1.0
        elif move.done and move.reward == 1.0:
            reward_by_agent[current] = 1.0
            reward_by_agent[other] = -1.0
        else:
            reward_by_agent[current] = 0.0
            reward_by_agent[other] = 0.0

        if not self._done:
            self.next_player *= -1

        board_text = _render_board(self.board)
        legal = _legal_actions(self.board)
        next_actor = self._current_actor() if not self._done else None

        content = f"{self._last_move_summary}\n\nBoard:\n{board_text}\n\nLegal actions: {legal}\n" + (
            "Game over." if self._done else f"Turn: {next_actor}."
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

        assert next_actor is not None
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
        """Apply single-agent action and return requests.

        Args:
            actions: Mapping of agent ids to actions.

        Returns:
            Requests produced after the move.
        """
        if self._done:
            return []

        try:
            col = _parse_column_action(actions["player"], maximum=self.COLUMNS - 1)
        except Exception as exc:
            raise ActionError(
                f'Invalid action. Expected integer column 0..{self.COLUMNS - 1} or JSON like {{"column": 3}}.'
            ) from exc

        player_move = self._drop_piece(col)
        if not player_move.played:
            self._done = True
            msg = f"You played column {col}. {player_move.reason}"
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": -1.0,
                    "message": {"role": "user", "content": msg},
                    "needs_action": False,
                    "info": {"result": "illegal_move"},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        if player_move.done:
            self._done = True
            msg = f"You played column {col}. {player_move.reason}\n\nBoard:\n{_render_board(self.board)}\nGame over."
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": 1.0 if player_move.reward == 1.0 else 0.0,
                    "message": {"role": "user", "content": msg},
                    "needs_action": False,
                    "info": {"result": "terminal"},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        # Opponent random move
        self.next_player *= -1
        legal = _legal_actions(self.board)
        opp_col = self._rng.choice(legal)
        opp_move = self._drop_piece(opp_col)

        summary = f"You played {col}. Opponent played {opp_col}."
        if opp_move.done:
            self._done = True
            if opp_move.reward == 1.0:
                result = "Opponent wins."
                reward = -1.0
            else:
                result = "Draw."
                reward = 0.0
            content = f"{summary} {result}\n\nBoard:\n{_render_board(self.board)}\nGame over."
            return [
                {
                    "actor": make_actor_id("player"),
                    "reward": reward,
                    "message": {"role": "user", "content": content},
                    "needs_action": False,
                    "info": {"result": "terminal"},
                    "episode_id": self._episode_id,
                    "episode_start": False,
                    "episode_end": True,
                }
            ]

        self.next_player *= -1
        board_text = _render_board(self.board)
        legal = _legal_actions(self.board)
        content = f"{summary}\n\nBoard:\n{board_text}\n\nLegal actions: {legal}\nYour turn."
        return [
            {
                "actor": make_actor_id("player"),
                "reward": 0.0,
                "message": {"role": "user", "content": content},
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
