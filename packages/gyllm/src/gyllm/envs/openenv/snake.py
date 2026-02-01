import random
from dataclasses import dataclass

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json, parse_int

_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_LEFT_TURN = {"up": "left", "left": "down", "down": "right", "right": "up"}
_RIGHT_TURN = {"up": "right", "right": "down", "down": "left", "left": "up"}


def _render_grid(height: int, width: int, snake: list[tuple[int, int]], fruit: tuple[int, int]) -> str:
    """Render the snake grid as a string.

    Args:
        height: Grid height.
        width: Grid width.
        snake: List of snake segment positions.
        fruit: Fruit position.

    Returns:
        Rendered grid string.
    """
    grid = [["." for _ in range(width)] for _ in range(height)]
    fr, fc = fruit
    grid[fr][fc] = "F"
    for i, (r, c) in enumerate(snake):
        grid[r][c] = "H" if i == 0 else "S"
    return "\n".join(" ".join(row) for row in grid)


@dataclass(slots=True)
class _StepOutcome:
    reward: float
    done: bool
    info: str


class SnakeEnv(LLMEnv):
    """
    A lightweight Snake environment (inspired by OpenEnv's SnakeEnvironment).

    This implementation is pure-Python (no gym/marlenv dependency) and uses a
    single-agent interface.

    Actions:
    - observer="snake": 0=noop, 1=turn left, 2=turn right
    - observer="human": 0=noop, 1=left, 2=right, 3=down, 4=up
    """

    agents: list[str] = ["player"]

    def __init__(
        self,
        *,
        height: int = 10,
        width: int = 10,
        snake_length: int = 3,
        observer: str = "snake",
        max_episode_steps: int = 200,
        reward_dict: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Snake environment.

        Args:
            height: Grid height.
            width: Grid width.
            snake_length: Initial snake length.
            observer: Action mode ("snake" or "human").
            max_episode_steps: Maximum steps per episode.
            reward_dict: Optional reward overrides.
            seed: Optional RNG seed.

        Raises:
            ValueError: If configuration values are invalid.
        """
        super().__init__()
        if height <= 2 or width <= 2:
            raise ValueError("height/width must be > 2")
        if snake_length <= 1:
            raise ValueError("snake_length must be > 1")
        if observer not in {"snake", "human"}:
            raise ValueError("observer must be 'snake' or 'human'")

        self.height = height
        self.width = width
        self.snake_length = snake_length
        self.observer = observer
        self.max_episode_steps = max_episode_steps
        self.reward_dict = reward_dict or {
            "fruit": 1.0,
            "lose": -1.0,
            "time": -0.001,
            "win": 0.0,
        }
        self._rng = random.Random(seed)

        self._step_count = 0
        self._score = 0.0
        self._snake: list[tuple[int, int]] = []
        self._direction: str = "right"
        self._fruit: tuple[int, int] = (0, 0)
        self._done = False
        self._last_info = ""

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

        if self.observer == "snake":
            action_help = "0=noop, 1=turn_left, 2=turn_right"
        else:
            action_help = "0=noop, 1=left, 2=right, 3=down, 4=up"

        return {
            "role": "system",
            "content": (
                "You are controlling a snake on a grid.\n"
                "Avoid walls and your own body. Eat fruit (F) to grow.\n"
                f"Action space: {action_help}.\n"
                "Respond with either:\n"
                "- an integer action id (e.g. `1`), or\n"
                '- a JSON object like `{"action": 1}`.'
            ),
        }

    def _spawn_fruit(self) -> tuple[int, int]:
        """Spawn a new fruit in a free cell.

        Returns:
            Fruit position.
        """
        occupied = set(self._snake)
        candidates = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in occupied]
        if not candidates:
            return (0, 0)
        return self._rng.choice(candidates)

    def _reset_game(self) -> None:
        """Reset internal game state."""
        self._step_count = 0
        self._score = 0.0
        self._done = False
        self._last_info = "New game."

        center_r = self.height // 2
        start_c = max(0, (self.width // 2) - (self.snake_length // 2))
        self._direction = "right"
        self._snake = [(center_r, start_c + i) for i in range(self.snake_length)][::-1]
        self._fruit = self._spawn_fruit()

    def _apply_action(self, action_id: int) -> None:
        """Update the snake direction based on the action.

        Args:
            action_id: Action id to apply.

        Raises:
            ActionError: If the action is invalid for the observer mode.
        """
        if self.observer == "snake":
            if action_id == 0:
                return
            if action_id == 1:
                self._direction = _LEFT_TURN[self._direction]
                return
            if action_id == 2:
                self._direction = _RIGHT_TURN[self._direction]
                return
            raise ActionError("Invalid action for observer='snake'. Expected 0, 1, or 2.")

        # observer == "human"
        if action_id == 0:
            return
        if action_id == 1:
            self._direction = "left"
            return
        if action_id == 2:
            self._direction = "right"
            return
        if action_id == 3:
            self._direction = "down"
            return
        if action_id == 4:
            self._direction = "up"
            return
        raise ActionError("Invalid action for observer='human'. Expected 0..4.")

    def _step_snake(self) -> _StepOutcome:
        """Advance the snake by one step.

        Returns:
            Outcome of the step.
        """
        self._step_count += 1

        dr, dc = _DIRS[self._direction]
        head_r, head_c = self._snake[0]
        new_head = (head_r + dr, head_c + dc)

        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            return _StepOutcome(reward=self.reward_dict["lose"], done=True, info="Hit a wall.")

        if new_head in self._snake:
            return _StepOutcome(reward=self.reward_dict["lose"], done=True, info="Hit your body.")

        ate_fruit = new_head == self._fruit
        self._snake.insert(0, new_head)
        if ate_fruit:
            self._score += self.reward_dict["fruit"]
            self._fruit = self._spawn_fruit()
            info = "Ate fruit."
            reward = self.reward_dict["fruit"]
        else:
            self._snake.pop()
            info = "Moved."
            reward = self.reward_dict["time"]

        if self._step_count >= self.max_episode_steps:
            return _StepOutcome(reward=self.reward_dict.get("win", 0.0), done=True, info="Max steps reached.")

        return _StepOutcome(reward=reward, done=False, info=info)

    def _user_message(self, reward: float, done: bool) -> Message:
        """Build a user-facing message for the current state.

        Args:
            reward: Last reward value.
            done: Whether the episode is done.

        Returns:
            Message payload for the user.
        """
        grid = _render_grid(self.height, self.width, self._snake, self._fruit)
        return {
            "role": "user",
            "content": (
                f"{self._last_info}\n\n"
                f"Grid:\n{grid}\n\n"
                f"Step: {self._step_count}/{self.max_episode_steps} | Score: {self._score:.3f} | Last reward: {reward:.3f}\n"
                + ("Game over." if done else "Choose your next action.")
            ),
        }

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the environment and return initial request.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "seed" in options:
            self._rng = random.Random(int(options["seed"]))
        if "max_episode_steps" in options:
            max_episode_steps = int(options["max_episode_steps"])
            if max_episode_steps <= 0:
                raise ValueError(f"max_episode_steps must be > 0; got {max_episode_steps}")
            self.max_episode_steps = max_episode_steps
        self._begin_episode()
        self._reset_game()
        msg = self._user_message(reward=0.0, done=False)
        requests: list[Request] = [
            {
                "actor": make_actor_id("player"),
                "reward": 0.0,
                "system_message": self._system_message(make_actor_id("player")),
                "message": msg,
                "needs_action": True,
                "info": {},
            }
        ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply an action and return the next request.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests produced after the step.
        """
        if self._done:
            return []

        actions = self._normalize_actions(actions)
        raw = actions["player"].strip()
        parsed = maybe_parse_json(raw)
        if isinstance(parsed, dict) and "action" in parsed:
            raw = str(parsed["action"])

        try:
            action_id = parse_int(raw)
        except Exception as exc:
            raise ActionError('Action must be an integer (or JSON like {"action": 1}).') from exc

        self._apply_action(action_id)
        outcome = self._step_snake()
        self._done = outcome.done
        self._last_info = outcome.info

        msg = self._user_message(reward=outcome.reward, done=outcome.done)
        requests: list[Request] = [
            {
                "actor": make_actor_id("player"),
                "reward": float(outcome.reward),
                "message": msg,
                "needs_action": not outcome.done,
                "info": {
                    "action_id": action_id,
                    "done": outcome.done,
                    "detail": outcome.info,
                },
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
