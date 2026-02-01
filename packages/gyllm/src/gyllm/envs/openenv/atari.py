from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json, parse_int


class AtariEnv(LLMEnv):
    """
    Atari environment wrapper (inspired by OpenEnv's atari_env).

    Requires optional `ale-py` + ROMs.
    """

    agents: list[str] = ["player"]

    def __init__(
        self,
        *,
        game_name: str = "pong",
        obs_type: str = "rgb",
        full_action_space: bool = False,
        mode: int | None = None,
        difficulty: int | None = None,
        repeat_action_probability: float = 0.0,
        frameskip: int = 4,
    ) -> None:
        """Initialize the Atari environment wrapper.

        Args:
            game_name: Atari game name.
            obs_type: Observation type ("rgb", "grayscale", "ram").
            full_action_space: Whether to use full action set.
            mode: Optional Atari mode.
            difficulty: Optional Atari difficulty.
            repeat_action_probability: Probability of repeating actions.
            frameskip: Number of frames to repeat per action.

        Raises:
            ImportError: If ale-py is unavailable.
            ValueError: If obs_type is invalid or ROM loading fails.
        """
        super().__init__()
        try:
            from ale_py import ALEInterface, LoggerMode, roms  # type: ignore[import]
        except Exception as exc:  # pragma: no cover
            raise ImportError("AtariEnv requires the optional `ale-py` dependency and ROMs.") from exc

        if obs_type not in {"rgb", "grayscale", "ram"}:
            raise ValueError("obs_type must be one of: 'rgb', 'grayscale', 'ram'")

        self.game_name = game_name
        self.obs_type = obs_type
        self.full_action_space = full_action_space
        self.mode = mode
        self.difficulty = difficulty
        self.repeat_action_probability = repeat_action_probability
        self.frameskip = frameskip

        self._ALEInterface = ALEInterface
        self._roms = roms

        self.ale = self._ALEInterface()
        self.ale.setLoggerMode(LoggerMode.Error)
        self.ale.setFloat("repeat_action_probability", float(repeat_action_probability))

        try:
            rom_path = self._roms.get_rom_path(game_name)
            self.ale.loadROM(rom_path)
        except Exception as exc:
            raise ValueError(f"Failed to load Atari ROM for game {game_name!r}: {exc}") from exc

        if mode is not None:
            self.ale.setMode(int(mode))
        if difficulty is not None:
            self.ale.setDifficulty(int(difficulty))

        if full_action_space:
            self._action_set = list(self.ale.getLegalActionSet())
        else:
            self._action_set = list(self.ale.getMinimalActionSet())

        self._step = 0

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
        return {
            "role": "system",
            "content": (
                f"You are playing Atari: {self.game_name}.\n"
                "Pick an `action_id` index into the provided action list.\n"
                "Respond with either:\n"
                "- an integer action_id (e.g. `0`), or\n"
                '- a JSON object like `{"action_id": 0}`.'
            ),
        }

    def _get_obs_summary(self) -> str:
        """Return a compact observation summary string.

        Returns:
            Observation summary string.
        """
        if self.obs_type == "rgb":
            screen = self.ale.getScreenRGB()
        elif self.obs_type == "grayscale":
            screen = self.ale.getScreenGrayscale()
        else:
            screen = self.ale.getRAM()

        # Avoid dumping huge arrays into the prompt; provide lightweight summary.
        try:
            shape = list(getattr(screen, "shape", [])) or [len(screen)]
            flat = screen.flatten().tolist() if hasattr(screen, "flatten") else list(screen)
        except Exception:
            shape = []
            flat = []
        preview = flat[:32]
        return f"obs_type={self.obs_type} shape={shape} preview={preview}"

    def reset(self, options: dict[str, object] | None = None) -> list[Request]:
        """Reset the Atari environment.

        Args:
            options: Optional reset overrides.

        Returns:
            Initial request list.
        """
        options = options or {}
        if "frameskip" in options:
            frameskip = int(options["frameskip"])
            if frameskip <= 0:
                raise ValueError(f"frameskip must be > 0; got {frameskip}")
            self.frameskip = frameskip
        prompt = options.get("prompt")
        self._begin_episode()
        self.ale.reset_game()
        self._step = 0
        content = f"Reset complete.\n{self._get_obs_summary()}\nlegal_action_ids=0..{len(self._action_set) - 1}"
        if prompt:
            content = f"{prompt}\n{content}"
        msg: Message = {
            "role": "user",
            "content": content,
        }
        request: Request = {
            "actor": make_actor_id("player"),
            "reward": 0.0,
            "system_message": self._system_message(make_actor_id("player")),
            "message": msg,
            "needs_action": True,
            "info": {
                "frameskip": self.frameskip,
                "legal_action_min": 0,
                "legal_action_max": len(self._action_set) - 1,
            },
        }
        requests: list[Request] = [request]
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
        if self.ale.game_over():
            return []

        actions = self._normalize_actions(actions)
        raw = actions["player"].strip()
        parsed = maybe_parse_json(raw)
        if isinstance(parsed, dict) and "action_id" in parsed:
            raw = str(parsed["action_id"])
        try:
            action_id = parse_int(raw, minimum=0, maximum=len(self._action_set) - 1)
        except Exception as exc:
            raise ActionError("Action must be an integer action_id (or JSON like {'action_id': 0}).") from exc

        ale_action = self._action_set[action_id]
        total_reward = 0.0
        for _ in range(self.frameskip):
            total_reward += float(self.ale.act(ale_action))
            if self.ale.game_over():
                break

        self._step += 1
        done = bool(self.ale.game_over())
        lives = int(self.ale.lives())
        msg: Message = {
            "role": "user",
            "content": (
                f"Step {self._step}: action_id={action_id} reward={total_reward} done={done} lives={lives}\n"
                f"{self._get_obs_summary()}"
            ),
        }
        request: Request = {
            "actor": make_actor_id("player"),
            "reward": total_reward,
            "message": msg,
            "needs_action": not done,
            "info": {
                "action_id": action_id,
                "lives": lives,
                "done": done,
            },
        }
        requests: list[Request] = [request]
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
