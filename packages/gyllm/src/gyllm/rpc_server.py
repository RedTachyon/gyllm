import argparse
import importlib
import json
import sys
import traceback
from typing import Any

from gyllm.core import LLMEnv, Request, actor_agent_id, actor_env_id, make_actor_id


def _resolve_env_class(spec: str) -> type[LLMEnv]:
    """Resolve an env class from a module path specification.

    Args:
        spec: Entry point string in "module.path:ClassName" form.

    Returns:
        Environment class.

    Raises:
        ValueError: If the spec is malformed.
        AttributeError: If the class does not exist.
        TypeError: If the resolved object is not a class.
    """
    if ":" not in spec:
        raise ValueError("Env spec must be of the form 'module.path:ClassName'")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Module {module_name!r} has no attribute {class_name!r}")
    if not isinstance(cls, type):
        raise TypeError(f"{spec!r} did not resolve to a class")
    return cls


def _wire_requests(requests: list[Request]) -> list[dict[str, Any]]:
    """Serialize requests to RPC wire format.

    Args:
        requests: Requests to serialize.

    Returns:
        Wire-ready request dictionaries.

    Raises:
        ValueError: If an actor id includes env metadata.
    """
    wire: list[dict[str, Any]] = []
    for request in requests:
        agent_id = actor_agent_id(request["actor"])
        env_id = actor_env_id(request["actor"])
        if env_id not in (None, 0):
            raise ValueError(f"RPC server only supports env_id=0; got {request['actor']!r}")
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "reward": request["reward"],
            "message": request["message"],
            "needs_action": request["needs_action"],
            "info": request.get("info", {}),
            "episode_id": request.get("episode_id", 0),
            "group_id": request.get("group_id"),
            "episode_start": request.get("episode_start", False),
            "episode_end": request.get("episode_end", False),
        }
        if "system_message" in request:
            payload["system_message"] = request["system_message"]
        wire.append(payload)
    return wire


def _ok(msg_id: int, result: Any) -> dict[str, Any]:
    """Create a successful RPC response payload.

    Args:
        msg_id: Message id for correlation.
        result: Result payload.

    Returns:
        RPC response dict.
    """
    return {"id": msg_id, "ok": True, "result": result}


def _err(msg_id: int, exc: BaseException) -> dict[str, Any]:
    """Create an error RPC response payload.

    Args:
        msg_id: Message id for correlation.
        exc: Exception to serialize.

    Returns:
        RPC response dict with error metadata.
    """
    return {
        "id": msg_id,
        "ok": False,
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the JSONL-RPC server.

    Args:
        argv: Optional argument list for CLI parsing.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Run a GYLLM env as a JSONL-RPC server over stdin/stdout.")
    parser.add_argument(
        "--env",
        required=True,
        help="Environment class spec, e.g. 'gyllm.envs.simple.iterated_games:TftIpdEnv'",
    )
    parser.add_argument(
        "--env-kwargs-json",
        default="{}",
        help="JSON object passed as kwargs to the env constructor.",
    )
    args = parser.parse_args(argv)

    env_cls = _resolve_env_class(args.env)
    env_kwargs = json.loads(args.env_kwargs_json)
    if not isinstance(env_kwargs, dict):
        raise TypeError("--env-kwargs-json must decode to a JSON object/dict")

    env: LLMEnv = env_cls(**env_kwargs)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        msg = json.loads(line)
        msg_id = int(msg.get("id"))
        method = msg.get("method")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            resp = _err(msg_id, TypeError("params must be an object/dict"))
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        try:
            if method == "spec":
                result = {"protocol_version": 5, "agent_ids": list(getattr(env, "agent_ids", []))}
            elif method == "reset":
                opts = params.get("options")
                if opts is not None and not isinstance(opts, dict):
                    raise TypeError("options must be an object/dict or null")
                result = _wire_requests(env.reset(opts))
            elif method == "step":
                raw_actions = params["actions"]
                if not isinstance(raw_actions, dict):
                    raise TypeError("actions must be an object/dict mapping agent_id -> completion")
                actions = {str(agent_id): str(text) for agent_id, text in raw_actions.items()}
                result = _wire_requests(env.step({make_actor_id(a): t for a, t in actions.items()}))
            elif method == "close":
                env.close()
                result = None
            else:
                raise ValueError(f"Unknown method: {method!r}")

            resp = _ok(msg_id, result)
        except BaseException as exc:
            resp = _err(msg_id, exc)

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

        if method == "close":
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
