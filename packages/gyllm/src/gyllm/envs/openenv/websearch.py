import json
import os
import urllib.error
import urllib.request

from gyllm.core import ActionError, ActorId, LLMEnv, Message, Request, make_actor_id
from gyllm.envs.openenv.common import maybe_parse_json


class WebSearchEnv(LLMEnv):
    """
    Web search environment (inspired by OpenEnv's websearch_env).

    This env uses Serper (SERPER_API_KEY) when available.
    If the key is missing, it returns an error message and ends the episode.
    """

    agents: list[str] = ["agent"]

    def __init__(self, *, top_k: int = 5) -> None:
        """Initialize the web search environment.

        Args:
            top_k: Number of search results to return.
        """
        super().__init__()
        self.top_k = top_k
        self._api_key = os.environ.get("SERPER_API_KEY")

    def _system_message(self, actor: ActorId) -> Message:
        """Return the system prompt for the agent.

        Args:
            actor: Actor id string.

        Returns:
            System message payload.
        """
        agent_id = self.agent_id(actor)
        if agent_id != "agent":
            raise KeyError(f"Unknown agent_id: {agent_id!r}")
        return {
            "role": "system",
            "content": (
                "You can search the web via Serper.\n"
                "Send a query as plain text (e.g. `best ramen in tokyo`), or JSON like:\n"
                '  {"query": "best ramen in tokyo", "temp_api_key": null}\n'
                "The environment will return top results.\n"
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
        if "top_k" in options:
            top_k = int(options["top_k"])
            if top_k <= 0:
                raise ValueError(f"top_k must be > 0; got {top_k}")
            self.top_k = top_k
        prompt = options.get("prompt", "Enter a web search query.")
        self._begin_episode()
        if not self._api_key:
            requests: list[Request] = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": -1.0,
                    "system_message": self._system_message(make_actor_id("agent")),
                    "message": {
                        "role": "user",
                        "content": "SERPER_API_KEY is not set. Cannot perform web search.",
                    },
                    "needs_action": False,
                    "info": {"error": "missing_api_key"},
                }
            ]
        else:
            requests: list[Request] = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": 0.0,
                    "system_message": self._system_message(make_actor_id("agent")),
                    "message": {"role": "user", "content": str(prompt)},
                    "needs_action": True,
                    "info": {"top_k": self.top_k},
                }
            ]
        done = False
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = True
            request["episode_end"] = done
        return requests

    def _search(self, query: str, *, api_key: str) -> dict:
        """Execute a Serper search query.

        Args:
            query: Search query string.
            api_key: Serper API key.

        Returns:
            Parsed JSON response.
        """
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": self.top_k}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def step(self, actions: dict[ActorId, str]) -> list[Request]:
        """Apply a search query and return results.

        Args:
            actions: Mapping of actor ids to action strings.

        Returns:
            Requests containing search results.
        """
        if not self._api_key:
            return []

        actions = self._normalize_actions(actions)
        raw = actions["agent"].strip()
        parsed = maybe_parse_json(raw)
        if isinstance(parsed, dict):
            query = str(parsed.get("query", "") or "").strip()
            api_key = str(parsed.get("temp_api_key") or self._api_key or "")
        else:
            query = raw
            api_key = str(self._api_key or "")

        if not query:
            raise ActionError("Empty query.")

        try:
            data = self._search(query, api_key=api_key)
        except urllib.error.URLError as exc:
            requests: list[Request] = [
                {
                    "actor": make_actor_id("agent"),
                    "reward": -1.0,
                    "message": {"role": "user", "content": f"Search failed: {exc}"},
                    "needs_action": False,
                    "info": {"error": str(exc), "query": query},
                }
            ]
            done = not requests or not any(r["needs_action"] for r in requests)
            for request in requests:
                request["episode_id"] = self._episode_id
                request["episode_start"] = False
                request["episode_end"] = done
            return requests

        organic = data.get("organic") or []
        lines = [f"Query: {query}", ""]
        for i, item in enumerate(organic[: self.top_k]):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"{i + 1}. {title}\n{link}\n{snippet}\n")

        requests: list[Request] = [
            {
                "actor": make_actor_id("agent"),
                "reward": 0.0,
                "message": {"role": "user", "content": "\n".join(lines).strip()},
                "needs_action": True,
                "info": {"query": query, "result_count": len(organic)},
            }
        ]
        done = not requests or not any(r["needs_action"] for r in requests)
        for request in requests:
            request["episode_id"] = self._episode_id
            request["episode_start"] = False
            request["episode_end"] = done
        return requests
