from __future__ import annotations

import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from gyllm import list_envs, make
from gyllm.core import ActionError, LLMEnv, Request


@dataclass(slots=True)
class SessionState:
    id: str
    env_name: str
    env: LLMEnv | None
    created_at: float
    archived: bool = False
    actors: list[str] = field(default_factory=list)
    pending_actors: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1
    next_group_id: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        env_name: str,
        env_kwargs: dict[str, Any] | None,
        num_envs: int | None,
        validate_actions: bool,
        autoreset: bool,
    ) -> SessionState:
        env = make(
            env_name,
            mode="local",
            env_kwargs=env_kwargs,
            num_envs=num_envs,
            validate_actions=validate_actions,
            autoreset=autoreset,
        )
        session_id = uuid.uuid4().hex
        state = SessionState(
            id=session_id,
            env_name=env_name,
            env=env,
            created_at=time.time(),
            actors=list(env.actors),
        )
        with self._lock:
            self._sessions[session_id] = state
        return state

    def list_sessions(self) -> list[SessionState]:
        with self._lock:
            return list(self._sessions.values())

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def archive(self, session_id: str) -> SessionState:
        session = self.get_or_raise(session_id)
        with session.lock:
            if session.archived:
                return session
            session.archived = True
            if session.env is not None:
                session.env.close()
                session.env = None
            session.pending_actors = []
        return session

    def get_or_raise(self, session_id: str) -> SessionState:
        session = self.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def close_all(self) -> None:
        for session in self.list_sessions():
            with session.lock:
                if session.env is not None:
                    session.env.close()
                    session.env = None


class CreateSessionRequest(BaseModel):
    env: str = Field(..., min_length=1)
    env_kwargs: dict[str, Any] | None = None
    num_envs: int | None = Field(default=None, ge=1)
    validate_actions: bool = True
    autoreset: bool = False


class ResetRequest(BaseModel):
    options: dict[str, Any] | None = None


class StepRequest(BaseModel):
    actions: dict[str, str]


@cache
def _load_static_text(name: str) -> str:
    return resources.files("gyllm.web").joinpath("static", name).read_text(encoding="utf-8")


def _session_summary(session: SessionState) -> dict[str, Any]:
    return {
        "id": session.id,
        "env": session.env_name,
        "created_at": session.created_at,
        "archived": session.archived,
        "actors": list(session.actors),
        "pending_actors": list(session.pending_actors),
    }


def _append_event(
    session: SessionState,
    new_events: list[dict[str, Any]],
    *,
    event_type: str,
    content: str,
    actor: str | None = None,
    meta: dict[str, Any] | None = None,
    group_id: int | None = None,
) -> None:
    event = {
        "id": session.next_event_id,
        "type": event_type,
        "actor": actor,
        "content": content,
        "meta": meta or {},
        "group_id": group_id,
    }
    session.next_event_id += 1
    session.events.append(event)
    new_events.append(event)


def _next_group_id(session: SessionState) -> int:
    group_id = session.next_group_id
    session.next_group_id += 1
    return group_id


def _request_meta(request: Request) -> dict[str, Any]:
    episode_id = request.get("episode_id")
    return {
        "reward": float(request.get("reward", 0.0)),
        "needs_action": bool(request.get("needs_action", False)),
        "episode_id": int(episode_id) if episode_id is not None else None,
        "episode_start": bool(request.get("episode_start", False)),
        "episode_end": bool(request.get("episode_end", False)),
    }


def _apply_requests(session: SessionState, requests: list[Request]) -> list[dict[str, Any]]:
    new_events: list[dict[str, Any]] = []
    group_id = _next_group_id(session) if requests else None
    for request in requests:
        meta = _request_meta(request)
        system_message = request.get("system_message")
        if system_message is not None:
            _append_event(
                session,
                new_events,
                event_type="system",
                actor=request.get("actor"),
                content=str(system_message.get("content", "")),
                meta=meta,
                group_id=group_id,
            )
        message = request.get("message") or {}
        _append_event(
            session,
            new_events,
            event_type="env",
            actor=request.get("actor"),
            content=str(message.get("content", "")),
            meta=meta,
            group_id=group_id,
        )
    pending: list[str] = []
    seen: set[str] = set()
    for req in requests:
        if req.get("needs_action"):
            actor = req["actor"]
            if actor not in seen:
                seen.add(actor)
                pending.append(actor)
    session.pending_actors = pending
    if any(req.get("episode_end") for req in requests):
        _append_event(
            session,
            new_events,
            event_type="divider",
            content="Episode ended. Start a new episode?",
        )
    return new_events


def _append_actions(session: SessionState, actions: dict[str, str]) -> list[dict[str, Any]]:
    new_events: list[dict[str, Any]] = []
    group_id = _next_group_id(session) if actions else None
    for actor in sorted(actions):
        _append_event(
            session,
            new_events,
            event_type="action",
            actor=actor,
            content=str(actions[actor]),
            group_id=group_id,
        )
    return new_events


def create_app() -> FastAPI:
    store = SessionStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            store.close_all()

    app = FastAPI(title="GYLLM Web", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _load_static_text("index.html")

    @app.get("/assets/styles.css")
    def styles() -> Response:
        return Response(content=_load_static_text("styles.css"), media_type="text/css")

    @app.get("/assets/app.js")
    def app_js() -> Response:
        return Response(content=_load_static_text("app.js"), media_type="text/javascript")

    @app.get("/api/envs")
    def envs() -> dict[str, Any]:
        return {"envs": list_envs()}

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        sessions = sorted(store.list_sessions(), key=lambda item: item.created_at, reverse=True)
        return {"sessions": [_session_summary(session) for session in sessions]}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            session = store.get_or_raise(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return {"session": _session_summary(session), "events": list(session.events)}

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest) -> dict[str, Any]:
        try:
            session = store.create(
                env_name=request.env,
                env_kwargs=request.env_kwargs,
                num_envs=request.num_envs,
                validate_actions=request.validate_actions,
                autoreset=request.autoreset,
            )
        except (ImportError, KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"session": _session_summary(session), "events": []}

    @app.post("/api/sessions/{session_id}/reset")
    def reset_session(session_id: str, request: ResetRequest) -> dict[str, Any]:
        try:
            session = store.get_or_raise(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        with session.lock:
            if session.archived or session.env is None:
                raise HTTPException(status_code=400, detail="Session is archived")
            try:
                requests = session.env.reset(options=request.options)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            new_events = _apply_requests(session, requests)
        return {"session": _session_summary(session), "events": new_events}

    @app.post("/api/sessions/{session_id}/step")
    def step_session(session_id: str, request: StepRequest) -> dict[str, Any]:
        try:
            session = store.get_or_raise(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        with session.lock:
            if session.archived or session.env is None:
                raise HTTPException(status_code=400, detail="Session is archived")
            try:
                requests = session.env.step(request.actions)
            except ActionError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            new_events = _append_actions(session, request.actions)
            new_events.extend(_apply_requests(session, requests))
        return {"session": _session_summary(session), "events": new_events}

    @app.post("/api/sessions/{session_id}/archive")
    def archive_session(session_id: str) -> dict[str, Any]:
        try:
            session = store.archive(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        new_events: list[dict[str, Any]] = []
        with session.lock:
            _append_event(
                session,
                new_events,
                event_type="notice",
                content="Session archived. The environment has been closed.",
            )
        return {"session": _session_summary(session), "events": new_events}

    return app


app = create_app()
