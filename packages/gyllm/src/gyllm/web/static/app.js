
(() => {
  const { useEffect, useRef, useState } = React;
  const html = htm.bind(React.createElement);

  async function api(path, options = {}) {
    const opts = { ...options };
    if (opts.body && typeof opts.body !== "string") {
      opts.body = JSON.stringify(opts.body);
    }
    opts.headers = opts.headers || {};
    if (opts.body) {
      opts.headers["Content-Type"] = "application/json";
    }
    const response = await fetch(path, opts);
    const contentType = response.headers.get("content-type") || "";
    let data = null;
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }
    if (!response.ok) {
      const detail =
        data && data.detail ? data.detail : typeof data === "string" ? data : "Request failed";
      const error = new Error(detail);
      error.payload = data;
      throw error;
    }
    return data;
  }

  function formatActor(actor) {
    if (!actor) {
      return "";
    }
    const parts = actor.split("::");
    const base = parts[0];
    const meta = parts.slice(1).filter(Boolean);
    if (!meta.length) {
      return base;
    }
    return `${base} (${meta.join(", ")})`;
  }

  function formatTime(ts) {
    if (!ts) {
      return "";
    }
    try {
      return new Date(ts * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (err) {
      return "";
    }
  }

  function getInitialTheme() {
    try {
      const saved = localStorage.getItem("gyllm-theme");
      if (saved === "light" || saved === "dark") {
        return saved;
      }
    } catch (err) {}
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }
    return "light";
  }

  function groupEventBlocks(events) {
    const blocks = [];
    let index = 0;
    while (index < events.length) {
      const event = events[index];
      const groupId = event.group_id;
      if (!groupId) {
        blocks.push({ kind: "single", event });
        index += 1;
        continue;
      }
      const grouped = [event];
      index += 1;
      while (index < events.length && events[index].group_id === groupId) {
        grouped.push(events[index]);
        index += 1;
      }
      blocks.push({ kind: "group", events: grouped });
    }
    return blocks;
  }

  function App() {
    const [envs, setEnvs] = useState([]);
    const [theme, setTheme] = useState(() => getInitialTheme());
    const [sessions, setSessions] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [activeSession, setActiveSession] = useState(null);
    const [events, setEvents] = useState([]);
    const [selectedEnv, setSelectedEnv] = useState("");
    const [draftsBySession, setDraftsBySession] = useState({});
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const [loading, setLoading] = useState(true);
    const scrollRef = useRef(null);

    useEffect(() => {
      loadInitial();
    }, []);

    useEffect(() => {
      document.documentElement.dataset.theme = theme;
      try {
        localStorage.setItem("gyllm-theme", theme);
      } catch (err) {}
    }, [theme]);

    useEffect(() => {
      if (activeId) {
        loadSession(activeId);
      } else {
        setActiveSession(null);
        setEvents([]);
      }
    }, [activeId]);

    useEffect(() => {
      if (!scrollRef.current) {
        return;
      }
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [events]);

    async function loadInitial() {
      setLoading(true);
      setError("");
      try {
        const [envData, sessionData] = await Promise.all([api("/api/envs"), api("/api/sessions")]);
        const envList = envData.envs || [];
        const sessionList = sessionData.sessions || [];
        setEnvs(envList);
        setSessions(sessionList);
        if (!selectedEnv && envList.length) {
          setSelectedEnv(envList[0]);
        }
        if (!activeId && sessionList.length) {
          setActiveId(sessionList[0].id);
        }
      } catch (err) {
        setError(err.message || "Failed to load data.");
      } finally {
        setLoading(false);
      }
    }

    async function loadSession(sessionId) {
      setError("");
      try {
        const data = await api(`/api/sessions/${sessionId}`);
        setActiveSession(data.session);
        setEvents(data.events || []);
      } catch (err) {
        setError(err.message || "Failed to load session.");
      }
    }

    function upsertSession(nextSession) {
      setSessions((prev) => {
        const filtered = prev.filter((item) => item.id !== nextSession.id);
        const merged = [nextSession, ...filtered];
        merged.sort((a, b) => b.created_at - a.created_at);
        return merged;
      });
    }

    function updateDraft(actor, value) {
      if (!activeId) {
        return;
      }
      setDraftsBySession((prev) => {
        const next = { ...prev };
        const sessionDrafts = { ...(next[activeId] || {}) };
        sessionDrafts[actor] = value;
        next[activeId] = sessionDrafts;
        return next;
      });
    }

    function clearDrafts(actors) {
      if (!activeId) {
        return;
      }
      setDraftsBySession((prev) => {
        const next = { ...prev };
        const sessionDrafts = { ...(next[activeId] || {}) };
        actors.forEach((actor) => {
          delete sessionDrafts[actor];
        });
        next[activeId] = sessionDrafts;
        return next;
      });
    }

    async function handleCreate() {
      if (!selectedEnv) {
        setError("Select an environment to start.");
        return;
      }
      setBusy(true);
      setError("");
      try {
        const data = await api("/api/sessions", {
          method: "POST",
          body: { env: selectedEnv },
        });
        upsertSession(data.session);
        setActiveId(data.session.id);
        setActiveSession(data.session);
        setEvents(data.events || []);
      } catch (err) {
        setError(err.message || "Failed to create session.");
      } finally {
        setBusy(false);
      }
    }

    async function handleReset() {
      if (!activeSession) {
        return;
      }
      setBusy(true);
      setError("");
      try {
        const data = await api(`/api/sessions/${activeSession.id}/reset`, {
          method: "POST",
          body: {},
        });
        upsertSession(data.session);
        setActiveSession(data.session);
        setEvents((prev) => prev.concat(data.events || []));
      } catch (err) {
        setError(err.message || "Failed to reset episode.");
      } finally {
        setBusy(false);
      }
    }

    async function handleSend() {
      if (!activeSession) {
        return;
      }
      const pending = activeSession.pending_actors || [];
      const drafts = draftsBySession[activeSession.id] || {};
      if (!pending.length) {
        setError("No pending actions.");
        return;
      }
      const missing = pending.filter((actor) => !drafts[actor] || !drafts[actor].trim());
      if (missing.length) {
        setError(`Missing actions for: ${missing.map(formatActor).join(", ")}`);
        return;
      }
      const actions = {};
      pending.forEach((actor) => {
        actions[actor] = drafts[actor];
      });
      setBusy(true);
      setError("");
      try {
        const data = await api(`/api/sessions/${activeSession.id}/step`, {
          method: "POST",
          body: { actions },
        });
        upsertSession(data.session);
        setActiveSession(data.session);
        setEvents((prev) => prev.concat(data.events || []));
        clearDrafts(pending);
      } catch (err) {
        setError(err.message || "Failed to send actions.");
      } finally {
        setBusy(false);
      }
    }

    function handleActionKey(event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!busy) {
          handleSend();
        }
      }
    }

    async function handleArchive() {
      if (!activeSession) {
        return;
      }
      setBusy(true);
      setError("");
      try {
        const data = await api(`/api/sessions/${activeSession.id}/archive`, { method: "POST" });
        upsertSession(data.session);
        setActiveSession(data.session);
        setEvents((prev) => prev.concat(data.events || []));
      } catch (err) {
        setError(err.message || "Failed to archive session.");
      } finally {
        setBusy(false);
      }
    }

    const pendingActors = activeSession ? activeSession.pending_actors || [] : [];
    const drafts = activeSession ? draftsBySession[activeSession.id] || {} : {};
    const nextTheme = theme === "dark" ? "light" : "dark";
    const nextThemeLabel = nextTheme === "dark" ? "Dark" : "Light";

    function renderEventCard(event, options = {}) {
      const { showActor = true } = options;
      const meta = event.meta || {};
      const bits = [];
      if (typeof meta.episode_id === "number") {
        bits.push(`episode ${meta.episode_id}`);
      }
      if (typeof meta.reward === "number") {
        bits.push(`reward ${meta.reward.toFixed(3)}`);
      }
      if (meta.needs_action) {
        bits.push("needs action");
      }
      const metaText = bits.join(" | ");
      const roleLabels = {
        env: "Environment",
        action: "Action",
        system: "System",
        notice: "Notice",
        error: "Error",
      };
      return html`
        <div className=${`message ${event.type}`} key=${event.id}>
          <div className="message-head">
            <span>${roleLabels[event.type] || "Event"}</span>
            ${showActor && event.actor
              ? html`<span className="message-actor" title=${event.actor}
                  >${formatActor(event.actor)}</span
                >`
              : ""}
            ${metaText ? html`<span className="message-meta">${metaText}</span>` : ""}
          </div>
          <div className="message-body">${event.content || ""}</div>
        </div>
      `;
    }

    function renderGroup(events) {
      const actorOrder = [];
      const byActor = new Map();
      events.forEach((event) => {
        const actor = event.actor || "unknown";
        if (!byActor.has(actor)) {
          byActor.set(actor, []);
          actorOrder.push(actor);
        }
        byActor.get(actor).push(event);
      });
      const groupKey = events[0] ? events[0].group_id || events[0].id : Math.random();
      return html`
        <div className="group-row" key=${groupKey}>
          ${actorOrder.map((actor) => {
            const actorEvents = byActor.get(actor) || [];
            return html`
              <div className="agent-column" key=${actor}>
                <div className="agent-header">
                  <span>Agent</span>
                  <span className="actor-label" title=${actor}>${formatActor(actor)}</span>
                </div>
                ${actorEvents.map((event) => renderEventCard(event, { showActor: false }))}
              </div>
            `;
          })}
        </div>
      `;
    }

    function renderEvent(event) {
      if (event.type === "divider") {
        return html`
          <div className="episode-divider" key=${event.id}>
            <div className="line"></div>
            <div className="prompt">
              <span>${event.content}</span>
              <button
                className="btn secondary"
                disabled=${busy || !activeSession || activeSession.archived}
                onClick=${handleReset}
              >
                Start new episode
              </button>
            </div>
            <div className="line"></div>
          </div>
        `;
      }
      return renderEventCard(event, { showActor: true });
    }

    return html`
      <div className="app">
        <header className="hero">
          <div>
            <h1>GYLLM Studio</h1>
            <p>Chat with environments, track episodes, and coordinate multi-agent turns.</p>
          </div>
          <div className="hero-actions">
            <button
              className="btn ghost theme-toggle"
              onClick=${() => setTheme(nextTheme)}
              aria-label="Toggle theme"
            >
              Switch to ${nextThemeLabel} mode
            </button>
            <div className="hero-badge">web interface</div>
          </div>
        </header>

        <section className="panel sessions">
          <div className="panel-header">
            <h2>Sessions</h2>
            <div className="row">
              <select
                className="select"
                value=${selectedEnv}
                onChange=${(event) => setSelectedEnv(event.target.value)}
                disabled=${busy}
              >
                ${envs.map(
                  (env) => html`<option value=${env} key=${env}>${env}</option>`,
                )}
              </select>
              <button className="btn" onClick=${handleCreate} disabled=${busy || !selectedEnv}>
                New session
              </button>
            </div>
          </div>

          <div className="session-list">
            ${loading
              ? html`<div className="empty-state">Loading sessions...</div>`
              : sessions.length
                ? sessions.map((session) => {
                    const isActive = activeId === session.id;
                    return html`
                      <div
                        className=${`session-card ${isActive ? "active" : ""} ${
                          session.archived ? "archived" : ""
                        }`}
                        key=${session.id}
                        onClick=${() => setActiveId(session.id)}
                      >
                        <div className="session-title">${session.env}</div>
                        <div className="session-meta">
                          <span>${formatTime(session.created_at)}</span>
                          <span>${session.archived ? "archived" : "live"}</span>
                        </div>
                      </div>
                    `;
                  })
                : html`<div className="empty-state">No sessions yet.</div>`}
          </div>
        </section>

        <section className="panel chat-panel">
          <div className="chat-header">
            <div className="chat-title">
              <h2>${activeSession ? activeSession.env : "Select a session"}</h2>
              <div className="pill">
                ${activeSession
                  ? activeSession.archived
                    ? "Archived"
                    : `Pending actions: ${pendingActors.length}`
                  : "No session active"}
              </div>
            </div>
            <div className="row">
              <button
                className="btn secondary"
                onClick=${handleReset}
                disabled=${busy || !activeSession || activeSession.archived}
              >
                Start episode
              </button>
              <button
                className="btn ghost"
                onClick=${handleArchive}
                disabled=${busy || !activeSession || activeSession.archived}
              >
                Archive session
              </button>
            </div>
          </div>

          ${error ? html`<div className="banner">${error}</div>` : ""}

          <div className="chat-body" ref=${scrollRef}>
            ${activeSession
              ? events.length
                ? groupEventBlocks(events).map((block) =>
                    block.kind === "group" ? renderGroup(block.events) : renderEvent(block.event),
                  )
                : html`<div className="empty-state">Start an episode to see messages.</div>`
              : html`<div className="empty-state">Pick a session to begin.</div>`}
          </div>

          <div className="action-bar">
            ${activeSession
              ? activeSession.archived
                ? html`<div className="empty-state">This session is archived.</div>`
                : pendingActors.length
                  ? html`
                      <div className="action-grid">
                        ${pendingActors.map(
                          (actor) => html`
                            <div className="action-card" key=${actor}>
                              <label title=${actor}>${formatActor(actor) || actor}</label>
                              <textarea
                                value=${drafts[actor] || ""}
                                onInput=${(event) => updateDraft(actor, event.target.value)}
                                onKeyDown=${handleActionKey}
                                placeholder="Type action for this agent..."
                                rows="3"
                              ></textarea>
                            </div>
                          `,
                        )}
                      </div>
                      <div className="row">
                        <button className="btn" onClick=${handleSend} disabled=${busy}>
                          Send actions
                        </button>
                      </div>
                    `
                  : html`<div className="empty-state">No pending actions. Start or continue an episode.</div>`
              : ""}
          </div>
        </section>
      </div>
    `;
  }

  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(html`<${App} />`);
})();
