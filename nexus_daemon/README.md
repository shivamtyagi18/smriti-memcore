# NEXUS Local Assistant Daemon

Unlike the core `nexus/` engine (benchmarks and research), this `nexus_daemon/` module transforms NEXUS into a **persistent, asynchronous background AI assistant** that lives natively on your Mac 24/7.

## Architecture Highlights
- **FastAPI Backend Server** (`server.py`): Hosts the LLM and vector search. Maintains LangChain chat history in SQLite.
- **Full Web UI** (`web/`): Browse and delete memories, chat, hot-swap LLMs, and stream live backend logs — all from `http://127.0.0.1:8000`.
- **Background Consolidation**: After each reply, the daemon silently triggers `nexus.consolidate("light")` in a background thread.
- **Persistent Storage**: All memories (episodic + abstract) are saved to `~/.nexus_agent/` on graceful shutdown. **Important**: always use `SIGTERM` (not `kill -9`) to stop the server so memories are flushed to disk.

---

## 🚀 Quickstart (Recommended)

Run from the **project root** (`Memory/`) directory:

```bash
./nexus_daemon/start.sh
```

This script:
1. Activates the Python virtual environment (`.venv_langchain`)
2. Gracefully shuts down any existing server (SIGTERM, so memories are saved)
3. Boots the daemon in the background and writes logs to `daemon.log`
4. Opens the terminal Chat UI

Then open your browser at **http://127.0.0.1:8000** for the full Web UI.

---

## 🧠 LLM Model Selection

By default the daemon uses **local Ollama (Mistral)**.

To use **OpenAI GPT-4o-mini** instead, export your API key **before** starting:
```bash
export OPENAI_API_KEY=sk-...
./nexus_daemon/start.sh
```

You can also switch models live from the **Web UI dropdown** without restarting the server.

---

## 🛠️ Manual Start (For Live Log Viewing)

**Terminal 1 — Server:**
```bash
source .venv_langchain/bin/activate
python3 nexus_daemon/server.py
```

**Terminal 2 — Chat Client:**
```bash
source .venv_langchain/bin/activate
python3 nexus_daemon/chat.py
```

---

## 📋 Monitoring
- **Live Logs (terminal):** `tail -f daemon.log`
- **Live Logs (Web UI):** Click the **📋 Logs** button in the browser dashboard header
- **Check active model:** `grep 'Using' daemon.log`

---

## 💬 Terminal Chat Commands
- `/stats` — Show episodes and palace room count
- `/quit` or `/exit` — Safely close (flushes memory to disk)

