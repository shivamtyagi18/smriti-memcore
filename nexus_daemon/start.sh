#!/bin/bash
# Resolve the project root relative to where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

echo "🧠 Activating virtual environment..."
source .venv_langchain/bin/activate

echo "🧹 Ensuring port 8000 is perfectly clear (graceful shutdown)..."
# SIGTERM allows Python atexit/lifespan handlers to flush memory to disk
OLD_PID=$(lsof -ti:8000)
if [ -n "$OLD_PID" ]; then
    kill $OLD_PID 2>/dev/null
    sleep 2  # Give existing server time to save state gracefully
fi

echo "⚙️ Booting NEXUS Local Assistant Daemon in the background..."
nohup python3 "$SCRIPT_DIR/server.py" > "$PROJECT_ROOT/daemon.log" 2>&1 &
sleep 2
echo "✅ Backend live at http://127.0.0.1:8000  |  Logs: $PROJECT_ROOT/daemon.log"

echo "🚀 Daemon is permanently online! Launching your chat client..."
python3 nexus_daemon/chat.py