import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi.responses import StreamingResponse

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from nexus import NEXUS, NexusConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus-daemon")

@asynccontextmanager
async def lifespan(app):
    # Startup
    yield
    # Shutdown: flush all NEXUS state to disk
    logger.info("Shutting down — saving NEXUS state to disk...")
    nexus_engine.save()
    logger.info("NEXUS state saved successfully.")

# Core App
app = FastAPI(title="NEXUS Local Assistant API", lifespan=lifespan)

# Setup paths
DAEMON_DIR = os.path.expanduser("~/.nexus_agent")
os.makedirs(DAEMON_DIR, exist_ok=True)
DB_PATH = os.path.join(DAEMON_DIR, "nexus_memory")
CHAT_DB = f"sqlite:///{os.path.join(DAEMON_DIR, 'chat_history.db')}"
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)

# Initialize NEXUS
config = NexusConfig(
    storage_path=DB_PATH,
    llm_model="mistral", # default
)

if os.getenv("OPENAI_API_KEY"):
    config.llm_model = "gpt-4o-mini"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    logger.info("Using OpenAI GPT-4o-mini")
else:
    llm = ChatOllama(model="mistral", temperature=0.7)
    logger.info("Using Local Ollama (Mistral)")

nexus_engine = NEXUS(config=config)

# Build Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly intelligent and persistent AI assistant named NEXUS.\n"
               "You have a perfect, long-term memory of all your conversations with the user.\n"
               "Here is relevant context retrieved from your Semantic Palace for this interaction:\n\n"
               "{nexus_context}\n\n"
               "Answer the user's inquiry naturally and concisely. Do not explicitly state that you are retrieving memories unless asked."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id, CHAT_DB)

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str = "default_user_session"
    message: str

class ChatResponse(BaseModel):
    response: str
    processed_in_background: bool = True

class StatsResponse(BaseModel):
    storage_path: str
    episodes_count: int
    palace_rooms: int

class SettingsRequest(BaseModel):
    model_provider: str

# --- Endpoints ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # Load the latest context from NEXUS using its Dual-Process semantic retrieval
        memories = nexus_engine.recall(request.message, top_k=5)
        episodes = nexus_engine.episode_buffer.search_semantic(request.message, top_k=5)
        
        context_blocks = []
        if memories:
            context_blocks.append("Abstract Knowledge:\n" + "\n".join(f"- {m.content}" for m in memories))
        if episodes:
            context_blocks.append("Specific Past Events:\n" + "\n".join(f"- {ep.content}" for ep in episodes))
            
        nexus_context_str = "Relevant Long-Term Memories:\n\n" + "\n\n".join(context_blocks) if context_blocks else "No relevant past memories found."
        
        # Invoke the LangChain with SQLite history
        response = chain_with_history.invoke(
            {"input": request.message, "nexus_context": nexus_context_str},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        msg_content = response.content if hasattr(response, "content") else str(response)
        
        # Add the raw interaction to NEXUS System 1 (Episode Buffer)
        nexus_engine.encode(f"Human: {request.message}")
        nexus_engine.encode(f"AI: {msg_content}")
        
        # Defer System 2 Consolidation to Background
        background_tasks.add_task(nexus_engine.consolidate, depth="light")
        
        return ChatResponse(response=msg_content)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    stats = nexus_engine.stats()
    return StatsResponse(
        storage_path=DB_PATH,
        episodes_count=stats["episode_buffer"]["total_episodes"],
        palace_rooms=stats["palace"]["room_count"]
    )

@app.post("/save")
def force_save():
    nexus_engine.save()
    return {"status": "saved"}

@app.get("/api/memories")
def get_memories():
    from nexus.models import MemoryStatus
    # Filter out explicitly archived (forgotten) memories
    memories = [
        {"id": m.id, "content": m.content, "confidence": m.confidence}
        for m in nexus_engine.palace.memories.values()
        if m.status != MemoryStatus.ARCHIVED
    ]
    return {"memories": memories}

@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str):
    nexus_engine.forget(memory_id)
    return {"status": "success"}

@app.get("/api/data")
def get_all_data():
    """Returns all three persistence layers for the Data Management tab."""
    import sqlite3

    # 1. Palace memories (System 2 — abstract)
    palace_memories = []
    try:
        palace_memories = [
            {
                "id": m.id, "content": m.content, "confidence": round(m.confidence, 2),
                "status": m.status.value if hasattr(m.status, "value") else str(m.status),
                "modality": m.modality.value if hasattr(m.modality, "value") else str(m.modality),
            }
            for m in nexus_engine.palace.memories.values()
        ]
    except Exception as e:
        palace_memories = [{"error": str(e)}]

    # 2. Episodes — read directly from SQLite
    episodes = []
    try:
        ep_db = nexus_engine.episode_buffer._db_path
        conn = sqlite3.connect(ep_db)
        cur = conn.execute(
            "SELECT id, content, salience_json, consolidated FROM episodes ORDER BY timestamp DESC"
        )
        for row in cur.fetchall():
            salience = None
            try:
                s = json.loads(row[2] or "{}")
                salience = round(s.get("composite", 0), 3)
            except Exception:
                pass
            episodes.append({
                "id": row[0], "content": row[1],
                "salience": salience, "consolidated": bool(row[3])
            })
        conn.close()
    except Exception as e:
        episodes = [{"error": str(e)}]

    # 3. Chat history from SQLite
    chat_log = []
    try:
        db_path = CHAT_DB.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT session_id, message_type, content, created_at FROM message_store ORDER BY created_at"
        )
        for row in cur.fetchall():
            chat_log.append({"session_id": row[0], "role": row[1], "content": row[2], "timestamp": row[3]})
        conn.close()
    except Exception as e:
        chat_log = [{"error": str(e)}]

    return {
        "palace_memories": palace_memories,
        "episodes": episodes,
        "chat_history": chat_log,
    }

@app.get("/api/settings")
def get_settings():
    model_name = "gpt-4o-mini" if isinstance(llm, ChatOpenAI) else "mistral"
    return {"active_model": model_name, "openai_available": bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/api/settings")
def update_settings(request: SettingsRequest):
    global llm, chain, chain_with_history
    if request.model_provider == "gpt-4o-mini":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY is not set in this server's environment. "
                       "Restart the server with the key exported: export OPENAI_API_KEY=sk-..."
            )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    else:
        llm = ChatOllama(model="mistral", temperature=0.7)
        
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    logger.info(f"Model switched to: {request.model_provider}")
    return {"status": "success", "active_model": request.model_provider}

@app.get("/api/logs")
def stream_logs():
    """SSE endpoint: tails daemon.log and streams new lines to the browser."""
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "daemon.log")
    log_file = os.path.normpath(log_file)

    def event_generator():
        try:
            with open(log_file, "r") as f:
                # Seek to end so we only stream new lines
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        # Strip ANSI progress bars / carriage returns
                        clean = line.replace("\r", "").strip()
                        if clean:
                            yield f"data: {clean}\n\n"
                    else:
                        time.sleep(0.3)
        except FileNotFoundError:
            yield "data: [daemon.log not found — start the server via start.sh]\n\n"
        except GeneratorExit:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount static web directory
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
