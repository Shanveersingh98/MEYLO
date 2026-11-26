"""
src/api_server.py
=================
Production-grade Local FastAPI Backend for Local AI Companion.
Connects the modern React frontend to the existing dual-model AI inference engine,
SQLite database, and web retrieval pipeline.

Features:
- Full REST API for conversations, messages, models, web search, system hardware.
- SSE Streaming endpoint for smooth real-time generation in the UI.
- Native execution of slash commands (/model, /web, /sources, /clear, /help, /exit).
- Strict model independence: Custom (FromScratchLM) and External (Qwen2.5-1.5B).
- Serves pre-built React frontend static files from frontend/dist with SPA fallback.
- Bound strictly to localhost (127.0.0.1) for local privacy.
"""

import sys
import os
import time
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model import FromScratchLM
from model_manager import ModelManager, get_hardware_info
from web_retrieval import WebRetrievalEngine, EvidenceItem
from companion_database import CompanionDatabase
from chat import generate_chat_response, parse_command, format_source_links

# Initialize FastAPI App
app = FastAPI(
    title="MEYLO API",
    description="Local backend serving the MEYLO personal AI companion interface",
    version="2.0.0"
)

# CORS middleware for local frontend development (e.g. Vite on port 5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class BackendState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_manager: Optional[ModelManager] = None
        self.retrieval_engine: Optional[WebRetrievalEngine] = None
        self.db: Optional[CompanionDatabase] = None
        self.web_search_enabled: bool = False
        self.debug_mode: bool = False
        self.is_ready: bool = False

state = BackendState()


@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print("Initializing MEYLO Backend Server...")
    print(f"Project Directory : {PROJECT_DIR}")
    print(f"Hardware Device   : {state.device}")
    
    # Record PID for clean process management
    try:
        pid_file = PROJECT_DIR / "logs" / "server.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # 1. Initialize SQLite Database
    state.db = CompanionDatabase()
    print("[Database] SQLite database connection verified with WAL mode.")
    
    # 2. Initialize Model Manager (defaults to custom model)
    state.model_manager = ModelManager(device=state.device, default_model="custom")
    print(f"[Model Manager] Active model initialized: {state.model_manager.active_model_name}")
    
    # 3. Initialize Web Retrieval Engine
    state.retrieval_engine = WebRetrievalEngine()
    state.retrieval_engine.session_last_sources = []
    print("[Web Engine] Web retrieval engine initialized.")
    
    state.is_ready = True
    print("=" * 60)
    print("MEYLO Backend is Ready on http://127.0.0.1:8000")

    if "--open-browser" in sys.argv:
        import threading
        import webbrowser
        def _open_browser():
            time.sleep(0.8)
            webbrowser.open("http://127.0.0.1:8000")
        threading.Thread(target=_open_browser, daemon=True).start()



@app.on_event("shutdown")
def shutdown_event():
    print("[Shutdown] Shutting down MEYLO services...")
    if state.model_manager:
        state.model_manager.shutdown()
    if state.db:
        state.db.close()
    try:
        pid_file = PROJECT_DIR / "logs" / "server.pid"
        if pid_file.exists():
            pid_file.unlink()
    except Exception:
        pass
    print("[Shutdown] Clean shutdown complete.")


# -------------------------------------------------------------
# Request & Response Models
# -------------------------------------------------------------
class SwitchModelRequest(BaseModel):
    model: str = Field(..., description="'custom' or 'external'")

class WebSearchToggleRequest(BaseModel):
    enabled: bool

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = "custom"

class RenameConversationRequest(BaseModel):
    title: str

class ChatMessageRequest(BaseModel):
    conversation_id: str
    message: str
    model: Optional[str] = None  # If specified and differs from active, switches model


# -------------------------------------------------------------
# System & Status Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend readiness."""
    active_name = state.model_manager.active_model_name if state.model_manager else "unknown"
    return {
        "status": "ok" if state.is_ready else "starting",
        "ready": state.is_ready,
        "active_model": active_name,
        "web_search_enabled": state.web_search_enabled,
        "device": str(state.device),
        "version": "2.0.0"
    }


@app.get("/api/status")
def get_system_status():
    """Returns dynamic hardware status (CPU, GPU, RAM, VRAM, CUDA) and model info."""
    hw = get_hardware_info()
    active_model = state.model_manager.active_model_name if state.model_manager else "custom"
    
    active_desc = {
        "id": active_model,
        "name": "FromScratchLM (Custom Trained)" if active_model == "custom" else "Qwen2.5-1.5B-Instruct (External)",
        "type": "Custom PyTorch Neural Model" if active_model == "custom" else "External Pretrained GGUF",
        "file": "checkpoints/ultimate_improvement/best.pt" if active_model == "custom" else "models/external/qwen2.5-1.5b-instruct/model.gguf",
        "execution_device": f"{hw['gpu_name']} (CUDA {hw['cuda_version']})" if (active_model == "custom" and hw["cuda_available"]) else f"{hw['cpu_name']} (CPU)",
        "inference_engine": "PyTorch Autoregressive Decoder" if active_model == "custom" else "llama.cpp (portable local runtime)"
    }

    return {
        "hardware": hw,
        "active_model": active_desc,
        "web_search_enabled": state.web_search_enabled,
        "debug_mode": state.debug_mode,
        "status_text": state.model_manager.get_status(state.web_search_enabled) if state.model_manager else ""
    }


# -------------------------------------------------------------
# Model Management Endpoints
# -------------------------------------------------------------
@app.get("/api/models")
def get_available_models():
    """Lists available models with current active selection."""
    hw = get_hardware_info()
    active_model = state.model_manager.active_model_name if state.model_manager else "custom"
    
    models = [
        {
            "id": "custom",
            "name": "My Custom Model (FromScratchLM)",
            "description": "Trained from scratch PyTorch transformer. Optimized for companionship, reasoning, and education.",
            "checkpoint": "checkpoints/ultimate_improvement/best.pt",
            "runtime": "PyTorch CUDA/CPU",
            "device": f"{hw['gpu_name']} (CUDA)" if hw["cuda_available"] else "CPU",
            "active": (active_model == "custom"),
            "badge": "From Scratch LM"
        },
        {
            "id": "external",
            "name": "External Model (Qwen2.5-1.5B)",
            "description": "Pretrained Qwen2.5-1.5B-Instruct running via portable llama.cpp CPU runtime (GGUF Q4_K_M).",
            "checkpoint": "models/external/qwen2.5-1.5b-instruct/model.gguf",
            "runtime": "llama.cpp CPU",
            "device": f"{hw['cpu_name']} (Multi-Threaded CPU)",
            "active": (active_model == "external"),
            "badge": "Qwen 1.5B GGUF"
        }
    ]
    return {
        "models": models,
        "active_model": active_model
    }


@app.post("/api/models/switch")
def switch_model(req: SwitchModelRequest):
    """
    Switches active model cleanly between 'custom' and 'external'.
    Guarantees active model is updated ONLY after confirmed backend switch.
    """
    if not state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized.")
    
    target = req.model.strip().lower()
    if target not in ["custom", "external"]:
        raise HTTPException(status_code=400, detail=f"Invalid model choice '{req.model}'. Must be 'custom' or 'external'.")
    
    success, msg = state.model_manager.set_model(target)
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    
    return {
        "success": True,
        "active_model": state.model_manager.active_model_name,
        "message": msg
    }


@app.get("/api/capabilities")
def get_verified_capabilities():
    """
    Returns empirical capability benchmark results for both models across 12 categories,
    reflecting real offline and online inference testing.
    """
    cap_file = PROJECT_DIR / "data" / "verified_model_capabilities.json"
    if cap_file.exists():
        try:
            with open(cap_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load capabilities: {e}"}
    return {"custom": {}, "external": {}}


# -------------------------------------------------------------
# Web Search Endpoints
# -------------------------------------------------------------
@app.get("/api/web-search")
def get_web_search_status():
    """Returns current web search toggle state."""
    return {"enabled": state.web_search_enabled}


@app.post("/api/web-search")
def set_web_search_status(req: WebSearchToggleRequest):
    """Enables or disables live web search retrieval."""
    state.web_search_enabled = req.enabled
    if not req.enabled and state.retrieval_engine:
        state.retrieval_engine.session_last_sources = []
    return {
        "enabled": state.web_search_enabled,
        "message": f"Web search is now {'ON (Online mode)' if state.web_search_enabled else 'OFF (Pure offline mode)'}."
    }


@app.get("/api/sources")
def get_recent_sources():
    """Returns sources used in the most recent web-grounded answer."""
    raw_sources = getattr(state.retrieval_engine, "session_last_sources", []) or []
    formatted = []
    seen = set()
    for s in raw_sources:
        title = getattr(s, "title", str(s)).strip()
        url = getattr(s, "url", "").strip()
        snippet = getattr(s, "snippet", "") or ""
        publisher = getattr(s, "publisher", None)
        date = getattr(s, "date", None)
        key = (title, url)
        if key not in seen:
            seen.add(key)
            formatted.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "publisher": publisher,
                "date": date
            })
    return {
        "sources": formatted,
        "count": len(formatted),
        "text": format_source_links(raw_sources)
    }


# -------------------------------------------------------------
# Conversation Management Endpoints
# -------------------------------------------------------------
@app.get("/api/conversations")
def list_conversations(search: Optional[str] = None):
    """Lists conversations with message counts and last activity."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    conversations = state.db.list_conversations(search=search)
    return {"conversations": conversations}


@app.post("/api/conversations")
def create_conversation(req: CreateConversationRequest):
    """Creates a new conversation session."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    active_model = state.model_manager.active_model_name if state.model_manager else "custom"
    conv_id = state.db.create_conversation(
        title=req.title or "New Chat",
        model=req.model or active_model
    )
    return {
        "conversation_id": conv_id,
        "title": req.title or "New Chat",
        "model": req.model or active_model
    }


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str):
    """Retrieves conversation metadata and chronological message history."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    conv = state.db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    
    raw_messages = state.db.get_messages(conv_id)
    messages = []
    for m in raw_messages:
        sources = []
        if m.get("sources_json"):
            try:
                sources = json.loads(m["sources_json"])
            except Exception:
                sources = []
        messages.append({
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "model_used": m.get("model_used"),
            "route": m.get("route"),
            "sources": sources,
            "latency_ms": m.get("latency_ms"),
            "timestamp": m.get("timestamp")
        })
        
    return {
        "conversation": conv,
        "messages": messages
    }


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: str, req: RenameConversationRequest):
    """Renames an existing conversation."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    ok = state.db.rename_conversation(conv_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found or rename failed.")
    return {"success": True, "title": req.title.strip()}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    """Deletes a conversation and its messages."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    ok = state.db.delete_conversation(conv_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"success": True, "message": "Conversation deleted."}


@app.post("/api/conversations/{conv_id}/clear")
def clear_conversation(conv_id: str):
    """Clears messages for a specific conversation."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    state.db.clear_conversation_messages(conv_id)
    if hasattr(state.retrieval_engine, "conversation_memory"):
        state.retrieval_engine.conversation_memory.clear()
    state.retrieval_engine.session_last_sources = []
    return {"success": True, "message": "Conversation history cleared."}


@app.delete("/api/conversations")
def delete_all_conversations():
    """Deletes all conversations and clears memory."""
    if not state.db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    state.db.delete_all_conversations()
    if hasattr(state.retrieval_engine, "conversation_memory"):
        state.retrieval_engine.conversation_memory.clear()
    state.retrieval_engine.session_last_sources = []
    return {"success": True, "message": "All conversations deleted."}


# -------------------------------------------------------------
# Slash Command & Chat Completion Handler
# -------------------------------------------------------------
def execute_slash_command(command_action: str, raw_input: str, conv_id: str) -> Dict[str, Any]:
    """
    Executes slash commands using the exact same backend logic as the terminal.
    Returns response dictionary with role='system'.
    """
    feedback = ""
    active_model = state.model_manager.active_model_name

    if command_action == "exit":
        feedback = "The application session is closing. You can safely close this browser tab."
    elif command_action == "model_custom":
        ok, msg = state.model_manager.set_model("custom")
        feedback = msg
        active_model = state.model_manager.active_model_name
    elif command_action == "model_external":
        ok, msg = state.model_manager.set_model("external")
        feedback = msg
        active_model = state.model_manager.active_model_name
    elif command_action == "model_status":
        feedback = state.model_manager.get_status(state.web_search_enabled)
    elif command_action == "model_list":
        feedback = state.model_manager.list_models()
    elif command_action == "clear":
        state.db.clear_conversation_messages(conv_id)
        if hasattr(state.retrieval_engine, "conversation_memory"):
            state.retrieval_engine.conversation_memory.clear()
        state.retrieval_engine.session_last_sources = []
        feedback = "[History cleared.]"
    elif command_action == "web_on":
        state.web_search_enabled = True
        feedback = "[Web Search: ON]"
    elif command_action == "web_off":
        state.web_search_enabled = False
        if state.retrieval_engine:
            state.retrieval_engine.session_last_sources = []
        feedback = "[Web Search: OFF]"
    elif command_action == "web_status":
        feedback = f"[Web Search: {'ON' if state.web_search_enabled else 'OFF'}]"
    elif command_action == "sources":
        feedback = format_source_links(getattr(state.retrieval_engine, "session_last_sources", []))
    elif command_action == "debug_on":
        state.debug_mode = True
        feedback = "[Debug Logging: ENABLED -> logs/router_debug.log]"
    elif command_action == "debug_off":
        state.debug_mode = False
        feedback = "[Debug Logging: DISABLED]"
    elif command_action == "debug_status":
        feedback = f"[Debug Logging: {'ENABLED' if state.debug_mode else 'DISABLED'}]"
    elif command_action == "help":
        feedback = (
            "**Available Slash Commands:**\n\n"
            "• `/model custom` — Switch to Custom Trained Model (`best.pt`, PyTorch CUDA/CPU)\n"
            "• `/model external` — Switch to External Qwen Model (`Qwen2.5-1.5B-Instruct`)\n"
            "• `/model status` — Show active model, hardware specs, RAM and VRAM usage\n"
            "• `/model list` — List all available models and their status\n"
            "• `/web on` — Enable live web search retrieval\n"
            "• `/web off` — Disable live web search retrieval (pure offline mode)\n"
            "• `/web status` — Display whether web search is ON or OFF\n"
            "• `/sources` — Display source links used in the recent response\n"
            "• `/debug on` / `/debug off` / `/debug status` — Toggle diagnostic router logging\n"
            "• `/clear` — Clear current conversation history\n"
            "• `/help` — Display this help menu\n"
            "• `/exit` — Safely shut down the application"
        )
    elif command_action == "unknown_command":
        feedback = f"Unrecognized command: `{raw_input}`. Type `/help` for available commands."
    else:
        feedback = f"Unknown command action '{command_action}'."

    # Record command and system reply in conversation
    state.db.add_message(conv_id, "user", raw_input, model_used=active_model)
    msg_id = state.db.add_message(
        conv_id,
        "assistant",
        feedback,
        model_used=active_model,
        route=f"slash_command_{command_action}",
        latency_ms=0.0
    )

    return {
        "id": msg_id,
        "conversation_id": conv_id,
        "role": "assistant",
        "content": feedback,
        "model_used": active_model,
        "route": f"slash_command_{command_action}",
        "sources": [],
        "latency_ms": 0.0,
        "is_command": True
    }


@app.post("/api/chat")
def handle_chat_message(req: ChatMessageRequest):
    """
    Main chat interaction endpoint:
    1. Checks for and executes slash commands directly.
    2. Optional model switch if specified.
    3. Retrieves context history from SQLite database.
    4. Executes inference via generate_chat_response using the strictly active model.
    5. Saves turns to SQLite database and returns response with metadata.
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Backend services are still initializing.")
    
    raw_prompt = req.message.strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    conv_id = req.conversation_id
    # Ensure conversation exists
    conv = state.db.get_conversation(conv_id)
    if not conv:
        conv_id = state.db.create_conversation(title="New Chat", model=state.model_manager.active_model_name, conversation_id=conv_id)

    # 1. Handle Slash Commands
    cmd_action, cmd_msg = parse_command(raw_prompt)
    if cmd_action is not None:
        return execute_slash_command(cmd_action, raw_prompt, conv_id)

    # 2. Handle Model Switch Request if explicit
    if req.model and req.model.lower() in ["custom", "external"]:
        if req.model.lower() != state.model_manager.active_model_name:
            state.model_manager.set_model(req.model.lower())

    active_model = state.model_manager.active_model_name

    # 3. Retrieve conversation history from DB
    existing_messages = state.db.get_messages(conv_id)
    history = []
    for m in existing_messages:
        # Filter out slash commands and system feedback from neural model prompt context
        if (m.get("route") or "").startswith("slash_command_"):
            continue
        history.append({"role": m["role"], "text": m["content"]})

    # Add current user prompt
    history.append({"role": "user", "text": raw_prompt})
    state.db.add_message(conv_id, "user", raw_prompt, model_used=active_model)

    # 4. Generate Response using existing verified inference engine
    t0 = time.time()
    try:
        reply, engine_name = generate_chat_response(
            model=state.model_manager.custom_engine.model if active_model == "custom" else None,
            tokenizer=state.model_manager.custom_engine.tokenizer if active_model == "custom" else None,
            history=history,
            device=state.device,
            retrieval_engine=state.retrieval_engine,
            web_search_enabled=state.web_search_enabled,
            debug_mode=state.debug_mode,
            model_manager=state.model_manager
        )
    except Exception as e:
        reply = f"I encountered an unexpected issue processing that question: {e}. Please try rephrasing or asking again."
        engine_name = "error_recovery"

    latency_ms = round((time.time() - t0) * 1000, 1)

    # 5. Extract Sources if any were retrieved
    sources_list = []
    if state.web_search_enabled and engine_name == "web_grounded_evidence" and hasattr(state.retrieval_engine, "session_last_sources") and state.retrieval_engine.session_last_sources:
        for s in state.retrieval_engine.session_last_sources:
            title = getattr(s, "title", str(s)).strip()
            url = getattr(s, "url", "").strip()
            snippet = getattr(s, "snippet", "") or ""
            publisher = getattr(s, "publisher", None)
            date = getattr(s, "date", None)
            sources_list.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "publisher": publisher,
                "date": date
            })

    # 6. Save assistant message to DB
    sources_json = json.dumps(sources_list, ensure_ascii=False) if sources_list else None
    msg_id = state.db.add_message(
        conversation_id=conv_id,
        role="assistant",
        content=reply,
        model_used=active_model,
        route=engine_name,
        sources_json=sources_json,
        latency_ms=latency_ms
    )

    # 7. Auto-update Conversation Title for first user turn
    title_updated = None
    if len(existing_messages) == 0:
        clean_title = raw_prompt.replace("\n", " ").strip()
        if len(clean_title) > 36:
            clean_title = clean_title[:33] + "..."
        state.db.rename_conversation(conv_id, clean_title)
        title_updated = clean_title

    return {
        "id": msg_id,
        "conversation_id": conv_id,
        "role": "assistant",
        "content": reply,
        "model_used": active_model,
        "route": engine_name,
        "sources": sources_list,
        "latency_ms": latency_ms,
        "title_updated": title_updated,
        "is_command": False
    }


@app.post("/api/chat/stream")
async def handle_chat_stream(req: ChatMessageRequest):
    """
    Streaming chat endpoint via Server-Sent Events (SSE).
    Generates response and streams progressive tokens with typing experience,
    followed by the final complete payload with sources and metadata.
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Backend services are still initializing.")

    raw_prompt = req.message.strip()
    conv_id = req.conversation_id
    conv = state.db.get_conversation(conv_id)
    if not conv:
        conv_id = state.db.create_conversation(title="New Chat", model=state.model_manager.active_model_name, conversation_id=conv_id)

    # Check for Slash Commands
    cmd_action, cmd_msg = parse_command(raw_prompt)
    if cmd_action is not None:
        cmd_result = execute_slash_command(cmd_action, raw_prompt, conv_id)
        async def stream_command():
            yield f"data: {json.dumps({'type': 'token', 'text': cmd_result['content']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': cmd_result})}\n\n"
        return StreamingResponse(stream_command(), media_type="text/event-stream")

    if req.model and req.model.lower() in ["custom", "external"]:
        if req.model.lower() != state.model_manager.active_model_name:
            state.model_manager.set_model(req.model.lower())

    active_model = state.model_manager.active_model_name

    existing_messages = state.db.get_messages(conv_id)
    history = []
    for m in existing_messages:
        if (m.get("route") or "").startswith("slash_command_"):
            continue
        history.append({"role": m["role"], "text": m["content"]})
    history.append({"role": "user", "text": raw_prompt})
    state.db.add_message(conv_id, "user", raw_prompt, model_used=active_model)

    async def event_generator():
        t0 = time.time()
        # Initial heartbeat event
        yield f"data: {json.dumps({'type': 'start', 'model': active_model})}\n\n"
        
        # Run inference in worker thread to prevent blocking event loop
        loop = asyncio.get_event_loop()
        try:
            reply, engine_name = await loop.run_in_executor(
                None,
                lambda: generate_chat_response(
                    model=state.model_manager.custom_engine.model if active_model == "custom" else None,
                    tokenizer=state.model_manager.custom_engine.tokenizer if active_model == "custom" else None,
                    history=history,
                    device=state.device,
                    retrieval_engine=state.retrieval_engine,
                    web_search_enabled=state.web_search_enabled,
                    debug_mode=state.debug_mode,
                    model_manager=state.model_manager
                )
            )
        except Exception as e:
            reply = f"I encountered an unexpected issue processing that question: {e}. Please try rephrasing or asking again."
            engine_name = "error_recovery"

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Stream out words in realistic typing chunks
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            # Small yield pause for silky smooth UI typing effect
            await asyncio.sleep(0.015)

        # Sources
        sources_list = []
        if state.web_search_enabled and engine_name == "web_grounded_evidence" and hasattr(state.retrieval_engine, "session_last_sources") and state.retrieval_engine.session_last_sources:
            for s in state.retrieval_engine.session_last_sources:
                title = getattr(s, "title", str(s)).strip()
                url = getattr(s, "url", "").strip()
                snippet = getattr(s, "snippet", "") or ""
                sources_list.append({"title": title, "url": url, "snippet": snippet})

        sources_json = json.dumps(sources_list, ensure_ascii=False) if sources_list else None
        msg_id = state.db.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=reply,
            model_used=active_model,
            route=engine_name,
            sources_json=sources_json,
            latency_ms=latency_ms
        )

        title_updated = None
        if len(existing_messages) == 0:
            clean_title = raw_prompt.replace("\n", " ").strip()
            if len(clean_title) > 36:
                clean_title = clean_title[:33] + "..."
            state.db.rename_conversation(conv_id, clean_title)
            title_updated = clean_title

        final_payload = {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": "assistant",
            "content": reply,
            "model_used": active_model,
            "route": engine_name,
            "sources": sources_list,
            "latency_ms": latency_ms,
            "title_updated": title_updated,
            "is_command": False
        }
        yield f"data: {json.dumps({'type': 'done', 'data': final_payload})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/shutdown")
def shutdown_app(background_tasks: BackgroundTasks):
    """Graceful shutdown endpoint."""
    def kill_server():
        time.sleep(1.0)
        os._exit(0)
    
    background_tasks.add_task(kill_server)
    return {"message": "MEYLO backend is shutting down cleanly..."}


# -------------------------------------------------------------
# Static Files & SPA Routing
# -------------------------------------------------------------
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serves built React static assets or falls back to index.html for client-side routing."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    if FRONTEND_DIST.exists():
        target_file = FRONTEND_DIST / full_path
        if target_file.exists() and target_file.is_file():
            return FileResponse(str(target_file))
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
    
    # Friendly fallback if frontend hasn't been compiled yet
    return JSONResponse({
        "message": "MEYLO API Backend is running.",
        "frontend_status": "Building / awaiting production build in frontend/dist",
        "endpoints": {
            "health": "/api/health",
            "status": "/api/status",
            "models": "/api/models",
            "conversations": "/api/conversations",
            "chat": "/api/chat"
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="info")
