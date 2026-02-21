"""
ixchio — Deep Research Engine
Thin FastAPI shell. All the real logic lives in pipeline/, clients/, and core/.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.helpers import sanitize_query
from core.vector_db import PersistentVectorDB
from pipeline.graph import DeepResearchGraph
import auth


# ==================== STATE ====================
research_tasks = {}
task_by_query = {}
tasks_lock = asyncio.Lock()
research_graph = None

MAX_TASK_AGE = 3600  # 1 hour


async def _evict_old_tasks():
    """Background loop — cleans up finished tasks after MAX_TASK_AGE."""
    while True:
        await asyncio.sleep(600)
        now = datetime.utcnow()
        async with tasks_lock:
            dead = []
            for tid, info in research_tasks.items():
                age = (now - datetime.fromisoformat(info["created_at"])).total_seconds()
                if age > MAX_TASK_AGE:
                    dead.append(tid)
            for tid in dead:
                q = research_tasks[tid].get("query")
                if q and task_by_query.get(q) == tid:
                    del task_by_query[q]
                del research_tasks[tid]


# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_graph
    print("🚀 booting ixchio...")

    # pick vector DB based on env
    if os.getenv("PINECONE_API_KEY"):
        from core.pinecone_db import PineconeDB
        vdb = PineconeDB()
        print("📌 using Pinecone")
    else:
        vdb = PersistentVectorDB()
        print("💾 using local ChromaDB")

    research_graph = DeepResearchGraph(vector_db=vdb)
    cleanup = asyncio.create_task(_evict_old_tasks())
    print("✅ ready")
    yield
    print("👋 shutting down")
    cleanup.cancel()


# ==================== APP ====================
app = FastAPI(
    title="ixchio Deep Research Engine",
    description="Multi-agent deep research with STORM perspectives, reflection loops, and adaptive search routing.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ixchio.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== AUTH ROUTES ====================
@app.post("/auth/signup", tags=["Auth"])
async def signup(req: auth.SignupRequest):
    return auth.signup(req)


@app.post("/auth/login", tags=["Auth"])
async def login(req: auth.LoginRequest):
    return auth.login(req)


@app.get("/auth/me", tags=["Auth"])
async def me(email: str = Depends(auth.get_current_user)):
    info = auth.get_user_info(email)
    if not info:
        raise HTTPException(404, "User not found")
    return info


# ==================== RESEARCH ROUTES ====================
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    depth: str = Field(default="medium")
    max_sources: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="standard")


@app.post("/api/v1/research", tags=["Research"])
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(auth.get_current_user),  # auth required
):
    """Start a research task. Deduplicates identical queries."""
    safe_q = sanitize_query(request.query)

    async with tasks_lock:
        if safe_q in task_by_query:
            existing = task_by_query[safe_q]
            status = research_tasks.get(existing, {}).get("status")
            if status in ("pending", "running", "completed"):
                return {"task_id": existing, "status": "deduplicated"}

        task_id = str(uuid.uuid4())
        research_tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "query": safe_q,
            "user": user,
            "created_at": datetime.utcnow().isoformat(),
        }
        task_by_query[safe_q] = task_id

    request.query = safe_q
    background_tasks.add_task(_run_research, task_id, request)
    return {"task_id": task_id, "status": "pending"}


async def _run_research(task_id: str, request: ResearchRequest):
    """Background task that actually runs the deep research graph."""
    try:
        async with tasks_lock:
            research_tasks[task_id]["status"] = "running"

        result = await research_graph.run(
            query=request.query,
            depth=request.depth,
            max_sources=request.max_sources,
            task_id=task_id,
        )

        async with tasks_lock:
            research_tasks[task_id].update({
                "status": "completed",
                "report": result.get("report", ""),
                "progress": 100,
                "stats": {
                    "api_calls": result.get("total_api_calls", 0),
                    "cache_hits": result.get("cache_hits", 0),
                    "errors": result.get("errors", []),
                },
            })
    except Exception as e:
        async with tasks_lock:
            research_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "progress": 0,
            })


@app.get("/api/v1/research/{task_id}", tags=["Research"])
async def get_research(task_id: str):
    task = research_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


# ==================== WEBSOCKET ====================
@app.websocket("/ws/research/{task_id}")
async def ws_research(ws: WebSocket, task_id: str):
    await ws.accept()
    try:
        while True:
            task = research_tasks.get(task_id)
            if not task:
                await ws.send_json({"error": "not found"})
                break

            await ws.send_json({
                "status": task["status"],
                "progress": task.get("progress", 0),
                "current_step": task.get("current_step", ""),
                "report": task.get("report", ""),
                "error": task.get("error", ""),
            })

            if task["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


# ==================== HEALTH ====================
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "active_tasks": len(research_tasks),
        "cache_stats": research_graph.cache.get_stats() if research_graph else {},
    }


# ==================== RUN ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
