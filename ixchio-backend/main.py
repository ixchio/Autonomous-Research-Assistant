"""
ixchio — Deep Research Engine
Thin FastAPI shell. All the real logic lives in pipeline/, clients/, and core/.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.helpers import sanitize_query
from core.vector_db import PersistentVectorDB
from core.db import get_db, ensure_indexes, ping as mongo_ping
from pipeline.graph import DeepResearchGraph
import auth


# ---- in-memory fallbacks (used when mongo isn't available) ----
_mem_tasks = {}
_mem_task_by_query = {}
_tasks_lock = asyncio.Lock()

research_graph = None

MAX_TASK_AGE = 3600


async def _evict_old_tasks():
    """Cleanup loop. Only matters for in-memory mode — mongo has TTL indexes."""
    while True:
        await asyncio.sleep(600)
        db = get_db()
        if db is not None:
            # mongo handles TTL cleanup on its own, skip
            continue

        now = datetime.utcnow()
        async with _tasks_lock:
            dead = [
                tid for tid, info in _mem_tasks.items()
                if (now - datetime.fromisoformat(info["created_at"])).total_seconds() > MAX_TASK_AGE
            ]
            for tid in dead:
                q = _mem_tasks[tid].get("query")
                if q and _mem_task_by_query.get(q) == tid:
                    del _mem_task_by_query[q]
                del _mem_tasks[tid]


# ---- lifespan ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_graph
    print("🚀 booting ixchio...")

    # mongo indexes (safe to call repeatedly)
    await ensure_indexes()

    mongo_ok = await mongo_ping()
    if mongo_ok:
        print("🍃 mongo connected")
    else:
        print("⚠️  mongo unavailable — running in-memory mode")

    # vector db
    if os.getenv("PINECONE_API_KEY"):
        from core.pinecone_db import PineconeDB
        vdb = PineconeDB()
        print("📌 pinecone loaded")
    else:
        vdb = PersistentVectorDB()
        print("💾 local chromadb")

    research_graph = DeepResearchGraph(vector_db=vdb)
    cleanup = asyncio.create_task(_evict_old_tasks())
    print("✅ ready")
    yield
    cleanup.cancel()


# ---- app ----

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


# ---- task helpers (mongo vs memory) ----

async def _save_task(task: dict):
    db = get_db()
    if db is not None:
        await db.research_tasks.update_one(
            {"task_id": task["task_id"]},
            {"$set": task},
            upsert=True,
        )
    else:
        async with _tasks_lock:
            _mem_tasks[task["task_id"]] = task


async def _get_task(task_id: str) -> dict | None:
    db = get_db()
    if db is not None:
        doc = await db.research_tasks.find_one({"task_id": task_id}, {"_id": 0})
        return doc
    else:
        return _mem_tasks.get(task_id)


async def _find_task_by_query(query: str) -> dict | None:
    db = get_db()
    if db is not None:
        doc = await db.research_tasks.find_one(
            {"query": query, "status": {"$in": ["pending", "running", "completed"]}},
            {"_id": 0},
        )
        return doc
    else:
        tid = _mem_task_by_query.get(query)
        if tid and tid in _mem_tasks:
            t = _mem_tasks[tid]
            if t.get("status") in ("pending", "running", "completed"):
                return t
        return None


# ---- auth routes ----

@app.post("/auth/signup", tags=["Auth"])
async def do_signup(req: auth.SignupRequest):
    return await auth.signup(req)


@app.post("/auth/login", tags=["Auth"])
async def do_login(req: auth.LoginRequest):
    return await auth.login(req)


@app.get("/auth/me", tags=["Auth"])
async def me(email: str = Depends(auth.get_current_user)):
    info = await auth.get_user_info(email)
    if not info:
        raise HTTPException(404, "User gone?")
    return info


# ---- research routes ----

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    depth: str = Field(default="medium")
    max_sources: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="standard")


@app.post("/api/v1/research", tags=["Research"])
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(auth.get_current_user),
):
    safe_q = sanitize_query(request.query)

    # dedup — don't re-run identical queries
    existing = await _find_task_by_query(safe_q)
    if existing:
        return {"task_id": existing["task_id"], "status": "deduplicated"}

    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "status": "pending",
        "query": safe_q,
        "user": user,
        "created_at": datetime.utcnow().isoformat(),
        "progress": 0,
    }
    await _save_task(task)

    # also track in memory for the query dedup shortcut
    _mem_task_by_query[safe_q] = task_id

    request.query = safe_q
    background_tasks.add_task(_run_research, task_id, request)
    return {"task_id": task_id, "status": "pending"}


async def _run_research(task_id: str, request: ResearchRequest):
    try:
        task = await _get_task(task_id) or {}
        task["status"] = "running"
        await _save_task(task)

        result = await research_graph.run(
            query=request.query,
            depth=request.depth,
            max_sources=request.max_sources,
            task_id=task_id,
        )

        task.update({
            "status": "completed",
            "report": result.get("report", ""),
            "progress": 100,
            "stats": {
                "api_calls": result.get("total_api_calls", 0),
                "cache_hits": result.get("cache_hits", 0),
                "errors": result.get("errors", []),
            },
        })
        await _save_task(task)

    except Exception as e:
        task = await _get_task(task_id) or {"task_id": task_id}
        task.update({"status": "failed", "error": str(e), "progress": 0})
        await _save_task(task)


@app.get("/api/v1/research/{task_id}", tags=["Research"])
async def get_research(task_id: str):
    task = await _get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


# ---- websocket (live progress) ----

@app.websocket("/ws/research/{task_id}")
async def ws_research(ws: WebSocket, task_id: str):
    await ws.accept()
    try:
        while True:
            task = await _get_task(task_id)
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


# ---- health ----

@app.get("/health", tags=["System"])
async def health():
    mongo_ok = await mongo_ping()
    return {
        "status": "healthy",
        "mongo": "connected" if mongo_ok else "unavailable (in-memory mode)",
        "active_tasks": len(_mem_tasks),
        "cache_stats": research_graph.cache.get_stats() if research_graph else {},
    }


# ---- history (new! users can see past research) ----

@app.get("/api/v1/history", tags=["Research"])
async def get_history(user: str = Depends(auth.get_current_user)):
    db = get_db()
    if db is None:
        # in-memory mode — just return what we have
        user_tasks = [t for t in _mem_tasks.values() if t.get("user") == user]
        return {"tasks": sorted(user_tasks, key=lambda x: x.get("created_at", ""), reverse=True)[:20]}

    cursor = db.research_tasks.find(
        {"user": user},
        {"_id": 0, "task_id": 1, "query": 1, "status": 1, "created_at": 1, "progress": 1},
    ).sort("created_at", -1).limit(20)

    tasks = await cursor.to_list(length=20)
    return {"tasks": tasks}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
