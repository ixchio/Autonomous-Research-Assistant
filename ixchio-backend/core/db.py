"""
Mongo connection. Uses Motor (async driver) so we don't block the event loop.
Falls back to a dict-based mock if MONGO_URI isn't set — handy for local dev
when you just wanna boot the server without docker-compose.
"""

import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient

_client = None
_db = None


def get_db():
    global _client, _db

    if _db is not None:
        return _db

    uri = os.getenv("MONGO_URI")
    if not uri:
        # no mongo? fine, we'll limp along with in-memory dicts.
        # good enough for `python main.py` on your laptop
        print("⚠️  MONGO_URI not set — using in-memory fallback (data won't persist)")
        return None

    _client = AsyncIOMotorClient(
        uri,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsCAFile=certifi.where(),
    )
    _db = _client.ixchio  # database name
    return _db


async def ensure_indexes():
    """Call once on startup. Idempotent."""
    db = get_db()
    if db is None:
        return

    # unique email, obviously
    await db.users.create_index("email", unique=True)
    # we query tasks by user a lot
    await db.research_tasks.create_index("user")
    await db.research_tasks.create_index("created_at")
    # ttl index — auto-delete old tasks after 7 days
    # mongo handles the cleanup for us, no cron needed
    await db.research_tasks.create_index(
        "created_at", expireAfterSeconds=60 * 60 * 24 * 7
    )


async def ping():
    """Health check. Returns True if mongo is reachable."""
    db = get_db()
    if db is None:
        return False
    try:
        await db.command("ping")
        return True
    except Exception:
        return False
