"""
CodeRL API Server — FastAPI server exposing the CodeReviewEnv as HTTP endpoints.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CodeReviewEnv

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("coderl.server")

# ──────────────────────────────────────────────
# App State
# ──────────────────────────────────────────────

env: Optional[CodeReviewEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    data_dir = os.environ.get("CODERL_DATA_DIR", None)
    env = CodeReviewEnv(data_dir=data_dir)
    logger.info("🚀 CodeRL environment loaded and ready")
    yield
    logger.info("👋 CodeRL server shutting down")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="CodeRL — Agentic Code Review RL Environment",
    description="OpenEnv-compliant RL environment for code review",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    comments: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    assert env is not None
    return env.get_summary()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    assert env is not None
    observation = env.reset(task_id=request.task_id)
    return {"success": True, "observation": observation}


@app.post("/step")
async def step(request: StepRequest):
    assert env is not None
    action = {
        "comments": request.comments,
        "tool_calls": request.tool_calls,
    }
    result = env.step(action)
    return {"success": True, **result}


@app.get("/state")
async def state():
    assert env is not None
    return {"state": env.get_state()}


@app.get("/tasks")
async def tasks():
    assert env is not None
    return {"tasks": env.get_task_ids()}


# ──────────────────────────────────────────────
# REQUIRED FOR VALIDATOR ✅
# ──────────────────────────────────────────────

def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)