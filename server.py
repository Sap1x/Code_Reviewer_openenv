"""
CodeRL API Server — FastAPI server exposing the CodeReviewEnv as HTTP endpoints.

Endpoints:
    POST /reset          — Reset environment (optionally with task_id)
    POST /step           — Submit an action
    GET  /state          — Get current state
    GET  /health         — Health check
    GET  /tasks          — List available tasks
    GET  /               — Environment info
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
    """Initialize environment on startup."""
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
    description=(
        "A production-grade OpenEnv-compliant Reinforcement Learning environment "
        "that simulates real-world code review workflows."
    ),
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
# Request/Response Models
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
    """Environment information."""
    assert env is not None
    return env.get_summary()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "CodeRL", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment for a new episode.

    Optionally specify a task_id to load a specific task.
    Returns the initial observation.
    """
    assert env is not None
    try:
        observation = env.reset(task_id=request.task_id)
        return {"success": True, "observation": observation}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """
    Submit an agent action and advance the environment.

    The action should contain review comments and/or tool calls.
    Returns observation, reward, done flag, and info dict.
    """
    assert env is not None
    try:
        action = {
            "comments": request.comments,
            "tool_calls": request.tool_calls,
        }
        result = env.step(action)
        return {"success": True, **result}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Return the current internal state of the environment."""
    assert env is not None
    try:
        current = env.get_state()
        return {"success": True, "state": current}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def tasks():
    """List all available task IDs."""
    assert env is not None
    return {"tasks": env.get_task_ids()}
