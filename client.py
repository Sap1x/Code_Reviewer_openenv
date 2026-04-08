"""
CodeRL Client — OpenEnv-compliant HTTP client for the CodeReview environment.

Supports multi-mode deployment:
  - Live URL:    CodeRLEnv(base_url="https://...")
  - Docker:      await CodeRLEnv.from_docker_image("coderl:latest")
  - HF Space:    CodeRLEnv(base_url="https://Sap1x-coderl-env.hf.space")

Usage:
    from client import CodeRLEnv, CodeRLAction, CodeRLObservation

    async with CodeRLEnv(base_url="https://Sap1x-coderl-env.hf.space") as env:
        obs = await env.reset()
        result = await env.step(CodeRLAction(comments=[], tool_calls=[]))
"""

from __future__ import annotations

import httpx
from typing import Any, Optional


class CodeRLAction:
    """Action for the CodeReview environment."""

    def __init__(
        self,
        comments: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.comments = comments or []
        self.tool_calls = tool_calls or []

    def to_dict(self) -> dict[str, Any]:
        return {"comments": self.comments, "tool_calls": self.tool_calls}


class CodeRLObservation:
    """Observation from the CodeReview environment."""

    def __init__(self, data: dict[str, Any]):
        self.code_diff: str = data.get("code_diff", "")
        self.file_name: str = data.get("file_name", "")
        self.context: str = data.get("context", "")
        self.language: str = data.get("language", "python")
        self.step: int = data.get("step", 0)
        self.max_steps: int = data.get("max_steps", 6)
        self.history: list = data.get("history", [])
        self.available_tools: list = data.get("available_tools", [])
        self.task_id: str = data.get("task_id", "")
        self.difficulty: str = data.get("difficulty", "easy")
        self._raw = data

    def __repr__(self) -> str:
        return (
            f"CodeRLObservation(task_id={self.task_id!r}, "
            f"step={self.step}/{self.max_steps}, "
            f"difficulty={self.difficulty!r})"
        )


class CodeRLEnv:
    """
    HTTP client for the CodeRL OpenEnv environment.

    Supports sync and async usage. Compatible with OpenEnv multi-mode deployment.
    """

    def __init__(self, base_url: str = "https://Sap1x-coderl-env.hf.space"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ── Async context manager ──────────────────────────────────────────────

    async def __aenter__(self) -> "CodeRLEnv":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ── Core OpenEnv API ───────────────────────────────────────────────────

    async def reset(self, task_id: Optional[str] = None) -> CodeRLObservation:
        """Reset the environment and return the initial observation."""
        assert self._client is not None, "Use 'async with CodeRLEnv(...) as env:'"
        payload: dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        resp = await self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return CodeRLObservation(data.get("observation", data))

    async def step(self, action: CodeRLAction) -> dict[str, Any]:
        """Submit a review action and advance the environment."""
        assert self._client is not None, "Use 'async with CodeRLEnv(...) as env:'"
        resp = await self._client.post("/step", json=action.to_dict())
        resp.raise_for_status()
        return resp.json()

    async def state(self) -> dict[str, Any]:
        """Return the current internal state of the environment."""
        assert self._client is not None, "Use 'async with CodeRLEnv(...) as env:'"
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    async def tasks(self) -> list[str]:
        """List all available task IDs."""
        assert self._client is not None, "Use 'async with CodeRLEnv(...) as env:'"
        resp = await self._client.get("/tasks")
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    # ── Sync helper ────────────────────────────────────────────────────────

    def sync(self) -> "_SyncCodeRLEnv":
        """Return a synchronous wrapper for non-async usage."""
        return _SyncCodeRLEnv(self.base_url)


class _SyncCodeRLEnv:
    """Synchronous wrapper around CodeRLEnv for non-async usage."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def __enter__(self) -> "_SyncCodeRLEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    def reset(self, task_id: Optional[str] = None) -> CodeRLObservation:
        payload: dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return CodeRLObservation(data.get("observation", data))

    def step(self, action: CodeRLAction) -> dict[str, Any]:
        resp = self._client.post("/step", json=action.to_dict())
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()
