"""
CodeRL — Agentic Code Review RL Environment

OpenEnv-compliant environment for code review RL training.

Quick start:
    from client import CodeRLEnv, CodeRLAction

    async with CodeRLEnv(base_url="https://Sap1x-coderl-env.hf.space") as env:
        obs = await env.reset()
        result = await env.step(CodeRLAction(comments=[], tool_calls=[]))
"""

from client import CodeRLEnv, CodeRLAction, CodeRLObservation
from models import Action, Observation, ReviewComment, Severity, Difficulty

__version__ = "1.0.0"

__all__ = [
    "CodeRLEnv",
    "CodeRLAction",
    "CodeRLObservation",
    "Action",
    "Observation",
    "ReviewComment",
    "Severity",
    "Difficulty",
]
