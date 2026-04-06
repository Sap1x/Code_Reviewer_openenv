"""
CodeRL State Models — Pydantic models for the OpenEnv-compliant RL environment.

Defines typed schemas for Observation, Action, Reward, State, and StepResult
used throughout the environment lifecycle.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Severity(str, Enum):
    """Issue severity levels, ordered by impact."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Difficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ──────────────────────────────────────────────
# Ground Truth Models
# ──────────────────────────────────────────────

class GroundTruthIssue(BaseModel):
    """A known issue in a task, used as ground truth for grading."""
    line: int = Field(..., description="Line number where the issue occurs")
    issue: str = Field(..., description="Short issue identifier (e.g., 'SQL Injection')")
    severity: Severity = Field(..., description="Issue severity level")
    explanation: str = Field(..., description="Detailed explanation of why this is an issue")
    suggestion: str = Field(..., description="Recommended fix")


# ──────────────────────────────────────────────
# Task Model
# ──────────────────────────────────────────────

class Task(BaseModel):
    """A code review task with code diff and ground truth issues."""
    id: str = Field(..., description="Unique task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty level")
    file_name: str = Field(..., description="File being reviewed")
    code_diff: str = Field(..., description="The code diff to review")
    context: str = Field(..., description="Context about the code's purpose")
    language: str = Field(default="python", description="Programming language")
    ground_truth: list[GroundTruthIssue] = Field(..., description="Known issues for grading")
    # Optional: cross-file context for hard tasks
    related_files: Optional[dict[str, str]] = Field(
        default=None,
        description="Related file contents for cross-file reasoning"
    )
    # Optional: function/variable metadata for tool simulation
    function_signatures: Optional[dict[str, str]] = Field(
        default=None,
        description="Function name → signature mapping for inspect_function tool"
    )
    variable_traces: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="Variable name → usage locations for trace_variable tool"
    )


# ──────────────────────────────────────────────
# Observation (returned to agent)
# ──────────────────────────────────────────────

class HistoryEntry(BaseModel):
    """A single step in the agent's interaction history."""
    step: int
    action_summary: str = Field(..., description="Brief summary of what the agent did")
    reward: float
    issues_found: int


class Observation(BaseModel):
    """What the agent sees at each step."""
    code_diff: str = Field(..., description="The code diff to review")
    file_name: str = Field(..., description="Name of the file being reviewed")
    context: str = Field(..., description="Purpose/context of the code")
    language: str = Field(default="python", description="Programming language")
    step: int = Field(..., description="Current step number (1-indexed)")
    max_steps: int = Field(..., description="Maximum allowed steps")
    history: list[HistoryEntry] = Field(default_factory=list, description="Previous interactions")
    available_tools: list[str] = Field(
        default_factory=lambda: ["inspect_function", "trace_variable"],
        description="Tools the agent can invoke"
    )
    related_files: Optional[dict[str, str]] = Field(
        default=None,
        description="Related file contents (if available)"
    )
    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty")


# ──────────────────────────────────────────────
# Action (agent's response)
# ──────────────────────────────────────────────

class ReviewComment(BaseModel):
    """A single code review comment from the agent."""
    line: int = Field(..., description="Line number of the issue")
    issue: str = Field(..., description="Short issue name")
    severity: Severity = Field(..., description="Issue severity")
    explanation: str = Field(..., description="Why this is a problem")
    suggestion: str = Field(..., description="How to fix it")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence [0-1]")


class ToolCall(BaseModel):
    """A tool invocation by the agent."""
    tool: str = Field(..., description="Tool name: inspect_function | trace_variable")
    argument: str = Field(..., description="Argument to pass to the tool")


class Action(BaseModel):
    """Agent's action: review comments and/or tool calls."""
    comments: list[ReviewComment] = Field(
        default_factory=list,
        description="Code review comments"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations for deeper analysis"
    )


# ──────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed breakdown of reward calculation."""
    precision_score: float = Field(..., description="Fraction of agent's issues that were correct")
    recall_score: float = Field(..., description="Fraction of ground truth issues found")
    severity_bonus: float = Field(..., description="Bonus for finding critical/high severity issues")
    false_positive_penalty: float = Field(..., description="Penalty for incorrect issues")
    duplicate_penalty: float = Field(..., description="Penalty for duplicate findings")
    total: float = Field(..., description="Final composite reward")


# ──────────────────────────────────────────────
# State (internal)
# ──────────────────────────────────────────────

class State(BaseModel):
    """Full internal state of the environment."""
    task_id: str = Field(..., description="Current task identifier")
    difficulty: Difficulty = Field(..., description="Task difficulty")
    current_step: int = Field(default=0, description="Current step (0 = not started)")
    max_steps: int = Field(default=6, description="Maximum steps allowed")
    issues_detected: list[ReviewComment] = Field(
        default_factory=list,
        description="All issues found so far"
    )
    history: list[HistoryEntry] = Field(
        default_factory=list,
        description="Interaction history"
    )
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
    done: bool = Field(default=False, description="Whether episode is finished")
    final_score: Optional[float] = Field(default=None, description="Final grader score [0-1]")
    tool_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Cached results from tool calls"
    )


# ──────────────────────────────────────────────
# Step Result (returned from env.step())
# ──────────────────────────────────────────────

class StepResult(BaseModel):
    """Result of a single environment step."""
    observation: Observation
    reward: RewardBreakdown
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
