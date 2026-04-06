"""
CodeRL Environment — Production-grade OpenEnv-compliant RL environment
for agentic code review.

Implements the core OpenEnv interface:
    - reset(task_id?) → Observation
    - step(action) → (Observation, Reward, done, info)
    - state() → State
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from env.grader import Grader, GradeResult
from env.reward import RewardCalculator
from env.state import (
    Action,
    Difficulty,
    HistoryEntry,
    Observation,
    RewardBreakdown,
    ReviewComment,
    State,
    StepResult,
    Task,
)
from env.task_loader import TaskLoader
from env.tools import ToolSimulator

logger = logging.getLogger("coderl.environment")


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_MAX_STEPS = 6

MAX_STEPS_BY_DIFFICULTY = {
    Difficulty.EASY: 5,
    Difficulty.MEDIUM: 6,
    Difficulty.HARD: 8,
}


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class CodeReviewEnv:
    """
    OpenEnv-compliant Reinforcement Learning environment for code review.

    The agent reviews code diffs, identifies issues, and receives
    dense rewards based on precision, recall, and severity matching.
    Supports multi-step reasoning with tool calls.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._task_loader = TaskLoader(data_dir=data_dir)
        self._grader = Grader()
        self._reward_calc = RewardCalculator(grader=self._grader)
        self._tool_sim: Optional[ToolSimulator] = None

        # Current episode state
        self._current_task: Optional[Task] = None
        self._state: Optional[State] = None
        self._initialized = False

        logger.info(
            "CodeReviewEnv initialized — %d tasks loaded", self._task_loader.task_count
        )

    # ======================================================================
    # OpenEnv Interface
    # ======================================================================

    def reset(self, task_id: Optional[str] = None) -> dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Specific task to load. If None, loads the first task.

        Returns:
            Initial observation as a dict.
        """
        if task_id:
            task = self._task_loader.get_task(task_id)
        else:
            # Default to first task
            all_ids = self._task_loader.get_all_task_ids()
            task = self._task_loader.get_task(all_ids[0])

        max_steps = MAX_STEPS_BY_DIFFICULTY.get(task.difficulty, DEFAULT_MAX_STEPS)

        self._current_task = task
        self._tool_sim = ToolSimulator(task)
        self._state = State(
            task_id=task.id,
            difficulty=task.difficulty,
            current_step=0,
            max_steps=max_steps,
            issues_detected=[],
            history=[],
            cumulative_reward=0.0,
            done=False,
            final_score=None,
            tool_results={},
        )
        self._initialized = True

        observation = self._build_observation()

        logger.info(
            "Environment reset — task=%s difficulty=%s max_steps=%d",
            task.id, task.difficulty.value, max_steps,
        )

        return observation.model_dump()

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Process an agent action and advance the environment.

        Args:
            action: Dict matching Action schema (comments + optional tool_calls)

        Returns:
            StepResult as a dict with observation, reward, done, info.
        """
        self._ensure_initialized()
        assert self._state is not None
        assert self._current_task is not None

        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Parse action
        parsed_action = Action(**action)

        # Advance step
        self._state.current_step += 1

        # ── Process tool calls ──
        tool_results = []
        if parsed_action.tool_calls and self._tool_sim:
            for tc in parsed_action.tool_calls:
                result = self._tool_sim.execute(tc.tool, tc.argument)
                tool_results.append(result)
                self._state.tool_results[f"{tc.tool}:{tc.argument}"] = result

        # ── Process review comments ──
        new_comments = parsed_action.comments
        self._state.issues_detected.extend(new_comments)

        # ── Calculate reward ──
        reward = self._reward_calc.calculate(
            new_comments=new_comments,
            all_comments=self._state.issues_detected,
            ground_truth=self._current_task.ground_truth,
        )
        self._state.cumulative_reward += reward.total

        # ── Check if done ──
        done = self._state.current_step >= self._state.max_steps

        # ── Add history entry ──
        action_summary = self._summarize_action(parsed_action, tool_results)
        self._state.history.append(
            HistoryEntry(
                step=self._state.current_step,
                action_summary=action_summary,
                reward=reward.total,
                issues_found=len(new_comments),
            )
        )

        # ── Final grading if done ──
        info: dict[str, Any] = {
            "step": self._state.current_step,
            "tool_results": tool_results,
            "new_issues_count": len(new_comments),
            "total_issues_found": len(self._state.issues_detected),
        }

        if done:
            self._state.done = True
            final_grade = self._grader.grade(
                self._state.issues_detected,
                self._current_task.ground_truth,
            )
            self._state.final_score = final_grade.total_score
            info["final_grade"] = {
                "score": final_grade.total_score,
                "precision": final_grade.precision,
                "recall": final_grade.recall,
                "f1": final_grade.f1,
                "severity_weighted": final_grade.severity_weighted_score,
                "details": final_grade.details,
            }
            logger.info(
                "Episode complete — task=%s score=%.4f steps=%d",
                self._state.task_id,
                final_grade.total_score,
                self._state.current_step,
            )

        # ── Build observation ──
        observation = self._build_observation()

        step_result = StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

        return step_result.model_dump()

    def get_state(self) -> dict[str, Any]:
        """Return the current internal state."""
        self._ensure_initialized()
        assert self._state is not None
        return self._state.model_dump()

    # ======================================================================
    # Convenience Methods
    # ======================================================================

    def get_task_ids(self) -> list[str]:
        """Return all available task IDs."""
        return self._task_loader.get_all_task_ids()

    def get_summary(self) -> dict[str, Any]:
        """Return environment summary."""
        return {
            "name": "CodeRL",
            "version": "1.0.0",
            "description": "Agentic Code Review RL Environment",
            "tasks": self._task_loader.summary(),
            "max_steps_by_difficulty": {
                k.value: v for k, v in MAX_STEPS_BY_DIFFICULTY.items()
            },
        }

    # ======================================================================
    # Internal
    # ======================================================================

    def _build_observation(self) -> Observation:
        """Build the current observation for the agent."""
        assert self._current_task is not None
        assert self._state is not None

        return Observation(
            code_diff=self._current_task.code_diff,
            file_name=self._current_task.file_name,
            context=self._current_task.context,
            language=self._current_task.language,
            step=self._state.current_step + 1,  # 1-indexed for agent
            max_steps=self._state.max_steps,
            history=list(self._state.history),
            available_tools=["inspect_function", "trace_variable"],
            related_files=self._current_task.related_files,
            task_id=self._current_task.id,
            difficulty=self._current_task.difficulty,
        )

    def _summarize_action(
        self, action: Action, tool_results: list[dict]
    ) -> str:
        """Create a brief summary of the agent's action."""
        parts = []
        if action.comments:
            issues = [c.issue for c in action.comments]
            parts.append(f"Found {len(action.comments)} issue(s): {', '.join(issues)}")
        if action.tool_calls:
            tools = [f"{tc.tool}({tc.argument})" for tc in action.tool_calls]
            parts.append(f"Used tools: {', '.join(tools)}")
        return "; ".join(parts) if parts else "No action taken"

    def _ensure_initialized(self) -> None:
        """Ensure the environment has been reset."""
        if not self._initialized or self._state is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )
