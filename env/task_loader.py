"""
CodeRL Task Loader — Loads and validates code review tasks from JSON files.

Provides task iteration, selection by difficulty, and random task selection.
All tasks are validated against Pydantic models on load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from env.state import Difficulty, Task


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DIFFICULTY_FILES = {
    Difficulty.EASY: "easy.json",
    Difficulty.MEDIUM: "medium.json",
    Difficulty.HARD: "hard.json",
}


# ──────────────────────────────────────────────
# Task Loader
# ──────────────────────────────────────────────

class TaskLoader:
    """Loads, validates, and provides access to code review tasks."""

    def __init__(self, data_dir: Optional[str | Path] = None):
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._tasks: dict[str, Task] = {}
        self._by_difficulty: dict[Difficulty, list[Task]] = {
            d: [] for d in Difficulty
        }
        self._load_all()

    # ── public API ──────────────────────────────

    def get_task(self, task_id: str) -> Task:
        """Get a specific task by ID."""
        if task_id not in self._tasks:
            available = ", ".join(sorted(self._tasks.keys()))
            raise KeyError(
                f"Task '{task_id}' not found. Available tasks: {available}"
            )
        return self._tasks[task_id]

    def get_tasks_by_difficulty(self, difficulty: Difficulty) -> list[Task]:
        """Get all tasks for a given difficulty level."""
        return list(self._by_difficulty[difficulty])

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks across all difficulties."""
        return list(self._tasks.values())

    def get_all_task_ids(self) -> list[str]:
        """Get all task IDs in deterministic order."""
        return sorted(self._tasks.keys())

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def summary(self) -> dict:
        """Return a summary of loaded tasks."""
        return {
            "total": self.task_count,
            "by_difficulty": {
                d.value: len(tasks) for d, tasks in self._by_difficulty.items()
            },
            "task_ids": self.get_all_task_ids(),
        }

    # ── internal ────────────────────────────────

    def _load_all(self) -> None:
        """Load all task files from the data directory."""
        for difficulty, filename in DIFFICULTY_FILES.items():
            filepath = self._data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Task file not found: {filepath}. "
                    f"Expected data directory: {self._data_dir}"
                )
            self._load_file(filepath, difficulty)

    def _load_file(self, filepath: Path, expected_difficulty: Difficulty) -> None:
        """Load and validate tasks from a single JSON file."""
        with open(filepath, "r") as f:
            raw_tasks = json.load(f)

        if not isinstance(raw_tasks, list):
            raise ValueError(f"Expected a JSON array in {filepath}, got {type(raw_tasks).__name__}")

        for idx, raw in enumerate(raw_tasks):
            try:
                task = Task(**raw)
            except Exception as e:
                raise ValueError(
                    f"Invalid task at index {idx} in {filepath}: {e}"
                ) from e

            # Validate difficulty matches the file
            if task.difficulty != expected_difficulty:
                raise ValueError(
                    f"Task '{task.id}' in {filepath.name} has difficulty "
                    f"'{task.difficulty.value}' but expected '{expected_difficulty.value}'"
                )

            # Ensure unique IDs
            if task.id in self._tasks:
                raise ValueError(f"Duplicate task ID: '{task.id}'")

            self._tasks[task.id] = task
            self._by_difficulty[expected_difficulty].append(task)

    def __repr__(self) -> str:
        return f"TaskLoader(tasks={self.task_count}, dir='{self._data_dir}')"
