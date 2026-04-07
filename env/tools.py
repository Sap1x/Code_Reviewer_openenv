"""
CodeRL Tool Simulator — Simulated developer tools for multi-step reasoning.

Provides `inspect_function` and `trace_variable` tools that return
pre-defined or dynamically generated information about the code being reviewed.
"""

from __future__ import annotations

from typing import Any, Optional

from env.state import Task


# ──────────────────────────────────────────────
# Tool Simulator
# ──────────────────────────────────────────────

class ToolSimulator:
    """
    Simulates developer tools that an agent can invoke during code review.

    Supported tools:
        - inspect_function: Returns function signature and docstring
        - trace_variable: Returns variable usage chain
    """

    def __init__(self, task: Task):
        self._task = task
        self._call_log: list[dict[str, Any]] = []

    # ── public API ──────────────────────────────

    def execute(self, tool_name: str, argument: str) -> dict[str, Any]:
        """
        Execute a simulated tool.

        Args:
            tool_name: One of 'inspect_function', 'trace_variable'
            argument: The function/variable name to inspect

        Returns:
            Tool result dict with 'success', 'tool', 'argument', and 'result' keys
        """
        handlers = {
            "inspect_function": self._inspect_function,
            "trace_variable": self._trace_variable,
        }

        handler = handlers.get(tool_name)
        if not handler:
            result = {
                "success": False,
                "tool": tool_name,
                "argument": argument,
                "error": f"Unknown tool: '{tool_name}'. Available: {list(handlers.keys())}",
            }
        else:
            result = handler(argument)

        # Log the call
        self._call_log.append({
            "tool": tool_name,
            "argument": argument,
            "success": result.get("success", False),
        })

        return result

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Return the history of tool calls."""
        return list(self._call_log)

    # ── tool implementations ────────────────────

    def _inspect_function(self, function_name: str) -> dict[str, Any]:
        """Return function signature and metadata."""
        # Check task-level function signatures first
        if self._task.function_signatures and function_name in self._task.function_signatures:
            signature = self._task.function_signatures[function_name]
            return {
                "success": True,
                "tool": "inspect_function",
                "argument": function_name,
                "result": {
                    "name": function_name,
                    "signature": signature,
                    "found_in": self._task.file_name,
                },
            }

        # Try to extract from the code diff
        extracted = self._extract_function_from_diff(function_name)
        if extracted:
            return {
                "success": True,
                "tool": "inspect_function",
                "argument": function_name,
                "result": extracted,
            }

        return {
            "success": False,
            "tool": "inspect_function",
            "argument": function_name,
            "error": f"Function '{function_name}' not found in the current context.",
        }

    def _trace_variable(self, variable_name: str) -> dict[str, Any]:
        """Return variable usage trace."""
        # Check task-level variable traces
        if self._task.variable_traces and variable_name in self._task.variable_traces:
            trace = self._task.variable_traces[variable_name]
            return {
                "success": True,
                "tool": "trace_variable",
                "argument": variable_name,
                "result": {
                    "variable": variable_name,
                    "usage_locations": trace,
                    "file": self._task.file_name,
                },
            }

        # Try to find in code diff
        locations = self._find_variable_in_diff(variable_name)
        if locations:
            return {
                "success": True,
                "tool": "trace_variable",
                "argument": variable_name,
                "result": {
                    "variable": variable_name,
                    "usage_locations": locations,
                    "file": self._task.file_name,
                },
            }

        return {
            "success": False,
            "tool": "trace_variable",
            "argument": variable_name,
            "error": f"Variable '{variable_name}' not found in the current context.",
        }

    # ── helpers ─────────────────────────────────

    def _extract_function_from_diff(self, function_name: str) -> Optional[dict]:
        """Try to extract function info from the code diff text."""
        lines = self._task.code_diff.split("\n")
        for i, line in enumerate(lines):
            stripped = line.lstrip("+").lstrip("-").lstrip()
            if stripped.startswith(f"def {function_name}("):
                return {
                    "name": function_name,
                    "signature": stripped.rstrip(":").strip(),
                    "found_in": self._task.file_name,
                    "approximate_line": i + 1,
                }
        return None

    def _find_variable_in_diff(self, variable_name: str) -> list[str]:
        """Find all lines in the diff that reference a variable."""
        locations = []
        lines = self._task.code_diff.split("\n")
        for i, line in enumerate(lines):
            content = line.lstrip("+").lstrip("-").lstrip()
            if variable_name in content:
                prefix = "added" if line.startswith("+") else "removed" if line.startswith("-") else "context"
                locations.append(f"line ~{i + 1} ({prefix}): {content.strip()}")
        return locations[:10]  # Cap at 10 results
