"""
CodeRL Inference — Baseline agent using an OpenAI-compatible LLM.

Runs the agent across all tasks in the environment, producing
reproducible scores with strict logging format.

Environment variables:
    API_BASE_URL  — LLM API base URL
    MODEL_NAME    — Model identifier
    HF_TOKEN      — Hugging Face token for authentication
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from openai import OpenAI

from env.environment import CodeReviewEnv

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("coderl.inference")


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = """You are an expert code reviewer. You analyze code diffs to find bugs, security vulnerabilities, and logic errors.

For each issue you find, respond with a JSON array of objects, each with these fields:
- "line": (int) the line number where the issue occurs in the diff
- "issue": (string) short name for the issue (e.g., "SQL Injection", "Off-by-one Error")
- "severity": (string) one of "low", "medium", "high", "critical"
- "explanation": (string) why this is a problem
- "suggestion": (string) how to fix it
- "confidence": (float) your confidence from 0.0 to 1.0

You may also invoke tools to inspect functions or trace variables:
- To inspect a function: include a "tool_calls" array with {"tool": "inspect_function", "argument": "function_name"}
- To trace a variable: include a "tool_calls" array with {"tool": "trace_variable", "argument": "variable_name"}

Respond with valid JSON only. Format:
{
  "comments": [...],
  "tool_calls": [...]
}

Be thorough but precise. Finding real issues earns rewards; false positives are penalized.
On later steps, use your history to go deeper — don't repeat findings, investigate new areas."""

STEP_PROMPTS = {
    1: "This is your first look at the code. Identify the most obvious issues — syntax errors, missing error handling, dangerous patterns. Focus on high-confidence findings.",
    2: "Look deeper now. Check for logical errors, edge cases, and security vulnerabilities. Use tools if available to trace variable usage or inspect function signatures.",
    3: "Final review. Look for subtle issues you may have missed — race conditions, data exposure, cross-function dependency bugs. Be thorough.",
}


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────

class CodeReviewAgent:
    """LLM-based code review agent."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate_action(
        self,
        observation: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        """
        Generate a review action based on the current observation.

        Returns an action dict with 'comments' and 'tool_calls'.
        """
        # Build user prompt
        user_prompt = self._build_prompt(observation, step)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temp for reproducibility
                max_tokens=4096,
            )

            content = response.choices[0].message.content or "{}"
            action = self._parse_response(content)
            return action

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"comments": [], "tool_calls": []}

    def _build_prompt(self, observation: dict, step: int) -> str:
        """Build the user prompt from the observation."""
        parts = [
            f"## Code Review Task: {observation['task_id']}",
            f"**File:** {observation['file_name']}",
            f"**Context:** {observation['context']}",
            f"**Language:** {observation.get('language', 'python')}",
            f"**Difficulty:** {observation['difficulty']}",
            f"**Step:** {observation['step']}/{observation['max_steps']}",
            "",
            "### Code Diff",
            "```",
            observation['code_diff'],
            "```",
        ]

        # Add history if available
        if observation.get('history'):
            parts.append("\n### Previous Steps")
            for entry in observation['history']:
                parts.append(
                    f"- Step {entry['step']}: {entry['action_summary']} "
                    f"(reward: {entry['reward']:.2f}, issues found: {entry['issues_found']})"
                )

        # Add related files if available
        if observation.get('related_files'):
            parts.append("\n### Related Files")
            for fname, content in observation['related_files'].items():
                parts.append(f"\n**{fname}:**\n```\n{content}\n```")

        # Add step-specific guidance
        step_guidance = STEP_PROMPTS.get(step, STEP_PROMPTS[3])
        parts.append(f"\n### Guidance\n{step_guidance}")
        parts.append(f"\n**Available tools:** {observation.get('available_tools', [])}")
        parts.append("\nRespond with valid JSON containing 'comments' and optionally 'tool_calls'.")

        return "\n".join(parts)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse the LLM response into an action dict."""
        # Try to extract JSON from the response
        content = content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (``` markers)
            json_lines = []
            inside = False
            for line in lines:
                if line.strip().startswith("```") and not inside:
                    inside = True
                    continue
                elif line.strip() == "```" and inside:
                    break
                elif inside:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(content[start:end])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON")
                    return {"comments": [], "tool_calls": []}
            else:
                return {"comments": [], "tool_calls": []}

        # Normalize: if data is a list, treat as comments
        if isinstance(data, list):
            data = {"comments": data, "tool_calls": []}

        # Ensure required keys
        if "comments" not in data:
            data["comments"] = []
        if "tool_calls" not in data:
            data["tool_calls"] = []

        return data


# ──────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────

def run_inference(
    api_base_url: str = API_BASE_URL,
    model_name: str = MODEL_NAME,
    hf_token: str = HF_TOKEN,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    """
    Run the agent across all tasks and return aggregated results.

    Produces strict logging format:
        [START] task=... env=... model=...
        [STEP] step=1 action=... reward=... done=false error=null
        [END] success=true steps=... score=... rewards=...
    """
    # Initialize LLM client
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )

    agent = CodeReviewAgent(client=client, model=model_name)
    env = CodeReviewEnv()

    all_task_ids = env.get_task_ids()
    results: dict[str, Any] = {
        "model": model_name,
        "tasks": {},
        "aggregate": {},
    }

    total_score = 0.0
    total_tasks = 0

    for task_id in all_task_ids:
        # ── START ──
        logger.info(f"[START] task={task_id} env=CodeRL model={model_name}")

        observation = env.reset(task_id=task_id)
        done = False
        step = 0
        task_rewards = []
        error = None

        try:
            while not done:
                step += 1

                # Agent generates action
                action = agent.generate_action(observation, step)

                # Environment processes action
                step_result = env.step(action)

                reward = step_result["reward"]["total"]
                done = step_result["done"]
                observation = step_result["observation"]
                task_rewards.append(reward)

                # Compact action summary for logging
                n_comments = len(action.get("comments", []))
                n_tools = len(action.get("tool_calls", []))
                action_summary = f"comments={n_comments},tools={n_tools}"

                # ── STEP ──
                logger.info(
                    f"[STEP] step={step} action={action_summary} "
                    f"reward={reward:.4f} done={str(done).lower()} error=null"
                )

                # Respect max steps override
                if max_steps_override and step >= max_steps_override:
                    break

            # ── END ──
            final_score = step_result.get("info", {}).get("final_grade", {}).get("score", 0.0)
            total_score += final_score
            total_tasks += 1

            logger.info(
                f"[END] success=true steps={step} score={final_score:.4f} "
                f"rewards={[round(r, 4) for r in task_rewards]}"
            )

            results["tasks"][task_id] = {
                "success": True,
                "steps": step,
                "score": final_score,
                "rewards": task_rewards,
                "final_grade": step_result.get("info", {}).get("final_grade"),
            }

        except Exception as e:
            error = str(e)
            logger.error(
                f"[END] success=false steps={step} score=0.0 "
                f"rewards={task_rewards} error={error}"
            )
            results["tasks"][task_id] = {
                "success": False,
                "steps": step,
                "score": 0.0,
                "error": error,
            }

    # ── Aggregate ──
    avg_score = total_score / total_tasks if total_tasks > 0 else 0.0
    results["aggregate"] = {
        "total_tasks": total_tasks,
        "average_score": round(avg_score, 4),
        "total_score": round(total_score, 4),
    }

    logger.info(
        f"\n{'='*60}\n"
        f"FINAL RESULTS: {total_tasks} tasks, avg_score={avg_score:.4f}\n"
        f"{'='*60}"
    )

    return results


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CodeRL Inference Agent")
    parser.add_argument("--api-base-url", default=API_BASE_URL, help="LLM API base URL")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument("--hf-token", default=HF_TOKEN, help="HF token")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per task")
    args = parser.parse_args()

    results = run_inference(
        api_base_url=args.api_base_url,
        model_name=args.model,
        hf_token=args.hf_token,
        max_steps_override=args.max_steps,
    )

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to inference_results.json")
