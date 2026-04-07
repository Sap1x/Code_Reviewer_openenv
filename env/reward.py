"""
CodeRL Reward Calculator — Dense, meaningful reward system for code review.

Computes per-step rewards based on:
  - Precision (0.5 weight)
  - Recall (0.3 weight)
  - Severity bonus (0.2 weight)
  - False positive penalty
  - Duplicate penalty
"""

from __future__ import annotations

from env.grader import Grader, GradeResult, SEVERITY_WEIGHTS
from env.state import (
    GroundTruthIssue,
    RewardBreakdown,
    ReviewComment,
    Severity,
)


# ──────────────────────────────────────────────
# Reward Constants
# ──────────────────────────────────────────────

PRECISION_WEIGHT = 0.5
RECALL_WEIGHT = 0.3
SEVERITY_BONUS_WEIGHT = 0.2

FALSE_POSITIVE_PENALTY = 0.5
DUPLICATE_PENALTY = 0.2

# Bonus for finding critical/high severity issues
SEVERITY_BONUS_MAP: dict[Severity, float] = {
    Severity.CRITICAL: 0.3,
    Severity.HIGH: 0.15,
    Severity.MEDIUM: 0.05,
    Severity.LOW: 0.0,
}


# ──────────────────────────────────────────────
# Reward Calculator
# ──────────────────────────────────────────────

class RewardCalculator:
    """
    Calculates dense, meaningful rewards for code review actions.

    Reward formula:
        reward = precision * 0.5 + recall * 0.3 + severity_bonus * 0.2
                 - false_positive_penalty - duplicate_penalty
    """

    def __init__(self, grader: Grader | None = None):
        self.grader = grader or Grader()

    def calculate(
        self,
        new_comments: list[ReviewComment],
        all_comments: list[ReviewComment],
        ground_truth: list[GroundTruthIssue],
    ) -> RewardBreakdown:
        """
        Calculate reward for a step.

        Args:
            new_comments: Comments submitted in this step only
            all_comments: All comments accumulated so far (including new ones)
            ground_truth: Ground truth issues for the current task

        Returns:
            RewardBreakdown with component scores and total
        """
        if not new_comments:
            return RewardBreakdown(
                precision_score=0.0,
                recall_score=0.0,
                severity_bonus=0.0,
                false_positive_penalty=0.0,
                duplicate_penalty=0.0,
                total=0.0,
            )

        # Grade the cumulative set of comments
        grade = self.grader.grade(all_comments, ground_truth)

        # ── Precision ──
        precision_score = grade.precision * PRECISION_WEIGHT

        # ── Recall ──
        recall_score = grade.recall * RECALL_WEIGHT

        # ── Severity bonus ──
        severity_bonus = self._calculate_severity_bonus(grade)

        # ── False positive penalty ──
        fp_penalty = len(grade.false_positives) * FALSE_POSITIVE_PENALTY

        # ── Duplicate penalty ──
        dup_penalty = self._calculate_duplicate_penalty(new_comments, all_comments)

        # ── Total ──
        total = (
            precision_score
            + recall_score
            + severity_bonus * SEVERITY_BONUS_WEIGHT
            - fp_penalty
            - dup_penalty
        )

        return RewardBreakdown(
            precision_score=round(precision_score, 4),
            recall_score=round(recall_score, 4),
            severity_bonus=round(severity_bonus, 4),
            false_positive_penalty=round(fp_penalty, 4),
            duplicate_penalty=round(dup_penalty, 4),
            total=round(total, 4),
        )

    def _calculate_severity_bonus(self, grade: GradeResult) -> float:
        """Bonus for finding critical/high severity issues."""
        bonus = 0.0
        for match in grade.matches:
            if match.is_match:
                sev = match.ground_truth.severity
                bonus += SEVERITY_BONUS_MAP.get(sev, 0.0)
        return bonus

    def _calculate_duplicate_penalty(
        self,
        new_comments: list[ReviewComment],
        all_comments: list[ReviewComment],
    ) -> float:
        """
        Penalize duplicate findings in the new batch.

        A comment is a duplicate if a previously submitted comment
        covers the same line AND has a similar issue name.
        """
        previous = all_comments[: len(all_comments) - len(new_comments)]
        if not previous:
            return 0.0

        duplicates = 0
        for new in new_comments:
            for old in previous:
                if (
                    abs(new.line - old.line) <= 1
                    and new.issue.lower().strip() == old.issue.lower().strip()
                ):
                    duplicates += 1
                    break

        return duplicates * DUPLICATE_PENALTY
