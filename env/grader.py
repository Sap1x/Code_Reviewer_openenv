"""
CodeRL Grader — Deterministic grading of agent review comments vs ground truth.

Compares predicted issues against ground truth using fuzzy matching
(line proximity + issue type similarity). Returns score in [0.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from env.state import GroundTruthIssue, ReviewComment, Severity


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

LINE_TOLERANCE = 3  # ±3 lines for matching
ISSUE_SIMILARITY_THRESHOLD = 0.4  # Minimum string similarity for issue names

SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.LOW: 0.5,
    Severity.MEDIUM: 1.0,
    Severity.HIGH: 1.5,
    Severity.CRITICAL: 2.0,
}


# ──────────────────────────────────────────────
# Match Result
# ──────────────────────────────────────────────

@dataclass
class IssueMatch:
    """Result of matching a predicted issue to a ground truth issue."""
    predicted: ReviewComment
    ground_truth: GroundTruthIssue
    line_match: bool
    issue_similarity: float
    severity_match: bool
    match_quality: str  # "exact", "partial", "none"

    @property
    def is_match(self) -> bool:
        return self.match_quality in ("exact", "partial")


@dataclass
class GradeResult:
    """Full grading result for a set of predicted issues."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    severity_weighted_score: float = 0.0
    total_score: float = 0.0
    matches: list[IssueMatch] = field(default_factory=list)
    false_positives: list[ReviewComment] = field(default_factory=list)
    missed_issues: list[GroundTruthIssue] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Grader
# ──────────────────────────────────────────────

class Grader:
    """
    Deterministic grader for code review issues.

    Compares predicted issues against ground truth using:
    - Line proximity (±3 lines)
    - Issue type string similarity
    - Severity matching
    """

    def __init__(
        self,
        line_tolerance: int = LINE_TOLERANCE,
        similarity_threshold: float = ISSUE_SIMILARITY_THRESHOLD,
    ):
        self.line_tolerance = line_tolerance
        self.similarity_threshold = similarity_threshold

    def grade(
        self,
        predicted: list[ReviewComment],
        ground_truth: list[GroundTruthIssue],
    ) -> GradeResult:
        """
        Grade predicted issues against ground truth.

        Returns a GradeResult with score in [0.0, 1.0].
        """
        if not ground_truth:
            # No issues to find — penalize false positives
            score = 1.0 if not predicted else max(0.0, 1.0 - len(predicted) * 0.2)
            return GradeResult(
                precision=1.0 if not predicted else 0.0,
                recall=1.0,
                f1=1.0 if not predicted else 0.0,
                total_score=score,
                false_positives=list(predicted),
            )

        if not predicted:
            return GradeResult(
                precision=0.0,
                recall=0.0,
                f1=0.0,
                total_score=0.0,
                missed_issues=list(ground_truth),
            )

        # Match predicted → ground truth (greedy best-first)
        matches, false_positives, unmatched_gt = self._match_issues(
            predicted, ground_truth
        )

        # Calculate metrics
        exact_matches = [m for m in matches if m.match_quality == "exact"]
        partial_matches = [m for m in matches if m.match_quality == "partial"]

        # Precision: fraction of predictions that were correct
        correct_count = len(exact_matches) + 0.5 * len(partial_matches)
        precision = correct_count / len(predicted) if predicted else 0.0

        # Recall: fraction of ground truth issues found
        found_count = len(exact_matches) + 0.5 * len(partial_matches)
        recall = found_count / len(ground_truth) if ground_truth else 0.0

        # F1
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Severity-weighted score
        total_weight = sum(
            SEVERITY_WEIGHTS[gt.severity] for gt in ground_truth
        )
        found_weight = 0.0
        for m in matches:
            w = SEVERITY_WEIGHTS[m.ground_truth.severity]
            if m.match_quality == "exact":
                found_weight += w
            elif m.match_quality == "partial":
                found_weight += w * 0.5
        severity_weighted = found_weight / total_weight if total_weight > 0 else 0.0

        # Composite score ∈ [0.0, 1.0]
        total_score = min(1.0, max(0.0,
            f1 * 0.6 + severity_weighted * 0.4
            - len(false_positives) * 0.05
        ))

        return GradeResult(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            severity_weighted_score=round(severity_weighted, 4),
            total_score=round(total_score, 4),
            matches=matches,
            false_positives=false_positives,
            missed_issues=unmatched_gt,
            details={
                "exact_matches": len(exact_matches),
                "partial_matches": len(partial_matches),
                "false_positives": len(false_positives),
                "missed": len(unmatched_gt),
                "total_ground_truth": len(ground_truth),
                "total_predicted": len(predicted),
            },
        )

    def _match_issues(
        self,
        predicted: list[ReviewComment],
        ground_truth: list[GroundTruthIssue],
    ) -> tuple[list[IssueMatch], list[ReviewComment], list[GroundTruthIssue]]:
        """
        Greedy best-first matching of predicted issues to ground truth.

        Each ground truth issue can be matched at most once.
        """
        # Score all possible (predicted, gt) pairs
        candidates: list[tuple[float, int, int, IssueMatch]] = []

        for pi, pred in enumerate(predicted):
            for gi, gt in enumerate(ground_truth):
                match = self._evaluate_match(pred, gt)
                if match.is_match:
                    score = (
                        1.0 if match.match_quality == "exact" else 0.5
                    )
                    candidates.append((score, pi, gi, match))

        # Sort by score descending (greedy best-first)
        candidates.sort(key=lambda x: (-x[0], x[1]))

        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        matches: list[IssueMatch] = []

        for score, pi, gi, match in candidates:
            if pi not in matched_pred and gi not in matched_gt:
                matches.append(match)
                matched_pred.add(pi)
                matched_gt.add(gi)

        false_positives = [
            predicted[i] for i in range(len(predicted)) if i not in matched_pred
        ]
        unmatched_gt = [
            ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt
        ]

        return matches, false_positives, unmatched_gt

    def _evaluate_match(
        self, pred: ReviewComment, gt: GroundTruthIssue
    ) -> IssueMatch:
        """Evaluate how well a predicted issue matches a ground truth issue."""
        # Line proximity check
        line_diff = abs(pred.line - gt.line)
        line_match = line_diff <= self.line_tolerance

        # Issue type similarity (string similarity)
        issue_similarity = SequenceMatcher(
            None,
            pred.issue.lower().strip(),
            gt.issue.lower().strip(),
        ).ratio()

        # Severity match
        severity_match = pred.severity == gt.severity

        # Determine match quality
        if line_match and issue_similarity >= 0.6:
            quality = "exact"
        elif line_match and issue_similarity >= self.similarity_threshold:
            quality = "partial"
        elif issue_similarity >= 0.7:
            # Strong issue name match even if line is off
            quality = "partial"
        else:
            quality = "none"

        return IssueMatch(
            predicted=pred,
            ground_truth=gt,
            line_match=line_match,
            issue_similarity=round(issue_similarity, 4),
            severity_match=severity_match,
            match_quality=quality,
        )
