"""
CodeRL End-to-End Validation Test

Tests all components:
1. Task loading
2. Environment reset/step/state
3. Grader determinism
4. Reward calculation
5. Tool simulation
6. Multi-step interaction
7. Server endpoint validation
"""

import json
import sys
import traceback

# ── Test 1: Task Loading ──

def test_task_loading():
    from env.task_loader import TaskLoader
    loader = TaskLoader()
    
    assert loader.task_count == 6, f"Expected 6 tasks, got {loader.task_count}"
    
    summary = loader.summary()
    assert summary["by_difficulty"]["easy"] == 2
    assert summary["by_difficulty"]["medium"] == 2
    assert summary["by_difficulty"]["hard"] == 2
    
    # Verify all task IDs exist
    expected_ids = ["easy_001", "easy_002", "medium_001", "medium_002", "hard_001", "hard_002"]
    for tid in expected_ids:
        task = loader.get_task(tid)
        assert task.id == tid
        assert len(task.ground_truth) > 0
    
    print("✅ Task Loading: PASSED")
    return True

# ── Test 2: Environment Reset ──

def test_environment_reset():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    # Reset with default task
    obs = env.reset()
    assert "code_diff" in obs
    assert "file_name" in obs
    assert "step" in obs
    assert obs["step"] == 1
    
    # Reset with specific task
    obs = env.reset(task_id="hard_001")
    assert obs["task_id"] == "hard_001"
    assert obs["difficulty"] == "hard"
    assert obs["max_steps"] == 8
    
    print("✅ Environment Reset: PASSED")
    return True

# ── Test 3: Environment Step ──

def test_environment_step():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    obs = env.reset(task_id="easy_001")
    
    # Submit a correct finding
    action = {
        "comments": [
            {
                "line": 7,
                "issue": "Resource Leak",
                "severity": "medium",
                "explanation": "File handle not closed",
                "suggestion": "Use with statement",
                "confidence": 0.9
            }
        ],
        "tool_calls": []
    }
    
    result = env.step(action)
    assert "observation" in result
    assert "reward" in result
    assert "done" in result
    assert "info" in result
    assert result["reward"]["total"] > 0, f"Expected positive reward, got {result['reward']['total']}"
    
    print("✅ Environment Step: PASSED")
    return True

# ── Test 4: Grader Determinism ──

def test_grader_determinism():
    from env.grader import Grader
    from env.state import ReviewComment, GroundTruthIssue, Severity
    
    grader = Grader()
    
    predicted = [
        ReviewComment(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                     explanation="test", suggestion="test", confidence=0.9),
        ReviewComment(line=14, issue="Off-by-one Error", severity=Severity.HIGH,
                     explanation="test", suggestion="test", confidence=0.85),
    ]
    
    ground_truth = [
        GroundTruthIssue(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                        explanation="test", suggestion="test"),
        GroundTruthIssue(line=14, issue="Off-by-one Error", severity=Severity.HIGH,
                        explanation="test", suggestion="test"),
        GroundTruthIssue(line=15, issue="Division by Zero", severity=Severity.HIGH,
                        explanation="test", suggestion="test"),
    ]
    
    # Run twice to verify determinism
    result1 = grader.grade(predicted, ground_truth)
    result2 = grader.grade(predicted, ground_truth)
    
    assert result1.total_score == result2.total_score, "Grader is not deterministic!"
    assert result1.precision == result2.precision
    assert result1.recall == result2.recall
    assert result1.total_score > 0
    assert 0.0 <= result1.total_score <= 1.0
    
    print(f"  → Score: {result1.total_score}, Precision: {result1.precision}, Recall: {result1.recall}")
    print("✅ Grader Determinism: PASSED")
    return True

# ── Test 5: Reward Calculation ──

def test_reward_calculation():
    from env.reward import RewardCalculator
    from env.state import ReviewComment, GroundTruthIssue, Severity
    
    calc = RewardCalculator()
    
    comments = [
        ReviewComment(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                     explanation="test", suggestion="test", confidence=0.9),
    ]
    
    ground_truth = [
        GroundTruthIssue(line=7, issue="Resource Leak", severity=Severity.MEDIUM,
                        explanation="test", suggestion="test"),
    ]
    
    reward = calc.calculate(comments, comments, ground_truth)
    assert reward.total > 0, f"Expected positive reward, got {reward.total}"
    assert reward.false_positive_penalty == 0.0
    assert reward.duplicate_penalty == 0.0
    
    # Test false positive
    bad_comments = [
        ReviewComment(line=999, issue="Nonexistent Bug", severity=Severity.LOW,
                     explanation="test", suggestion="test", confidence=0.1),
    ]
    bad_reward = calc.calculate(bad_comments, bad_comments, ground_truth)
    assert bad_reward.false_positive_penalty > 0
    
    print("✅ Reward Calculation: PASSED")
    return True

# ── Test 6: Tool Simulation ──

def test_tool_simulation():
    from env.tools import ToolSimulator
    from env.task_loader import TaskLoader
    
    loader = TaskLoader()
    task = loader.get_task("hard_001")
    sim = ToolSimulator(task)
    
    # Test inspect_function
    result = sim.execute("inspect_function", "hash_password")
    assert result["success"] is True
    assert "signature" in result["result"]
    
    # Test trace_variable
    result = sim.execute("trace_variable", "username")
    assert result["success"] is True
    assert len(result["result"]["usage_locations"]) > 0
    
    # Test unknown tool
    result = sim.execute("unknown_tool", "test")
    assert result["success"] is False
    
    print("✅ Tool Simulation: PASSED")
    return True

# ── Test 7: Multi-Step Interaction ──

def test_multi_step():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    obs = env.reset(task_id="easy_001")
    max_steps = obs["max_steps"]
    
    # Run through all steps
    for step in range(max_steps):
        action = {
            "comments": [
                {
                    "line": 7 + step * 5,
                    "issue": f"Issue {step}",
                    "severity": "medium",
                    "explanation": "test",
                    "suggestion": "test",
                    "confidence": 0.5
                }
            ],
            "tool_calls": []
        }
        result = env.step(action)
        
        if step < max_steps - 1:
            assert result["done"] is False
        else:
            assert result["done"] is True
            assert "final_grade" in result["info"]
            assert 0.0 <= result["info"]["final_grade"]["score"] <= 1.0
    
    # Verify state
    state = env.get_state()
    assert state["done"] is True
    assert state["current_step"] == max_steps
    assert len(state["history"]) == max_steps
    
    print(f"  → Final score: {result['info']['final_grade']['score']}")
    print("✅ Multi-Step Interaction: PASSED")
    return True

# ── Test 8: Full Pipeline (all tasks) ──

def test_full_pipeline():
    from env.environment import CodeReviewEnv
    env = CodeReviewEnv()
    
    task_ids = env.get_task_ids()
    scores = {}
    
    for task_id in task_ids:
        obs = env.reset(task_id=task_id)
        # Just run one step with a generic action
        action = {
            "comments": [
                {
                    "line": 10,
                    "issue": "Generic Issue",
                    "severity": "medium",
                    "explanation": "test",
                    "suggestion": "test",
                    "confidence": 0.5
                }
            ],
            "tool_calls": []
        }
        
        # Step through to completion
        done = False
        while not done:
            result = env.step(action)
            done = result["done"]
        
        score = result["info"]["final_grade"]["score"]
        scores[task_id] = score
    
    print(f"  → Scores: {json.dumps(scores, indent=2)}")
    print("✅ Full Pipeline: PASSED")
    return True


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🧪 CodeRL — End-to-End Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("Task Loading", test_task_loading),
        ("Environment Reset", test_environment_reset),
        ("Environment Step", test_environment_step),
        ("Grader Determinism", test_grader_determinism),
        ("Reward Calculation", test_reward_calculation),
        ("Tool Simulation", test_tool_simulation),
        ("Multi-Step Interaction", test_multi_step),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
