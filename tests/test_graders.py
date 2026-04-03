from models import EpisodeResult
from server.graders.composite_grader import grade
from server.graders.service_grader import grade_service
from server.graders.cost_grader import grade_cost


def _make_result(task_id, fulfilled, demanded, cost, stockouts=0):
    return EpisodeResult(
        task_id=task_id,
        total_reward=0.0,
        final_score=0.0,
        days_run=30,
        total_units_demanded=demanded,
        total_units_fulfilled=fulfilled,
        service_level=round(fulfilled / demanded, 4) if demanded else 0.0,
        total_cost=cost,
        num_stockouts=stockouts,
    )


def test_perfect_service_scores_1():
    r = _make_result("task_easy", 600, 600, 1000.0)
    assert grade_service(r) == 1.0


def test_zero_service_scores_0():
    r = _make_result("task_easy", 0, 600, 0.0)
    assert grade_service(r) == 0.0


def test_cost_better_than_baseline_scores_above_half():
    r = _make_result("task_easy", 600, 600, 1500.0)
    score = grade_cost(r, baseline_cost=3000.0)
    assert score > 0.5


def test_composite_score_in_range():
    r = _make_result("task_easy", 500, 600, 2000.0, stockouts=2)
    result = grade(r)
    assert 0.0 <= result["final_score"] <= 1.0


def test_hard_task_uses_correct_weights():
    r = _make_result("task_hard", 2000, 2500, 10000.0)
    result = grade(r)
    assert result["task_id"] == "task_hard"
    assert "service_score" in result
    assert "cost_score" in result