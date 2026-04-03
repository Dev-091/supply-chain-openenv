from models import EpisodeResult
from server.graders.cost_grader import grade_cost
from server.graders.service_grader import grade_service

TASK_WEIGHTS = {
    "task_easy": {
        "service": 0.7,
        "cost": 0.3,
    },
    "task_medium": {
        "service": 0.5,
        "cost": 0.5,
    },
    "task_hard": {
        "service": 0.6,
        "cost": 0.4,
    },
}

NAIVE_BASELINE_COSTS = {
    "task_easy":  3000.0,
    "task_medium": 9000.0,
    "task_hard":  18000.0,
}


def grade(result: EpisodeResult) -> dict:
    weights = TASK_WEIGHTS.get(result.task_id, {"service": 0.6, "cost": 0.4})
    baseline = NAIVE_BASELINE_COSTS.get(result.task_id, 10000.0)

    service_score = grade_service(result)
    cost_score = grade_cost(result, baseline)

    final = (
        weights["service"] * service_score +
        weights["cost"] * cost_score
    )

    return {
        "task_id": result.task_id,
        "final_score": round(final, 4),
        "service_score": service_score,
        "cost_score": cost_score,
        "service_level": result.service_level,
        "total_cost": result.total_cost,
        "num_stockouts": result.num_stockouts,
        "days_run": result.days_run,
    }