from models import EpisodeResult


def grade_cost(result: EpisodeResult, baseline_cost: float) -> float:
    """
    Scores cost efficiency vs a naive baseline.
    Returns 0.0 (terrible) to 1.0 (optimal).
    """
    if baseline_cost <= 0:
        return 0.0
    saving = (baseline_cost - result.total_cost) / baseline_cost
    return round(max(0.0, min(1.0, 0.5 + saving)), 4)