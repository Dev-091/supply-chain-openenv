from models import EpisodeResult

EPSILON = 0.001


def grade_cost(result: EpisodeResult, baseline_cost: float) -> float:
    """
    Scores cost efficiency vs a naive baseline.
    Returns 0.0 (terrible) to 1.0 (optimal).
    """
    if baseline_cost <= 0:
        return EPSILON
    saving = (baseline_cost - result.total_cost) / baseline_cost
    bounded = max(EPSILON, min(1.0 - EPSILON, 0.5 + saving))
    return round(bounded, 6)