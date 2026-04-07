from models import EpisodeResult

EPSILON = 0.001


def grade_service(result: EpisodeResult) -> float:
    """
    Scores fill rate (service level).
    0.0 = never fulfilled anything
    1.0 = fulfilled every unit of demand
    """
    bounded = max(EPSILON, min(1.0 - EPSILON, result.service_level))
    return round(bounded, 6)