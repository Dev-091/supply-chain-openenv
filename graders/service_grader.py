from env.models import EpisodeResult


def grade_service(result: EpisodeResult) -> float:
    """
    Scores fill rate (service level).
    0.0 = never fulfilled anything
    1.0 = fulfilled every unit of demand
    """
    return round(max(0.0, min(1.0, result.service_level)), 4)