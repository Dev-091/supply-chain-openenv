from dataclasses import dataclass, field


@dataclass
class DisruptionEvent:
    start_day: int
    end_day: int
    type: str           # "supplier_offline" | "demand_spike" | "shipment_delay"
    affected_id: str    # supplier_id or "all"
    magnitude: float    # multiplier or duration extension


class DisruptionSchedule:
    """
    Holds a list of disruption events and answers queries
    about what is currently active on a given day.
    """

    def __init__(self, events: list[dict]):
        self.events: list[DisruptionEvent] = [
            DisruptionEvent(**e) for e in events
        ]

    def get_active(self, day: int) -> list[DisruptionEvent]:
        return [
            e for e in self.events
            if e.start_day <= day <= e.end_day
        ]

    def get_warning(self, day: int) -> str | None:
        """
        Returns an early warning if a disruption starts within 3 days.
        """
        upcoming = [
            e for e in self.events
            if 0 < (e.start_day - day) <= 3
        ]
        if not upcoming:
            return None
        e = upcoming[0]
        days_away = e.start_day - day
        return (
            f"{e.type} affecting {e.affected_id} "
            f"starts in {days_away} day(s)"
        )

    def is_supplier_offline(self, supplier_id: str, day: int) -> bool:
        return any(
            e.type == "supplier_offline"
            and (e.affected_id == supplier_id or e.affected_id == "all")
            for e in self.get_active(day)
        )

    def demand_multiplier(self, day: int) -> float:
        multiplier = 1.0
        for e in self.get_active(day):
            if e.type == "demand_spike":
                multiplier *= e.magnitude
        return multiplier

    def shipment_delay_days(self, day: int) -> int:
        delay = 0
        for e in self.get_active(day):
            if e.type == "shipment_delay":
                delay += int(e.magnitude)
        return delay
