import numpy as np


class DemandGenerator:
    """
    Generates daily demand with optional seasonality and noise.
    Seeded for reproducibility.
    """

    def __init__(self, base_demand: float, seasonality: bool, seed: int):
        self.base_demand = base_demand
        self.seasonality = seasonality
        self.rng = np.random.default_rng(seed)

    def get_demand(self, day: int) -> int:
        demand = self.base_demand

        if self.seasonality:
            # mild sine wave — peaks around day 15 of each 30-day cycle
            cycle_pos = (day % 30) / 30
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * cycle_pos)
            demand *= seasonal_factor

        # add gaussian noise ±20%
        noise = self.rng.normal(loc=1.0, scale=0.2)
        demand *= max(0.1, noise)

        return max(1, int(round(demand)))

    def get_forecast(self, day: int) -> float:
        """
        Returns a noisy forecast — not the true demand.
        Agent sees this, not the real value.
        """
        true = self.base_demand
        if self.seasonality:
            cycle_pos = (day % 30) / 30
            true *= 1.0 + 0.3 * np.sin(2 * np.pi * cycle_pos)
        noise = self.rng.normal(loc=1.0, scale=0.15)
        return round(true * max(0.1, noise), 1)