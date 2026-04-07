from __future__ import annotations
import json
import uuid
from pathlib import Path
from copy import deepcopy

import numpy as np

from models import (
    Action, Observation, PendingOrder,
    Reward, RewardBreakdown, SupplierInfo, EpisodeResult
)
from server.demand_generator import DemandGenerator
from server.disruption import DisruptionSchedule

TASKS_DIR = Path(__file__).parent / "tasks"

FULFILLMENT_REWARD = 1.0
STOCKOUT_PENALTY   = 2.0
INACTION_PENALTY   = 0.3


class SupplyChainEnv:

    def __init__(self):
        self._config: dict = {}
        self._day: int = 0
        self._stock: int = 0
        self._pending: list[PendingOrder] = []
        self._cumulative_reward: float = 0.0
        self._budget_used: float = 0.0
        self._done: bool = False
        self._demand_gen: DemandGenerator | None = None
        self._disruptions: DisruptionSchedule | None = None
        self._suppliers: list[SupplierInfo] = []
        self._task_id: str = ""
        self._seed: int = 42

        # episode tracking
        self._total_demanded: int = 0
        self._total_fulfilled: int = 0
        self._total_cost: float = 0.0
        self._num_stockouts: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_easy", seed: int = 42) -> Observation:
        cfg_path = TASKS_DIR / f"{task_id}.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Task fixture not found: {cfg_path}")

        self._config = json.loads(cfg_path.read_text())
        self._task_id = task_id
        self._seed = seed
        self._day = 0
        self._stock = self._config["initial_stock"]
        self._pending = []
        self._cumulative_reward = 0.0
        self._budget_used = 0.0
        self._done = False
        self._total_demanded = 0
        self._total_fulfilled = 0
        self._total_cost = 0.0
        self._num_stockouts = 0

        self._suppliers = [
            SupplierInfo(**s) for s in self._config["suppliers"]
        ]
        self._demand_gen = DemandGenerator(
            base_demand=self._config["base_demand"],
            seasonality=self._config["seasonality"],
            seed=seed,
        )
        self._disruptions = DisruptionSchedule(
            self._config.get("disruptions", [])
        )

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._day += 1

        # 1. Deliver any pending orders that have arrived
        self._deliver_orders()

        # 2. Generate true demand (affected by disruptions)
        true_demand = self._demand_gen.get_demand(self._day)
        multiplier = self._disruptions.demand_multiplier(self._day)
        true_demand = int(true_demand * multiplier)
        self._total_demanded += true_demand

        # 3. Fulfil demand from stock
        units_sold = min(self._stock, true_demand)
        unmet = true_demand - units_sold
        self._stock -= units_sold
        self._total_fulfilled += units_sold
        if unmet > 0:
            self._num_stockouts += 1

        # 4. Apply action
        order_cost = 0.0
        order_placed = False

        if action.type == "reorder":
            order_cost = self._place_order(action)
            order_placed = True
        elif action.type == "expedite":
            order_cost = self._expedite_order(action)
            order_placed = True
        elif action.type == "cancel_order":
            self._cancel_order(action)

        self._budget_used += order_cost
        self._total_cost += order_cost

        # 5. Compute reward
        holding_cost = self._stock * self._config["holding_cost_rate"]
        self._total_cost += holding_cost

        disruption_active = bool(self._disruptions.get_active(self._day))
        inaction_pen = INACTION_PENALTY if (
            disruption_active and not order_placed and self._stock < 50
        ) else 0.0

        breakdown = RewardBreakdown(
            fulfillment=float(units_sold * FULFILLMENT_REWARD),
            stockout_penalty=float(-unmet * STOCKOUT_PENALTY),
            holding_cost=float(-holding_cost),
            order_cost=float(-order_cost * 0.01),
            inaction_penalty=float(-inaction_pen),
        )
        step_reward = sum([
            breakdown.fulfillment,
            breakdown.stockout_penalty,
            breakdown.holding_cost,
            breakdown.order_cost,
            breakdown.inaction_penalty,
        ])
        self._cumulative_reward += step_reward

        # 6. Check done
        if self._day >= self._config["max_steps"]:
            self._done = True

        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            breakdown=breakdown,
        )
        obs = self._build_observation()
        info = {"day": self._day, "unmet_demand": unmet, "order_cost": order_cost}

        return obs, reward, self._done, info

    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "day": self._day,
            "stock_level": self._stock,
            "pending_orders": [p.model_dump() for p in self._pending],
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
        }

    def get_final_score(self) -> EpisodeResult:
        service_level = (
            self._total_fulfilled / self._total_demanded
            if self._total_demanded > 0 else 0.0
        )
        # normalize cumulative reward to 0-1, strictly bounded between (0, 1)
        max_possible = self._total_demanded * FULFILLMENT_REWARD
        raw_score = self._cumulative_reward / max(1, max_possible)
        final_score = max(0.001, min(0.999, raw_score))

        return EpisodeResult(
            task_id=self._task_id,
            total_reward=round(self._cumulative_reward, 4),
            final_score=round(final_score, 4),
            days_run=self._day,
            total_units_demanded=self._total_demanded,
            total_units_fulfilled=self._total_fulfilled,
            service_level=round(service_level, 4),
            total_cost=round(self._total_cost, 4),
            num_stockouts=self._num_stockouts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        return Observation(
            day=self._day,
            stock_level=self._stock,
            pending_orders=deepcopy(self._pending),
            daily_demand_forecast=self._demand_gen.get_forecast(self._day + 1),
            suppliers=self._suppliers,
            disruption_warning=self._disruptions.get_warning(self._day),
            holding_cost_rate=self._config["holding_cost_rate"],
            stockout_cost_per_unit=self._config["stockout_cost_per_unit"],
            episode_budget_used=round(self._budget_used, 2),
            task_id=self._task_id,
            done=self._done,
        )

    def _get_supplier(self, supplier_id: str) -> SupplierInfo:
        for s in self._suppliers:
            if s.id == supplier_id:
                return s
        raise ValueError(f"Unknown supplier: {supplier_id}")

    def _place_order(self, action: Action) -> float:
        if not action.supplier_id or not action.quantity:
            return 0.0

        if self._disruptions.is_supplier_offline(action.supplier_id, self._day):
            return 0.0  # silently reject orders to offline suppliers

        supplier = self._get_supplier(action.supplier_id)
        qty = max(supplier.min_order_qty, min(action.quantity, supplier.max_order_qty))

        extra_delay = self._disruptions.shipment_delay_days(self._day)
        arrives = self._day + supplier.lead_time_days + extra_delay

        rng = np.random.default_rng(self._seed + self._day)
        if rng.random() > supplier.reliability:
            # order lost in transit — no stock added, cost still incurred
            pass
        else:
            self._pending.append(PendingOrder(
                order_id=str(uuid.uuid4())[:8],
                supplier_id=action.supplier_id,
                quantity=qty,
                ordered_on_day=self._day,
                arrives_on_day=arrives,
            ))

        return supplier.unit_cost * qty

    def _expedite_order(self, action: Action) -> float:
        if not action.supplier_id or not action.quantity:
            return 0.0
        supplier = self._get_supplier(action.supplier_id)
        qty = max(supplier.min_order_qty, min(action.quantity, supplier.max_order_qty))
        # expedite: arrives next day, costs 2x
        self._pending.append(PendingOrder(
            order_id=str(uuid.uuid4())[:8],
            supplier_id=action.supplier_id,
            quantity=qty,
            ordered_on_day=self._day,
            arrives_on_day=self._day + 1,
        ))
        return supplier.unit_cost * qty * 2.0

    def _cancel_order(self, action: Action):
        if action.order_id:
            self._pending = [
                p for p in self._pending if p.order_id != action.order_id
            ]

    def _deliver_orders(self):
        arrived = [p for p in self._pending if p.arrives_on_day <= self._day]
        for order in arrived:
            self._stock += order.quantity
        self._pending = [p for p in self._pending if p.arrives_on_day > self._day]