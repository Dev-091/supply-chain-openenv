from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional


class SupplierInfo(BaseModel):
    id: str
    name: str
    unit_cost: float
    lead_time_days: int
    reliability: float = Field(ge=0.0, le=1.0)
    min_order_qty: int
    max_order_qty: int


class PendingOrder(BaseModel):
    order_id: str
    supplier_id: str
    quantity: int
    ordered_on_day: int
    arrives_on_day: int


class Observation(BaseModel):
    day: int
    stock_level: int
    pending_orders: list[PendingOrder]
    daily_demand_forecast: float
    suppliers: list[SupplierInfo]
    disruption_warning: Optional[str]
    holding_cost_rate: float
    stockout_cost_per_unit: float
    episode_budget_used: float
    task_id: str
    done: bool


class Action(BaseModel):
    type: Literal["reorder", "wait", "expedite", "cancel_order"]
    supplier_id: Optional[str] = None
    quantity: Optional[int] = None
    order_id: Optional[str] = None


class RewardBreakdown(BaseModel):
    fulfillment: float
    stockout_penalty: float
    holding_cost: float
    order_cost: float
    inaction_penalty: float


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    breakdown: RewardBreakdown


class EpisodeResult(BaseModel):
    task_id: str
    total_reward: float
    final_score: float
    days_run: int
    total_units_demanded: int
    total_units_fulfilled: int
    service_level: float
    total_cost: float
    num_stockouts: int