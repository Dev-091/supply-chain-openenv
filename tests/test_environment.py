import pytest
from env.environment import SupplyChainEnv
from env.models import Action


def test_reset_returns_observation():
    env = SupplyChainEnv()
    obs = env.reset("task_easy", seed=42)
    assert obs.day == 0
    assert obs.stock_level == 100
    assert len(obs.suppliers) == 1


def test_step_wait_decreases_stock():
    env = SupplyChainEnv()
    env.reset("task_easy", seed=42)
    obs, reward, done, info = env.step(Action(type="wait"))
    assert obs.day == 1
    assert obs.stock_level < 100


def test_step_reorder_creates_pending_order():
    env = SupplyChainEnv()
    env.reset("task_easy", seed=42)
    env.step(Action(type="reorder", supplier_id="A", quantity=100))
    assert len(env._pending) >= 1


def test_episode_completes_in_max_steps():
    env = SupplyChainEnv()
    env.reset("task_easy", seed=42)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(Action(type="wait"))
        steps += 1
    assert steps == 30


def test_final_score_between_0_and_1():
    env = SupplyChainEnv()
    env.reset("task_easy", seed=42)
    done = False
    while not done:
        _, _, done, _ = env.step(Action(
            type="reorder", supplier_id="A", quantity=50
        ))
    result = env.get_final_score()
    assert 0.0 <= result.final_score <= 1.0


def test_deterministic_with_same_seed():
    env1 = SupplyChainEnv()
    env2 = SupplyChainEnv()
    obs1 = env1.reset("task_easy", seed=42)
    obs2 = env2.reset("task_easy", seed=42)
    assert obs1.daily_demand_forecast == obs2.daily_demand_forecast