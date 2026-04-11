
---
title: Supply Chain OpenEnv
emoji: 🚛
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Supply Chain OpenEnv (Winning Edition) 🚛

**Supply Chain OpenEnv** is a high-fidelity Reinforcement Learning (RL) simulation environment designed for evaluating complex supply chain and inventory management strategies.

> [!NOTE]
> This environment is built to be strictly compliant with the **Meta PyTorch OpenEnv** specification, supporting concurrent evaluation sessions and structured robotic logging.

---

## 🌍 Environment Context

The agent manages a regional distribution center. Its goal is to fulfill stochastic customer demand while minimizing operating costs (holding fees, expedited shipping premiums, and stockout penalties).

### Key Management Challenges:
1. **Demand Stochasticity**: Forecasts are noisy; true demand is hidden.
2. **Supplier Reliability**: Suppliers have varying lead times and reliability scores (orders can occasionally be lost in transit).
3. **Disruption Events**: Dock strikes, demand spikes, and factory shutdowns appear as textual warnings in the observation stream.

---

## 🔭 Observation Space

At each step, the agent receives a state JSON. All fields are designed for easy ingestion by LLM-based agents.

| Field | Type | Description | Range / Example |
| :--- | :--- | :--- | :--- |
| `day` | `int` | Current episode day | `0 - 90` |
| `stock_level` | `int` | Physical units in warehouse | `0+` |
| `pending_orders` | `list` | Incoming shipments queue | `[{"order_id": "...", ...}]` |
| `daily_demand_forecast` | `float` | Predicted demand for tomorrow | `~10.0 - 150.0` |
| `suppliers` | `list` | Static supplier performance data | `Lead times, Min/Max qty` |
| `disruption_warning` | `str?` | Textual warning of incoming events | `"Supplier A offline in 2 days"` |
| `holding_cost_rate` | `float` | Cost per unit per day | `0.1 - 0.5` |
| `stockout_cost_per_unit` | `float` | Penalty for unfulfilled demand | `2.0+` |
| `episode_budget_used` | `float` | Total reorder spend so far | `0.0+` |
| `task_id` | `str` | Active task identifier | `"task_easy"` |
| `done` | `bool` | Episode completion flag | `true / false` |

---

## 🕹️ Action Space

The agent must submit a structured JSON action. Validating against the supplier's constraints is mandatory.

| Field | Type | Required For | Description |
| :--- | :--- | :--- | :--- |
| `type` | `str` | All | `wait`, `reorder`, `expedite`, `cancel_order` |
| `supplier_id` | `str` | `reorder`, `expedite` | Target supplier ID (e.g., `"S1"`) |
| `quantity` | `int` | `reorder`, `expedite` | Units to purchase |
| `order_id` | `str` | `cancel_order` | The specific ID to remove from queue |

**Action Dynamics:**
- **Wait**: Cost-free pass of time.
- **Reorder**: Standard lead-time and standard unit cost.
- **Expedite**: Next-day arrival at **2x unit cost**.
- **Cancel**: Useful for mitigating overstock when demand unexpectedly drops.

---

## 📋 Task Specification & Thresholds

We provide three standardized benchmarks. Each is graded from `0.0` to `1.0`.

| Task ID | Focus | Steps | Pass | Excellent |
| :--- | :--- | :--- | :--- | :--- |
| **`task_easy`** | Inventory Balancing | 30 | `> 0.70` | `> 0.90` |
| **`task_medium`** | Supplier Optimization | 60 | `> 0.55` | `> 0.75` |
| **`task_hard`** | Crisis Management | 90 | `> 0.40` | `> 0.60` |

---

## 🏗️ Technical Architecture

This environment is engineered for robustness:
- **Session Isolation**: Supports `session_id` to allow multiple concurrent evaluators.
- **Grader Logic**: Uses `EpisodeResult` models to produce a composite score based on Profitability (40%) and Service Level (60%).
- **Robotic Logging**: Implements the `[START]`, `[STEP]`, `[END]` protocol for automated parsers.

---

## 🛠️ Setup & Deployment

### 1. Requirements
Ensure you have **Python 3.11+** and the `uv` manager.

### 2. Standard Installation
```bash
uv sync
```

### 3. API Serving (Port 7860)
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Running Benchmarks
```bash
# Set HF_TOKEN in your environment first
python inference.py
```

---

## 🐋 Docker Hub / HF Spaces
This repo is ready for native Hugging Face Spaces deployment using the provided `Dockerfile`.

```bash
docker build -t supply-chain-openenv .
docker run -p 7860:7860 supply-chain-openenv
```
