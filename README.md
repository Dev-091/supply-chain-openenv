
---
title: Supply Chain OpenEnv
emoji: 🚛
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Supply Chain OpenEnv

**Supply Chain OpenEnv** is a Reinforcement Learning (RL) simulation environment tailored for advanced supply chain and inventory decision-making tasks. 

It is designed to evaluate both traditional Reinforcement Learning algorithms and modern Large Language Model (LLM) agents. The agent acts as an inventory manager and must balance conflicting objectives: minimizing holding costs, preventing costly stockouts, and dynamically navigating supply chain disruptions (like sudden demand spikes or supplier factory shutdowns).

---

## 🌍 Environment Description

The environment operates on a discrete daily step sequence. Each "day," the agent receives an observation detailing the current state of the warehouse. The episode runs for a fixed number of days (e.g., 30 days) defined by the task difficulty (`task_easy`, `task_medium`, `task_hard`).

The agent's primary goals are:
1. **Prevent Stockouts:** Ensure there is always enough inventory to fulfill the stochastic daily customer demand. Stockouts incur harsh penalty costs.
2. **Minimize Holding Costs:** Keeping too much inventory incurs a daily holding fee.
3. **Navigate Disruptions:** React to early warnings provided by the `DisruptionSchedule` and pivot to secondary suppliers if primary suppliers go offline or experience shipping delays.

At the end of an episode, the agent receives a composite score (0.0 to 1.0) mathematically grading its service level against its total operating costs.

---

## 🔭 Observation Space

At each step, the environment returns a comprehensive JSON observation object describing the current state.

| Field | Type | Description |
| :--- | :--- | :--- |
| `day` | `int` | The current simulated day (e.g., 0 to 30). |
| `stock_level` | `int` | Current physical inventory available in the warehouse. |
| `pending_orders` | `List[PendingOrder]`| Array of incoming shipments, including quantities and expected arrival days. |
| `daily_demand_forecast`| `float` | A noisy estimate of tomorrow's demand. (True demand remains hidden). |
| `suppliers` | `List[SupplierInfo]`| Details on available suppliers, their unit costs, lead times, and min/max order constraints. |
| `disruption_warning` | `str` \| `null` | An early warning string if a disruption (e.g., supplier shutdown) is starting within 3 days. |
| `holding_cost_rate` | `float` | The cost incurred per unit of inventory held per day. |
| `stockout_cost_per_unit`| `float` | The penalty incurred per unit of unfulfilled customer demand. |
| `episode_budget_used` | `float` | Total money spent so far on purchasing inventory. |
| `task_id` | `str` | The active configuration task (e.g., `"task_medium"`). |
| `done` | `bool` | True if the episode has reached the maximum step limit. |

---

## 🕹️ Action Space

At each step, the agent must submit a JSON action object.

*Note: If the agent decides to buy inventory, the `quantity` must fall between the chosen supplier's `min_order_qty` and `max_order_qty`.*

| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `str` | The literal action. Must be `"reorder"`, `"wait"`, `"expedite"`, or `"cancel_order"`. |
| `supplier_id` | `str` \| `null` | Target supplier ID (e.g., `"A"`, `"B"`, `"C"`). Required for `reorder` and `expedite`. |
| `quantity` | `int` \| `null` | Units to purchase. Required for `reorder` and `expedite`. |
| `order_id` | `str` \| `null` | The specific order ID string to cancel. Required only for `cancel_order`. |

### Action Types Breakdown:
- **`wait`**: Do nothing today. Saves budget.
- **`reorder`**: Place a standard order. It will arrive after the supplier's specific `lead_time_days` plus any active disruption delays.
- **`expedite`**: Place an emergency order. Arrives the **next day** but costs 2x the standard unit price.
- **`cancel_order`**: Removes a pending shipment from the queue (simulating stopping a truck mid-transit).

---

## 📋 Task Descriptions

The environment comes with three standardized tasks, graded from 0.0 to 1.0.

| Task ID | Description | Expected Difficulty |
| :--- | :--- | :--- |
| **`task_easy`** | **Basic Reorder Policy** (30 steps). Single reliable supplier. Tests if the agent can maintain a basic 1-to-1 reorder cycle without overstocking. | Easy |
| **`task_medium`** | **Supplier Selection** (60 steps). Multiple suppliers with varying lead times and costs. Tests if the agent optimizes budget by using the slow/cheap supplier normally, but the fast/expensive supplier in emergencies. | Medium |
| **`task_hard`** | **Disruption Mitigation** (90 steps). Introduces chaotic events (dock strikes delaying shipments, sudden viral demand spikes). Tests the agent's ability to pivot rapidly and read text warnings. | Hard |

---

## 📈 Baseline Scores

The default baseline agent (`inference.py`) evaluates zero-shot performance using the local `qwen2.5:14b-instruct-q4_K_M` model (Seed: 42).

| Task | Final Score (Average) |
| :--- | :--- |
| `task_easy` | 0.8120 |
| `task_medium` | 0.6550 |
| `task_hard` | 0.4200 |
| **Average** | **0.6290** |

*Note: Baseline scores demonstrate that the environment differentiates difficulty well. Hard tasks pose significant challenges to current LLMs.*

---

## 🛠️ Setup Instructions

This project strictly follows the **OpenEnv multi-mode deployment** specification, utilizing `uv` for modern dependency locking and deployment standardization.

### 1. Requirements
Ensure you have **Python 3.11+** installed and the `uv` package manager available.

### 2. Local Installation
Clone the repository and sync the dependencies locally:

```bash
# Install uv globally if you don't have it
pip install uv

# Generate the uv.lock file
uv lock

# Sync the environment
uv sync
```

### 3. Running the API Server
The environment is exposed as an HTTP REST API. Start the server on port 7860:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Once running, you can explore the interactive API Documentation by navigating to:  
👉 `http://localhost:7860/docs`

### 4. Running the Baseline Agent
To execute the automated evaluation with the robotic `[START]` and `[STEP]` logging structures:

1. Ensure your chosen AI endpoint is running (or set `HF_TOKEN` / `API_BASE_URL` in your `.env`).
2. Run the inference script:

```bash
python inference.py
```

### 5. Docker Deployment (Optional)
To containerize the OpenEnv application for Hugging Face Spaces natively:

```bash
# Build the image
docker build -t supply-chain-openenv .

# Run the container exposing port 7860
docker run -p 7860:7860 supply-chain-openenv
```
