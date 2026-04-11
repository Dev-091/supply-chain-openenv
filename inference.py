import os
import json
import textwrap
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from server.environment import SupplyChainEnv
from models import Action
from server.graders.composite_grader import grade

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "supply-chain-env"
SUCCESS_SCORE_THRESHOLD = 0.5  # Consistent with README pass_threshold

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert supply chain manager. Each day you receive the current
    inventory state as JSON and must decide what action to take.

    Your objectives in order of priority:
    1. Never run out of stock (stockouts cost 2x per unit)
    2. Keep ordering costs low (holding costs apply daily)
    3. Use cheaper suppliers for bulk orders, fast suppliers only when urgent

    You must respond with ONLY a valid JSON object — no explanation, no markdown.
    Use exactly this schema:
    {
      "type": "reorder" | "wait" | "expedite",
      "supplier_id": "S1" | "S2" | "S3" | null,
      "quantity": <integer> | null
    }

    Rules:
    - "wait" means do nothing this day
    - "reorder" places a normal order (arrives after lead_time_days)
    - "expedite" arrives in 1-2 days but costs 2x
    - Only reorder if supplier is not offline (check disruption_warning)
    - quantity must be between supplier min_order_qty and max_order_qty
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    # [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # [END] success=<true|false> steps=<n> score=<... > rewards=<r1,r2,...,rn>
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

def get_action(client: OpenAI, observation_dict: dict) -> tuple[Action, str, Optional[str]]:
    prompt = (
        "Current supply chain state:\n"
        + json.dumps(observation_dict, indent=2)
        + "\n\nDecide your action."
    )
    raw_response = ""
    err_msg = None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=False,
        )
        raw_response = completion.choices[0].message.content.strip()
        data = json.loads(raw_response)
        # Basic validation to ensure fields are present
        if "type" not in data:
            data["type"] = "wait"
        return Action(**data), raw_response, None
    except Exception as e:
        err_msg = str(e).replace('\n', ' ')
        return Action(type="wait"), json.dumps({"type": "wait"}), err_msg


def run_task(client: OpenAI, task_id: str, seed: int = 42):
    # Unique session for this task run to ensure concurrency safety
    session_id = f"eval_{uuid.uuid4().hex[:8]}"
    
    env = SupplyChainEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    # Internally session_id is used for server calls if we used a remote client
    # For this local inference script, we just demonstrate the session_id pattern
    
    done = False
    step_num = 0
    rewards: List[float] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    last_score = 0.0001
    success = False
    
    try:
        while not done and step_num < 100: # Safety cap
            step_num += 1
            action, action_json, error = get_action(client, obs.model_dump())
            
            # Action string for logging (compact)
            action_str = action.model_dump_json(exclude_none=True)
            
            obs, reward_obj, done, info = env.step(action)
            r = reward_obj.step_reward
            rewards.append(r)

            log_step(step=step_num, action=action_str, reward=r, done=done, error=error)

        result = env.get_final_score()
        scores = grade(result)
        last_score = scores.get('final_score', 0.0001)
        success = last_score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as exc:
        print(f"[DEBUG] Error running task {task_id}: {exc}", flush=True)
    finally:
        # Final log must be emitted even on error
        log_end(success=success, steps=step_num, score=last_score, rewards=rewards)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["task_easy", "task_medium", "task_hard"]
    for task_id in tasks:
        run_task(client, task_id, seed=42)

if __name__ == "__main__":
    main()
