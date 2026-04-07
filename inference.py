import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from server.environment import SupplyChainEnv
from models import Action
from server.graders.composite_grader import grade

load_dotenv()

# Use os.environ to strictly comply with the proxy requirement (fail fast if missing)
# We support both API_KEY and HF_TOKEN as per validation variants
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

BENCHMARK = "supply-chain-env"
SUCCESS_SCORE_THRESHOLD = 0.3  # Threshold for success=true

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
    error_val = f'"{error}"' if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # [END] success=<true|false> steps=<n> score=<0.0000> rewards=<r1,r2,...,rn>
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
    env = SupplyChainEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    
    done = False
    step_num = 0
    rewards: List[float] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    last_score = 0.001
    try:
        while not done and step_num < 100: # Safety cap
            step_num += 1
            action, action_json, error = get_action(client, obs.model_dump())
            
            # Action string for logging (compact)
            action_str = action_json.replace('\n', '').replace(' ', '')
            
            obs, reward_obj, done, info = env.step(action)
            r = reward_obj.step_reward
            rewards.append(r)

            log_step(step=step_num, action=action_str, reward=r, done=done, error=error)

        result = env.get_final_score()
        scores = grade(result)
        last_score = scores.get('final_score', 0.001)
        success = last_score >= SUCCESS_SCORE_THRESHOLD
        
        log_end(success=success, steps=step_num, score=last_score, rewards=rewards)
        
    except Exception as exc:
        # Final log must be emitted even on error
        log_end(success=False, steps=step_num, score=last_score, rewards=rewards)

def main():
    if not API_BASE_URL:
        raise RuntimeError("Missing required environment variable: API_BASE_URL")
    if not API_KEY:
        raise RuntimeError("Missing required environment variable: API_KEY (or HF_TOKEN)")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["task_easy", "task_medium", "task_hard"]
    for task_id in tasks:
        run_task(client, task_id, seed=42)

if __name__ == "__main__":
    main()
