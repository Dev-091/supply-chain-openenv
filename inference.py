import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from env.environment import SupplyChainEnv
from env.models import Action
from graders.composite_grader import grade

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:14b-instruct-q4_K_M")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy-key")
BENCHMARK = os.getenv("BENCHMARK_ENV", "supply-chain-env")
SUCCESS_SCORE_THRESHOLD = 0.5  # Adjust based on task difficulty

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
    "supplier_id": "A" | "B" | "C" | null,
    "quantity": <integer> | null
    }

    Rules:
    - "wait" means do nothing this day
    - "reorder" places a normal order (arrives after lead_time_days)
    - "expedite" arrives next day but costs 2x
    - Only reorder if supplier is not offline (check disruption_warning)
    - quantity must be between supplier min_order_qty and max_order_qty
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=False,
        )
        raw_response = completion.choices[0].message.content.strip()
        data = json.loads(raw_response)
        return Action(**data), raw_response, None
    except Exception as e:
        err_msg = str(e)
        # We must return a default valid action to prevent environment crashes on LLM failure
        return Action(type="wait"), raw_response, err_msg


def run_task(client: OpenAI, task_id: str, seed: int = 42) -> dict:
    env = SupplyChainEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    
    done = False
    step_num = 0
    rewards: List[float] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not done:
            step_num += 1
            action, action_str, error = get_action(client, obs.model_dump())
            
            # Format action string to compress JSON for logging compliance (no newlines)
            safe_action_str = action_str.replace('\\n', '').replace(' ', '')
            if not safe_action_str:
                 safe_action_str = "wait"

            obs, reward, done, info = env.step(action)
            rewards.append(reward.step_reward)

            log_step(step=step_num, action=safe_action_str, reward=reward.step_reward, done=done, error=error)

        result = env.get_final_score()
        scores = grade(result)
        final_score = scores.get('final_score', 0.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        
        log_end(success=success, steps=step_num, score=final_score, rewards=rewards)
        return scores
        
    except Exception as exc:
        print(f"[DEBUG] env step error: {exc}", flush=True)
        # Even on exception, we must emit END log
        log_end(success=False, steps=step_num, score=0.0, rewards=rewards)
        return {}

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        run_task(client, task_id, seed=42)

if __name__ == "__main__":
    main()
