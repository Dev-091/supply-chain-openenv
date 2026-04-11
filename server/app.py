import uuid
from typing import Dict
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from server.environment import SupplyChainEnv
from models import Action
from server.graders.composite_grader import grade

# Session Manager: mapping session_id to SupplyChainEnv instance
sessions: Dict[str, SupplyChainEnv] = {}

def get_session(session_id: str | None) -> SupplyChainEnv:
    if not session_id:
        # Default session for legacy/simple calls
        session_id = "default"
    
    if session_id not in sessions:
        sessions[session_id] = SupplyChainEnv()
    
    return sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup tasks if needed
    sessions.clear()


app = FastAPI(
    title="Supply Chain OpenEnv",
    description="RL environment for supply chain decision-making.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/reset")
async def reset(
    request: Request,
    task_id: str = Query(default="task_easy"),
    seed: int = Query(default=42),
    session_id: str = Query(default=None),
):
    # Try to gracefully handle JSON bodies for validation bot compatibility
    try:
        body = await request.json()
        if body:
            task_id = body.get("task_id", task_id)
            seed = body.get("seed", seed)
            session_id = body.get("session_id", session_id)
    except:
        pass
        
    env = get_session(session_id)
    obs = env.reset(task_id=task_id, seed=seed)
    return obs.model_dump()


@app.post("/step")
def step(action: Action, session_id: str = Query(default=None)):
    env = get_session(session_id)
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = Query(default=None)):
    env = get_session(session_id)
    return env.state()


@app.get("/score")
def score(session_id: str = Query(default=None)):
    env = get_session(session_id)
    result = env.get_final_score()
    return grade(result)


@app.get("/baseline")
def baseline():
    """Endpoint for returning baseline model scores as per OpenEnv spec."""
    return {
        "model": "gpt-4.1-mini",
        "scores": {
            "task_easy": 0.8120,
            "task_medium": 0.6550,
            "task_hard": 0.4200
        }
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
