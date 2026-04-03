from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from env.environment import SupplyChainEnv
from env.models import Action
from graders.composite_grader import grade

env = SupplyChainEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
    return {"status": "ok"}


@app.post("/reset")
async def reset(
    request: Request,
    task_id: str = Query(default="task_easy"),
    seed: int = Query(default=42),
):
    # Try to gracefully handle JSON bodies for validation bot compatibility
    try:
        body = await request.json()
        if body:
            task_id = body.get("task_id", task_id)
            seed = body.get("seed", seed)
    except:
        pass
        
    obs = env.reset(task_id=task_id, seed=seed)
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
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
def state():
    return env.state()


@app.get("/score")
def score():
    result = env.get_final_score()
    return grade(result)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
