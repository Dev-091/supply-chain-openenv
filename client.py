from typing import Any, Optional
from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult
from models import Action, Observation

class SupplyChainClient(HTTPEnvClient[Action, Observation]):
    """
    Client for interacting with the SupplyChainOpenEnv via HTTP.
    """
    
    def _step_payload(self, action: Action) -> dict[str, Any]:
        """Convert an Action object into a JSON dictionary payload."""
        # Using model_dump() handles the conversion to dict cleanly, dropping Nones
        return action.model_dump(exclude_none=True)
        
    def _parse_result(self, result_dict: dict[str, Any]) -> StepResult[Observation]:
        """Parse the JSON response back into a StepResult[Observation]."""
        obs = Observation(**result_dict["observation"])
        
        # Determine the reward float
        reward_data = result_dict.get("reward", 0.0)
        if isinstance(reward_data, dict):
            step_reward = float(reward_data.get("step_reward", 0.0))
        else:
            step_reward = float(reward_data)

        return StepResult(
            observation=obs,
            reward=step_reward,
            done=result_dict.get("done", False),
            info=result_dict.get("info", {})
        )

    def _parse_state(self, state_dict: dict[str, Any]) -> Any:
        """Parse the state endpoint response. Defaults to passing the dict through."""
        return state_dict
