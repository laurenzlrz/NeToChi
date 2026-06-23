from pydantic import BaseModel, Field, ConfigDict
from typing import Any
from netochi.definitions.constants import (
    MCMC_DEFAULT_ITERATIONS,
    MCMC_DEFAULT_INITIAL_TEMP,
    MCMC_DEFAULT_SEED,
    MCMC_TIME_LIMIT_S
)

class JointInferenceConfig(BaseModel):
    """Configuration for physical hardware cost prefactors and MCMC parameters."""
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    objective: Any = Field(..., description="The objective function for mapping optimization.")
    alpha: float = Field(default=0.5, description="Silicon area cost for cores (compute & local memory)")
    beta: float = Field(default=0.1, description="Silicon area cost for routing fabric (switches & wires)")
    gamma: float = Field(default=1.0, description="Utilization penalty for wasted empty slots")
    slice_factor: int = Field(default=2, description="Slice factor for hardware config creation.")
    iterations: int = Field(default=MCMC_DEFAULT_ITERATIONS)
    initial_temp: float = Field(default=MCMC_DEFAULT_INITIAL_TEMP)
    seed: int = Field(default=MCMC_DEFAULT_SEED)
    verbose: bool = Field(default=False)
    time_limit_s: float = Field(default=MCMC_TIME_LIMIT_S)

    def create(self) -> "JointInferenceMapper":
        from netochi.mapping.mcmc.joint_inference_mapper import JointInferenceMapper
        return JointInferenceMapper(config=self)