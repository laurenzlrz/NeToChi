from pydantic import BaseModel, Field

class JointInferenceConfig(BaseModel):
    """Configuration for physical hardware cost prefactors."""
    alpha: float = Field(default=0.5, description="Silicon area cost for cores (compute & local memory)")
    beta: float = Field(default=0.1, description="Silicon area cost for routing fabric (switches & wires)")
    gamma: float = Field(default=1.0, description="Utilization penalty for wasted empty slots")
