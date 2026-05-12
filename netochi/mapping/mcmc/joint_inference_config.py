from pydantic import BaseModel, Field

class JointInferenceConfig(BaseModel):
    """Configuration for hardware cost prefactors."""
    lambda_K: float = Field(default=1.0)
    lambda_Nc: float = Field(default=1.0)
    lambda_Nr: float = Field(default=1.0)
    lambda_L: float = Field(default=0.5)
