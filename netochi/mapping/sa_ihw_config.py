from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from netochi.pipeline.config import PipelineOutput


class SimAnnealingIHWConfig(BaseModel):
    """
    Configuration for Inferred Hardware Simulated Annealing Mapper.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline_output: PipelineOutput = Field(..., description="Pipeline output configuration.")

    # Move Probabilities
    p_swap: float = Field(default=0.55, description="Probability of swapping two neurons.")
    p_move: float = Field(default=0.20, description="Probability of moving a neuron to an empty slot.")
    p_add_core: float = Field(default=0.03, description="Probability of adding a core.")
    p_remove_core: float = Field(default=0.03, description="Probability of removing a core.")
    p_increment_nc: float = Field(default=0.04, description="Probability of incrementing Nc.")
    p_decrement_nc: float = Field(default=0.04, description="Probability of decrementing Nc.")
    p_swap_cores: float = Field(default=0.06, description="Probability of swapping two cores.")

    # Cost weights
    weight_inconsistencies: float = Field(default=1.0, description="Weight of routing violations.")
    alpha: float = Field(default=0.01, description="Core area weight.")
    beta: float = Field(default=0.05, description="Router area weight.")
    gamma: float = Field(default=0.1, description="Wasted slot penalty.")

    # Annealing params
    T_start: float = Field(default=50.0)
    T_min: float = Field(default=0.01)
    alpha_temp: float = Field(default=0.98)
    steps_per_T: int = Field(default=10)
    time_limit: Optional[float] = Field(default=10, description="Time limit in seconds for simulated annealing.")
    slice_factor: int = Field(default=2)
    seed: int = Field(default=42)
    verbose: bool = Field(default=False)

    def create_mapper(self) -> "SimAnnealingInferredHWMapper":
        from netochi.mapping.sa_ihw_mapper import SimAnnealingInferredHWMapper
        return SimAnnealingInferredHWMapper(config=self)

    def create(self) -> "SimAnnealingInferredHWMapper":
        return self.create_mapper()

