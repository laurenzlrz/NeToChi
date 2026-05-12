import os
import re
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

class PipelineOutputConfig(BaseModel):
    """
    Configuration for pipeline output storage and plotting.
    """
    model_config = ConfigDict(frozen=True)
    
    base_dir: str = "results"
    run_prefix: str = "run_"
    plot_format: str = "png"
    
    # NICE palette: DeepMind-ish / Modern look
    palette: List[str] = Field(
        default_factory=lambda: [
            "#0077BB", # Blue
            "#EE7733", # Orange
            "#009988", # Teal
            "#CC3311", # Red
            "#EE3377", # Magenta
            "#BBBBBB"  # Grey
        ]
    )
    
    csv_filename: str = "results.csv"
    json_filename: str = "results.json"
    
    # Format string for plots: e.g. "{metric}_by_{category}"
    plot_filename_pattern: str = "{metric}_comparison"

    def get_next_run_dir(self) -> Path:
        """
        Finds the next available run directory by incrementing the index.
        """
        base = Path(self.base_dir)
        base.mkdir(parents=True, exist_ok=True)
        
        existing_runs = []
        if base.exists():
            for d in base.iterdir():
                if d.is_dir() and d.name.startswith(self.run_prefix):
                    match = re.search(rf"{self.run_prefix}(\d+)", d.name)
                    if match:
                        existing_runs.append(int(match.group(1)))
        
        next_id = max(existing_runs, default=0) + 1
        run_dir = base / f"{self.run_prefix}{next_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
