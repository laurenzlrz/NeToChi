import re
import json
import csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from netochi.definitions.exceptions import InvalidConfigError
from netochi.pipeline import PipelineSummary


def find_repo_root(start_path: Path = Path(__file__).resolve()) -> Path:
    """Traverses upwards to find the directory that contains the 'netochi' folder."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "netochi").is_dir():
            return parent
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


class PipelineOutputConfig(BaseModel):
    """
    Configuration for pipeline output storage and plotting.
    All relative paths are automatically resolved against the repository root.
    """
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    base_dir_name: Path = Field(
        default=Path("results"),
        description="Base directory for storing pipeline outputs (relative to repo root)."
    )
    base_path: Path = Field(
        default=None,
        description="Base directory for storing pipeline outputs (absolute path, resolved at initialization)."
    )
    run_path: Path = Field(
        default=None,
        description="Resolved path for the current run's output directory (set at runtime)."
    )
    run_prefix: str = Field(
        default="run_",
        description="Prefix for individual run directories."
    )
    plot_format: List[str] = Field(
        default_factory=lambda: ["png"],
        description="File format for saved plots (e.g., 'png', 'pdf')."
    )
    palette: List[str] = Field(
        default_factory=lambda: [
            "#0077BB", "#EE7733", "#009988",
            "#CC3311", "#EE3377", "#BBBBBB"
        ]
    )
    plot_filename_pattern: str = Field(default="{metric}_comparison")
    plot_path: Path = Field(default=None, description="Directory for storing generated plots.")
    dumps_path: Path = Field(default=None, description="Directory for storing intermediate dumps.")
    csv_path: Path = Field(default=None, description="Path for storing the flattened CSV results.")

    # ==========================================================
    # PHASE 1: CONSTRUCTOR / MUTATOR
    # ==========================================================
    @model_validator(mode="before")
    @classmethod
    def construct_and_resolve_paths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Mutate and normalize strings/formats
        if "plot_format" in data:
            data["plot_format"] = [
                fmt.strip().lower().lstrip(".") for fmt in data["plot_format"] if fmt
            ]

        # 2. Extract values needed for path calculations
        run_prefix = data.get("run_prefix", "run_")
        base_dir_name = Path(data.get("base_dir_name", "results"))

        # 3. Path Generation Mutations via Class Helper
        repo_root = find_repo_root()
        base_path = repo_root / base_dir_name
        data["base_path"] = base_path
        # Call the helper cleanly without needing 'self'
        data["run_path"] = cls._get_next_run_dir(base_path, run_prefix)
        data["dumps_path"] = data["run_path"] / "dumps"
        data["csv_path"] = data["run_path"]
        data["plot_path"] = data["run_path"] / "plots"

        return data

    def create_run_directory(self) -> None:
        """
        Explicit execution method to safely create directory structures on disk
        post-instantiation.
        """
        self.run_path.mkdir(parents=True, exist_ok=False)

    @classmethod
    def _get_next_run_dir(cls, base_path: Path, run_prefix: str) -> Path:
        """
        Finds the next available run directory by incrementing the index.
        Safe to call before model instantiation.
        """

        existing_runs = []
        prefix_pattern = re.compile(rf"^{re.escape(run_prefix)}(\d+)")

        if not base_path.exists():
            return base_path / f"{run_prefix}001"

        for d in base_path.iterdir():
            if d.is_dir():
                match = prefix_pattern.match(d.name)
                if match:
                    existing_runs.append(int(match.group(1)))

        next_id = max(existing_runs, default=0) + 1
        return base_path / f"{run_prefix}{next_id:03d}"

    # ==========================================================
    # PHASE 2: VALIDATOR
    # ==========================================================
    @model_validator(mode="after")
    def validate_frozen_attributes(self) -> Self:
        """
        Pure validation layer. Inspects attributes on the frozen
        object and raises configuration errors without performing mutations.
        """
        # 1. String/Format Validations
        if not re.match(r"^[a-zA-Z0-9_\-]+$", self.run_prefix):
            raise InvalidConfigError(f"run_prefix '{self.run_prefix}' contains invalid filesystem characters.")

        if not self.plot_format:
            raise InvalidConfigError("plot_format list cannot be empty.")

        for fmt in self.plot_format:
            if not re.match(r"^[a-zA-Z0-9]+$", fmt):
                raise InvalidConfigError(f"Invalid plot format '{fmt}' in plot_format list.")

        if "{metric}" not in self.plot_filename_pattern:
            raise InvalidConfigError("plot_filename_pattern must contain the '{metric}' placeholder.")

        # 2. Path State Verification
        if self.run_path.exists():
            raise InvalidConfigError(f"Run directory '{self.run_path}' already exists.")
        self.create_run_directory()

        return self

    def print_console(self, msg: str, name: Optional[str] = None) -> None:
        """Utility method for consistent console output."""
        print(msg)
        if name:
            save_path = self.dumps_path / f"{name}.txt" if name else None
            save_path.write_text(msg, encoding="utf-8")

    def save_to_csv(self, csv: pd.DataFrame, name: str) -> None:
        """
        Flattens results into a CSV file.
        """
        save_path = self.csv_path / f"{name}.csv"
        csv.to_csv(save_path, index=False)

    def save_plot(self, plt, name: str):
        """
        Saves a matplotlib plot to the designated plot directory.
        """
        for fmt in self.plot_format:
            save_path = self.plot_path / f"{name}.{fmt}"
        plt.savefig(save_path, dpi=150)
        plt.close()
