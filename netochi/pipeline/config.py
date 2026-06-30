import re
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, ClassVar

import pandas as pd
import icontract
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from netochi.definitions.exceptions import InvalidConfigError
from netochi.result_processing.config import RUN_PREFIX, OUTPUT_DIR_RESULTS


def find_repo_root(start_path: Path = Path(__file__).resolve()) -> Path:
    """Traverses upwards to find the directory that contains the 'netochi' folder."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "netochi").is_dir():
            return parent
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


_CREATED_PATHS = set()


class PipelineOutputConfig(BaseModel):
    """
    Configuration for pipeline output storage and plotting.
    """
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    base_dir_name: Path = Field(
        default=Path(OUTPUT_DIR_RESULTS),
        description="Base directory for storing pipeline outputs (relative to repo root)."
    )
    run_prefix: str = Field(
        default=RUN_PREFIX,
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

    def create(self) -> "PipelineOutput":
        """
        Creates and initializes the PipelineOutput manager class.
        """
        return PipelineOutput(config=self)


class PipelineOutput:
    """
    Manages the directories and operations for pipeline outputs.
    All relative paths are resolved against the repository root at initialization.
    """

    @icontract.require(lambda config: isinstance(config, PipelineOutputConfig))
    @icontract.require(lambda config: PipelineOutput._validate(config))
    def __init__(self, config: PipelineOutputConfig) -> None:
        self.config = config

        # Normalize and store configuration values
        self.plot_format = [
            fmt.strip().lower().lstrip(".") for fmt in config.plot_format if fmt
        ]
        self.palette = config.palette
        self.plot_filename_pattern = config.plot_filename_pattern

        # Paths resolution
        repo_root = find_repo_root()
        self.base_path = repo_root / config.base_dir_name
        self.run_path = self._get_next_run_dir(self.base_path, config.run_prefix)
        self.dumps_path = self.run_path / "dumps"
        self.csv_path = self.run_path
        self.plot_path = self.run_path / "plots"

        # Handle directory creation on disk (if this run_path hasn't been created yet)
        if self.run_path not in _CREATED_PATHS:
            if self.run_path.exists():
                raise InvalidConfigError(f"Run directory '{self.run_path}' already exists.")
            self.create_run_directory()
            _CREATED_PATHS.add(self.run_path)

    @staticmethod
    def _validate(config: PipelineOutputConfig) -> bool:
        """
        Pure validation function to check config properties.
        """
        if not re.match(r"^[a-zA-Z0-9_\-]+$", config.run_prefix):
            raise InvalidConfigError(f"run_prefix '{config.run_prefix}' contains invalid filesystem characters.")

        if not config.plot_format:
            raise InvalidConfigError("plot_format list cannot be empty.")

        for fmt in config.plot_format:
            cleaned = fmt.strip().lower().lstrip(".")
            if not cleaned or not re.match(r"^[a-zA-Z0-9]+$", cleaned):
                raise InvalidConfigError(f"Invalid plot format '{fmt}' in plot_format list.")

        if "{metric}" not in config.plot_filename_pattern:
            raise InvalidConfigError("plot_filename_pattern must contain the '{metric}' placeholder.")
        return True

    def create_run_directory(self) -> None:
        """
        Explicit execution method to safely create directory structures on disk.
        """
        self.run_path.mkdir(parents=True, exist_ok=False)
        self.dumps_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_next_run_dir(cls, base_path: Path, run_prefix: str) -> Path:
        """
        Finds the next available run directory by incrementing the index.
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

    def print_console(self, msg: str, name: Optional[str] = None) -> None:
        """Utility method for consistent console output."""
        print(msg)
        if name:
            save_path = self.dumps_path / f"{name}.txt"
            with save_path.open(mode="a", encoding="utf-8") as f:
                f.write(f"{msg}\n")

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
