import csv
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from netochi.pipeline.results import PipelineSummary
from netochi.pipeline.config import PipelineOutputConfig

class ResultManager(BaseModel):
    """
    Manages persistence of pipeline results to disk.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    config: PipelineOutputConfig
    
    def save(self, summary: PipelineSummary) -> Path:
        """
        Saves the summary to a new run directory.
        Returns the path to the run directory.
        """
        run_dir = self.config.get_next_run_dir()
        
        # 1. Save Full JSON
        json_path = run_dir / self.config.json_filename
        with open(json_path, "w") as f:
            f.write(summary.model_dump_json(indent=4))
            
        # 2. Save CSV for easy analysis
        csv_path = run_dir / self.config.csv_filename
        self._save_to_csv(summary, csv_path)
        
        return run_dir

    def load(self, run_dir: Path) -> PipelineSummary:
        """
        Loads a PipelineSummary from a run directory.
        """
        json_path = run_dir / self.config.json_filename
        if not json_path.exists():
            raise FileNotFoundError(f"Result file not found: {json_path}")
            
        with open(json_path, "r") as f:
            data = json.load(f)
            return PipelineSummary.model_validate(data)

    def _save_to_csv(self, summary: PipelineSummary, path: Path) -> None:
        """
        Flattens results into a CSV file.
        """
        if not summary.results:
            return
            
        # Collect all possible headers to ensure consistency
        headers = {"mapper", "execution_time_s", "error"}
        for res in summary.results:
            headers.update(res.input_metadata.keys())
            headers.update({f"metric_{k}" for k in res.metrics.keys()})
            headers.update({f"raw_{k}" for k in res.raw_metrics.keys()})
            
        sorted_headers = sorted(list(headers))
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_headers)
            writer.writeheader()
            for res in summary.results:
                row = {
                    "mapper": res.mapper_name,
                    "execution_time_s": str(res.execution_time_s),
                    "error": str(res.error) if res.error else ""
                }
                row.update(res.input_metadata)
                row.update({f"metric_{k}": str(v) for k, v in res.metrics.items()})
                row.update({f"raw_{k}": str(v) for k, v in res.raw_metrics.items()})
                
                # Fill missing columns with empty string
                full_row = {h: row.get(h, "") for h in sorted_headers}
                writer.writerow(full_row)
