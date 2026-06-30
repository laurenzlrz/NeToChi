import csv
import json
import os

from netochi.pipeline import PipelineSummary, ExperimentResult
from netochi.result_processing.config import OUTPUT_DIR_CSV


def export_summary_to_csv(summary: PipelineSummary) -> None:
    """
    Exports a PipelineSummary to a CSV file.
    Nested dictionaries are serialized as JSON strings to preserve their structure.
    """
    os.makedirs(OUTPUT_DIR_CSV, exist_ok=True)
    filename = "summary.csv"
    filepath = os.path.join(OUTPUT_DIR_CSV, filename)

    # Ensure the target directory exists if a path is provided
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "mapper_name",
        "input_id",
        "input_metadata",
        "metrics",
        "raw_metrics",
        "error"
    ]

    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for res in summary.results:
            row = {
                "mapper_name": res.mapper_name,
                "input_id": res.input_id,
                # Serialize dictionaries to JSON strings
                "input_metadata": json.dumps(res.input_metadata),
                "metrics": json.dumps(res.metrics),
                "raw_metrics": json.dumps(res.raw_metrics),
                # Handle the Optional string safely
                "error": res.error if res.error is not None else ""
            }
            writer.writerow(row)

    print(f"Successfully exported summary to {filepath}")


def import_summary_from_csv(filepath: str) -> PipelineSummary:
    """
    Imports a CSV file and reconstructs the PipelineSummary object.
    """
    results = []
    total_time = 0.0

    with open(filepath, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:

            error_val = row["error"]

            # Reconstruct the Pydantic model
            result = ExperimentResult(
                mapper_name=row["mapper_name"],
                input_id=row["input_id"],
                # Deserialize JSON strings back into Python dictionaries
                input_metadata=json.loads(row["input_metadata"]),
                metrics=json.loads(row["metrics"]),
                raw_metrics=json.loads(row["raw_metrics"]),
                error=error_val if error_val != "" else None
            )
            results.append(result)

    # Note: total_time_s is reconstructed as the sum of individual execution times.
    # If your original pipeline had global overhead time, it won't be captured here.
    return PipelineSummary(results=results, total_time_s=None)