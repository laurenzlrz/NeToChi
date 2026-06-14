import os
from pathlib import Path
from netochi.pipeline.config import PipelineOutputConfig
from netochi.pipeline.storage import ResultManager
from netochi.pipeline.plotter import PipelinePlotter

def replot_latest():
    # 1. Setup config
    config = PipelineOutputConfig(
        base_dir="results",
        plot_format="png"
    )
    
    base_path = Path(config.base_dir)
    if not base_path.exists():
        print(f"Base directory {base_path} does not exist.")
        return

    # 2. Find the latest run folder
    run_folders = []
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            parts = d.name.split("_")
            # Look for the last part if it's numeric
            if parts[-1].isdigit():
                run_folders.append((int(parts[-1]), d))
    
    if not run_folders:
        print("No numeric run folders found.")
        return
        
    # Sort by number and pick the highest
    run_folders.sort(key=lambda x: x[0], reverse=True)
    latest_run = run_folders[0][1]
    print(f"Latest run folder identified: {latest_run}")

    # 3. Load Results
    manager = ResultManager(config=config)
    try:
        summary = manager.load(latest_run)
        print(f"Successfully loaded results with {len(summary.results)} experiments.")
    except Exception as e:
        print(f"Failed to load results: {e}")
        return

    # 4. Generate Plots
    print(f"Generating asynchronous plots in {latest_run}/plot_asynch ...")
    plotter = PipelinePlotter(config=config)
    plotter.plot_all(summary, latest_run, plot_subfolder="plot_asynch")
    
    print("Plotting complete.")

if __name__ == "__main__":
    replot_latest()
