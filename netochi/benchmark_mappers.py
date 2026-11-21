"""
Runtime benchmark for the mappers across a list of hardware configurations.

For every (input generator, hardware config, seed) a mapping input is generated, and every selected
mapper is run on it. Each run executes in its own subprocess, so a run that exceeds the time limit
(or crashes, or runs out of memory) is killed without taking the benchmark down. Results are written
to a CSV after every run, and a plot of runtime and relative inconsistency vs. hardware size is drawn.

Timing:
  - mapper_time_s  wall time of mapper.run() measured with time.perf_counter(); input generation,
                   evaluation and process start-up are not included. perf_counter is monotonic and on
                   macOS does not advance while the machine sleeps.
  - cpu_time_s     time.process_time() of the worker process (all its threads, but not solver
                   subprocesses such as CBC), useful to spot multi-threaded or out-of-process solvers.
  - On macOS the script runs `caffeinate -i` for its lifetime so the laptop does not go to idle sleep
    (pass --allow-sleep to disable). Closing the lid still sleeps the machine.

Once a mapper hits the time limit on every seed of a hardware config, it is skipped for the larger configs
of that generator (they are ordered by total_neurons).

Usage:
    python -m netochi.benchmark_mappers --list                       # show available mappers / hardware
    python -m netochi.benchmark_mappers                              # default mappers and hardware
    python -m netochi.benchmark_mappers --mappers three_step theoretical_bounds --hw small medium \
        --generators mosaic_p05 --seeds 0 1 2 --time-limit 300
    python -m netochi.benchmark_mappers --plot-only results/mapper_benchmark/mapper_benchmark.csv
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from netochi.input_generator.mosaic_hardware_config import MosaicHardwareConfig

OUTPUT_DIR = os.path.join("results", "mapper_benchmark")
DEFAULT_TIME_LIMIT_S = 3000.0
DEFAULT_SEEDS = [1, 2]#, 3]

MOSAIC_PROBABILITIES = {"mosaic_p05": 0.5} #, "mosaic_p10": 1.0}   # generator key -> edge probability

# ======================= HARDWARE CONFIGS ====================================
# Keyed by a short name for the CLI; always benchmarked in order of total_neurons.

HW_CONFIGS: Dict[str, MosaicHardwareConfig] = {
    "mini": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=4, router_levels=2, slice_factor=2),
    "small": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=16, router_levels=2, slice_factor=2),
    "medium": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=16, router_levels=3, slice_factor=2),
    "large": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=32, router_levels=3, slice_factor=2),
    "large_2": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=64, router_levels=3, slice_factor=2),
    "large_3": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=64, router_levels=4, slice_factor=2),
    "large_4": MosaicHardwareConfig(nodes_per_router=2, neurons_per_core=128, router_levels=4, slice_factor=2),
}
DEFAULT_HW = ["mini", "small", "medium", "large", "large_2", "large_3", "large_4"]


# ======================= MAPPERS =============================================
# Each entry builds a fresh mapper. Imports happen inside the builders so the worker process only
# loads what it needs. `time_limit_s` is passed so mappers with their own limit can stop gracefully.

def _feedback_loop_random(_: float) -> Any:
    from netochi.mapping.feedback_loop_mapper import FeedbackLoopConfig
    from netochi.mapping.three_step_mapping.clustering.clusterer.random_clusterer import RandomClusterer  # ADAPT path
    from netochi.mapping.three_step_mapping.local_address_assignment.random_address_assigner import RandomAddressAssigner
    return FeedbackLoopConfig(core_initializer=RandomClusterer(),
                              address_initializer=RandomAddressAssigner()).create()


def _three_step(_: float) -> Any:
    from netochi.mapping.bottom_up_mapper import ConstraintGraphMode, SliceSolveMode
    from netochi.mapping.three_step_mapping.clustering.clusterer.kaMinPar_cluster import KaMinParHierarchicalClusterer
    from netochi.mapping.three_step_mapping.local_address_assignment.top_down_local_address_assigner import \
        TopDownLocalAddressAssignerConfig
    from netochi.mapping.three_step_mapping.slice_assignment.optimal_slice_assigner import OptimalSliceAssigner
    from netochi.mapping.three_step_mapping.three_step_hw_mapper import ThreeStepHwMapperConfig
    return ThreeStepHwMapperConfig(
        clusterer=KaMinParHierarchicalClusterer(),
        address_assigner=TopDownLocalAddressAssignerConfig(
            slice_solve_mode=SliceSolveMode.KL_CONSTRAINED_BISECTION,
            constraint_graph_mode=ConstraintGraphMode.CLIQUE).create(),
        slice_assigner=OptimalSliceAssigner()).create()


def _theoretical_bounds(_: float) -> Any:
    from netochi.mapping.theoretical_bounds_mapper import TheoreticalBoundsMapperConfig
    return TheoreticalBoundsMapperConfig().create()


MAPPERS: Dict[str, Callable[[float], Any]] = {
    "feedback_loop": _feedback_loop_random,
    "three_step": _three_step,
    "theoretical_bounds": _theoretical_bounds,
}
DEFAULT_MAPPERS = list(MAPPERS)


# ======================= INPUT GENERATORS ====================================

def make_input(generator: str, hw: MosaicHardwareConfig, seed: int) -> Any:
    if generator not in MOSAIC_PROBABILITIES:
        raise ValueError(f"Unknown generator {generator!r}")
    from netochi.input_generator.mosaic_network_factory import MosaicNetworkConfig
    factory = MosaicNetworkConfig(hw_config=hw, probability=MOSAIC_PROBABILITIES[generator], seed=seed).create()
    return factory.generate()


GENERATORS = list(MOSAIC_PROBABILITIES)

# ==============================================================================

FIELDS = [
    "generator", "hw", "nodes_per_router", "router_levels", "neurons_per_core", "total_cores", "total_neurons",
    "seed", "mapper", "edges", "status", "mapper_time_s", "cpu_time_s", "input_time_s",
    "inconsistency", "inconsistency_relative", "error",
]
STATUS_OK, STATUS_TIMEOUT, STATUS_ERROR, STATUS_SKIPPED = "ok", "timeout", "error", "skipped"


def _worker(generator: str, hw: MosaicHardwareConfig, seed: int, mapper_key: str, time_limit_s: float,
            quiet: bool, result_queue: "mp.Queue[Dict[str, Any]]") -> None:
    """Runs one (input, mapper) case in a fresh process and puts a result dict on the queue."""
    os.setpgrp()  # own process group, so the parent can also kill solver subprocesses on timeout
    if quiet:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)  # file-descriptor level, also silences solver subprocess logs
        os.dup2(devnull, 2)  # drops library warnings (e.g. GTK); errors are still reported via the queue
    result: Dict[str, Any] = {"status": STATUS_ERROR}
    try:
        import netochi.mapping.interfaces  # noqa: F401  must come first: objectives <-> mapping import cycle
        from netochi.objectives.obj_inconsistency import InconsistencyObjectiveConfig, InconsistencyRelativeObjectiveConfig

        t0 = time.perf_counter()
        mapping_input = make_input(generator, hw, seed)
        result["input_time_s"] = time.perf_counter() - t0
        result["edges"] = mapping_input.graph.num_edges()

        mapper = MAPPERS[mapper_key](time_limit_s)
        cpu0 = time.process_time()
        t0 = time.perf_counter()
        try:
            state = mapper.run(mapping_input)
        finally:  # also recorded when the mapper raises, to tell "gave up at the time limit" from a crash
            result["mapper_time_s"] = time.perf_counter() - t0
            result["cpu_time_s"] = time.process_time() - cpu0

        result["inconsistency"] = InconsistencyObjectiveConfig().create().evaluate(state).value
        result["inconsistency_relative"] = InconsistencyRelativeObjectiveConfig().create().evaluate(state).value
        result["status"] = STATUS_OK
    except BaseException as e:  # report anything, including SystemExit from native libraries
        result["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:500]
        if not quiet:
            traceback.print_exc()
    result_queue.put(result)


def _kill_group(proc: "mp.process.BaseProcess") -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.join()


def run_case(ctx: Any, generator: str, hw_name: str, seed: int, mapper_key: str, time_limit_s: float,
             quiet: bool) -> Dict[str, Any]:
    hw = HW_CONFIGS[hw_name]
    row: Dict[str, Any] = {
        "generator": generator, "hw": hw_name, "nodes_per_router": hw.nodes_per_router,
        "router_levels": hw.router_levels, "neurons_per_core": hw.neurons_per_core, "total_cores": hw.total_cores,
        "total_neurons": hw.total_neurons, "seed": seed, "mapper": mapper_key,
    }
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(generator, hw, seed, mapper_key, time_limit_s, quiet, result_queue))
    proc.start()
    # The limit covers the whole worker (input generation + mapping + evaluation) with some slack for start-up.
    deadline = time_limit_s * 1.1 + 30
    try:
        result = result_queue.get(timeout=deadline)
        proc.join(timeout=30)
        if proc.is_alive():
            _kill_group(proc)
    except queue.Empty:
        _kill_group(proc)
        exit_code = proc.exitcode
        if exit_code is not None and exit_code != -signal.SIGKILL:
            result = {"status": STATUS_ERROR, "error": f"worker died with exit code {exit_code}"}
        else:
            result = {"status": STATUS_TIMEOUT, "error": f"killed after {deadline:.0f}s"}
    row.update(result)
    # A run that used up the time limit counts as a timeout, whether it still returned a mapping
    # (e.g. the ILP's best incumbent) or raised (e.g. no feasible solution found in time).
    if row["status"] in (STATUS_OK, STATUS_ERROR) and row.get("mapper_time_s", 0.0) >= 0.99 * time_limit_s:
        row["status"] = STATUS_TIMEOUT
    return row


def run_benchmark(mappers: List[str], hw_names: List[str], generators: List[str], seeds: List[int],
                  time_limit_s: float, csv_path: str, quiet: bool) -> List[Dict[str, Any]]:
    ctx = mp.get_context("spawn")
    hw_names = sorted(hw_names, key=lambda name: HW_CONFIGS[name].total_neurons)
    rows: List[Dict[str, Any]] = []
    total = len(generators) * len(hw_names) * len(seeds) * len(mappers)
    done = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for generator in generators:
            given_up: set = set()  # mappers that timed out on all seeds of a smaller hardware
            for hw_name in hw_names:
                hw = HW_CONFIGS[hw_name]
                status_per_mapper: Dict[str, List[str]] = defaultdict(list)
                for seed in seeds:
                    for mapper_key in mappers:
                        done += 1
                        if mapper_key in given_up:
                            row = {"generator": generator, "hw": hw_name, "total_neurons": hw.total_neurons,
                                   "total_cores": hw.total_cores, "nodes_per_router": hw.nodes_per_router,
                                   "router_levels": hw.router_levels, "neurons_per_core": hw.neurons_per_core,
                                   "seed": seed, "mapper": mapper_key, "status": STATUS_SKIPPED,
                                   "error": "timed out on a smaller hardware config"}
                        else:
                            row = run_case(ctx, generator, hw_name, seed, mapper_key, time_limit_s, quiet)
                            status_per_mapper[mapper_key].append(row["status"])
                        writer.writerow(row)
                        f.flush()
                        rows.append(row)
                        print(f"[{done:>4}/{total}] {generator:<12} hw={hw_name:<13} N={hw.total_neurons:>5} "
                              f"seed={seed:<3} {mapper_key:<24} {_fmt_status(row)}", flush=True)
                for mapper_key, statuses in status_per_mapper.items():
                    if statuses and all(s == STATUS_TIMEOUT for s in statuses):
                        given_up.add(mapper_key)
    return rows


def _fmt_status(row: Dict[str, Any]) -> str:
    if row["status"] == STATUS_OK:
        return (f"time={row['mapper_time_s']:9.3f}s cpu={row['cpu_time_s']:9.3f}s "
                f"incons_rel={row['inconsistency_relative']:.4f}")
    elapsed = f"time={row['mapper_time_s']:9.3f}s " if row.get("mapper_time_s") is not None else ""
    return f"{row['status'].upper()}  {elapsed}{row.get('error', '')}"


def load_csv(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("total_neurons", "total_cores", "seed"):
            r[key] = int(r[key])
        for key in ("mapper_time_s", "cpu_time_s", "input_time_s", "inconsistency", "inconsistency_relative"):
            r[key] = float(r[key]) if r.get(key) not in (None, "") else float("nan")
    return rows


# ==============================================================================
# Plotting
# ==============================================================================

# Categorical slots of the default dataviz palette, assigned in fixed order (never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#1f1e1c"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def plot_results(rows: List[Dict[str, Any]], time_limit_s: float, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generators = list(dict.fromkeys(r["generator"] for r in rows))
    mappers = list(dict.fromkeys(r["mapper"] for r in rows))
    if len(mappers) > len(SERIES_COLORS):
        print(f"Warning: {len(mappers)} mappers but only {len(SERIES_COLORS)} distinguishable colors; "
              f"plotting the first {len(SERIES_COLORS)}. Use --mappers to split the plot.")
        mappers = mappers[:len(SERIES_COLORS)]
    color = {m: SERIES_COLORS[i] for i, m in enumerate(mappers)}  # color follows the mapper, not its rank

    fig, axes = plt.subplots(len(generators), 2, figsize=(13, 4.2 * len(generators) + 1.2), squeeze=False,
                             constrained_layout=True)
    for gi, generator in enumerate(generators):
        ax_t, ax_q = axes[gi]
        for mapper in mappers:
            mrows = [r for r in rows if r["generator"] == generator and r["mapper"] == mapper]
            sizes = sorted({r["total_neurons"] for r in mrows})
            med_n, med_t, med_q, timeout_n = [], [], [], []
            for n in sizes:
                runs = [r for r in mrows if r["total_neurons"] == n]
                ok = [r for r in runs if r["status"] == STATUS_OK]
                if any(r["status"] == STATUS_TIMEOUT for r in runs):
                    timeout_n.append(n)
                # Median only over sizes where every attempted seed finished; partial medians would be biased low.
                attempted = [r for r in runs if r["status"] in (STATUS_OK, STATUS_TIMEOUT)]
                if ok and len(ok) == len(attempted):
                    med_n.append(n)
                    med_t.append(float(np.median([r["mapper_time_s"] for r in ok])))
                    med_q.append(float(np.median([r["inconsistency_relative"] for r in ok])))
                ok_n = [r["total_neurons"] for r in ok]
                ax_t.scatter(ok_n, [r["mapper_time_s"] for r in ok], color=color[mapper], s=14, alpha=0.35,
                             linewidths=0, zorder=2)
                ax_q.scatter(ok_n, [r["inconsistency_relative"] for r in ok], color=color[mapper], s=14,
                             alpha=0.35, linewidths=0, zorder=2)
            for ax, ys in ((ax_t, med_t), (ax_q, med_q)):
                ax.plot(med_n, ys, color=color[mapper], linewidth=2, marker="o", markersize=7,
                        markeredgecolor="white", markeredgewidth=1.5, label=mapper, zorder=3)
            if timeout_n:
                ax_t.scatter(timeout_n, [time_limit_s] * len(timeout_n), facecolors="none", edgecolors=color[mapper],
                             s=70, linewidths=1.5, zorder=3)

        ax_t.axhline(time_limit_s, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
        ax_t.annotate(f"time limit {time_limit_s:g} s (hollow = hit limit)", xy=(0.01, time_limit_s),
                      xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                      fontsize=8, color=TEXT_SECONDARY)
        ax_t.set_yscale("log")
        ax_t.set_ylim(top=time_limit_s * 3)
        ax_t.set_xscale("log", base=2)
        ax_q.set_xscale("log", base=2)
        _style_axis(ax_t, f"{generator}: mapper runtime", "wall time [s]")
        _style_axis(ax_q, f"{generator}: relative inconsistency (lower is better)", "inconsistent edges / edges")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4), frameon=False, fontsize=9)
    fig.suptitle("Mapper benchmark — line = median over seeds (only sizes where all seeds finished), "
                 "dots = individual seeds", fontsize=10, color=TEXT_SECONDARY)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _style_axis(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("total neurons (hardware size)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


# ==============================================================================

def prevent_sleep() -> Optional[subprocess.Popen]:
    """On macOS, keeps the machine from idle-sleeping until this process exits."""
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except FileNotFoundError:
        print("Warning: caffeinate not found, the machine may go to sleep during the benchmark.")
        return None


def _check_keys(given: List[str], available: Dict[str, Any] | List[str], what: str) -> None:
    unknown = [k for k in given if k not in available]
    if unknown:
        sys.exit(f"Unknown {what}: {', '.join(unknown)}. Available: {', '.join(available)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mappers", nargs="+", default=DEFAULT_MAPPERS, help="Mapper keys (see --list).")
    parser.add_argument("--hw", nargs="+", default=DEFAULT_HW, help="Hardware config keys (see --list).")
    parser.add_argument("--generators", nargs="+", default=GENERATORS, help=f"Any of {', '.join(GENERATORS)}.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S, help="Per-run time limit [s].")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--plot-only", metavar="CSV", help="Skip benchmarking and re-plot an existing CSV.")
    parser.add_argument("--allow-sleep", action="store_true", help="Do not run caffeinate on macOS.")
    parser.add_argument("--verbose", action="store_true", help="Show mapper/solver output and tracebacks.")
    parser.add_argument("--list", action="store_true", help="List available mappers and hardware configs.")
    args = parser.parse_args()

    if args.list:
        print("Mappers:   " + ", ".join(MAPPERS))
        print("Hardware:")
        for name, hw in sorted(HW_CONFIGS.items(), key=lambda kv: kv[1].total_neurons):
            print(f"  {name:<14} {hw.nodes_per_router}^{hw.router_levels} cores x {hw.neurons_per_core} neurons "
                  f"= {hw.total_neurons}")
        print("Generators: " + ", ".join(GENERATORS))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "mapper_benchmark.pdf")

    if args.plot_only:
        rows = load_csv(args.plot_only)
    else:
        _check_keys(args.mappers, MAPPERS, "mappers")
        _check_keys(args.hw, HW_CONFIGS, "hardware configs")
        _check_keys(args.generators, GENERATORS, "generators")
        if not args.allow_sleep:
            prevent_sleep()
        csv_path = os.path.join(args.output_dir, "mapper_benchmark.csv")
        start = time.perf_counter()
        rows = run_benchmark(args.mappers, args.hw, args.generators, args.seeds, args.time_limit, csv_path,
                             quiet=not args.verbose)
        print(f"Finished in {time.perf_counter() - start:.1f}s. Saved results to {csv_path}")
        rows = load_csv(csv_path)

    if any(r["status"] == STATUS_OK and not math.isnan(r["mapper_time_s"]) for r in rows):
        plot_results(rows, args.time_limit, plot_path)
    else:
        print("No successful runs, nothing to plot.")


if __name__ == "__main__":
    main()