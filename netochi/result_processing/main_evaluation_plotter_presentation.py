"""
Expects a long-format CSV with columns: mapper, input_id, seed, metric, value
  metric is "Inconsistencies (Fraction)" or "Time (s)".

Creates in output_dir (default: ./plots), as PDF:
  1_mosaic_inconsistencies_p05 / _p10
  2_noisy_mosaic_inconsistencies_p05 / _p10
  3_runtime
  4_swta_inconsistencies
  5_er_inconsistencies

Lines are means over seeds (runtime: median); the error bars show +- one standard
deviation over seeds (runtime: interquartile range, 25th-75th percentile).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

INC = "Inconsistencies (Fraction)"
TIME = "Time (s)"
SIZES = [128, 256, 512, 1024, 2048]
RUN_NAME = "inconsistencies_evaluation"

# Erdős–Rényi input graphs: number of neurons -> edge probability (has the same number of expected edges as equivalent Mosaic with p = 0.5)
ER_GRAPHS_05p = {
    128: 0.15354330708661418, # L = 3
    256: 0.15490196078431373, # L = 3
    512: 0.15557729941291584, # L = 3
    1024: 0.09335288367546432, # L = 4
}

# mapper -> (label, color, linestyle, marker, hollow marker)
STYLE = {
    "KaMinPar_KL": ("Three-Step", "#184774", "-", "o", False),
    "TheoreticalBoundsMapper": ("Theoretical Bounds", "#E97132", "--", "^", True),
    "Feedback_RandomClusterer_RandomAddressAssigner": ("Feedback Loop", "#196B24", "-", "s", False),
    "RandomMapper": ("Random", "#A6A6A6", "-", "D", False),
    "Feedback_KaMinPar_KL": ("Feedback Loop (Three-Step init)", "#4EA72E", "--", "s", True),
}
ORDER = ["KaMinPar_KL", "TheoreticalBoundsMapper", "Feedback_RandomClusterer_RandomAddressAssigner", "RandomMapper"]
SWTA_ORDER = ["KaMinPar_KL", "TheoreticalBoundsMapper", "Feedback_RandomClusterer_RandomAddressAssigner",
              "Feedback_KaMinPar_KL", "RandomMapper"]

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E3E3E3",
    "axes.edgecolor": "#BFBFBF",
    "axes.labelcolor": "#404040",
    "xtick.color": "#404040",
    "ytick.color": "#404040",
})


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["N"] = df["input_id"].str.extract(r"_(\d+)n", expand=False).astype(int)
    return df


def summarize(df: pd.DataFrame, pattern: str, metric: str, agg: str) -> pd.DataFrame:
    """Rows: number of neurons, columns: mapper. Aggregated over seeds."""
    sel = df[df["input_id"].str.contains(pattern, regex=True) & (df["metric"] == metric)]
    return sel.groupby(["N", "mapper"])["value"].agg(agg).unstack().reindex(SIZES)


def band_std(mean: pd.DataFrame, std: pd.DataFrame, lower: float = 0.0, upper: float = 1.0):
    """(lower, upper) tables for a +- one standard deviation band, clipped to [lower, upper]."""
    return (mean - std).clip(lower=lower), (mean + std).clip(upper=upper)


def _draw(ax, x, table, mappers, band):
    for m in mappers:
        if m not in table.columns:
            continue
        label, color, ls, marker, hollow = STYLE[m]
        xs = x
        y = table[m].values.astype(float)
        if band is not None:
            lo, hi = band
            yerr = [y - lo[m].values.astype(float), hi[m].values.astype(float) - y]
            ax.errorbar(xs, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.5,
                        capsize=4, capthick=1.5, zorder=1)
        ax.plot(xs, y, label=label, color=color, linestyle=ls, linewidth=2.25,
                marker=marker, markersize=8, markeredgewidth=1.5,
                markerfacecolor="white" if hollow else color, markeredgecolor=color, zorder=2)


def plot_lines(ax, table: pd.DataFrame, mappers, title=None, band=None):
    """band: optional (lower, upper) tables with the same shape as table."""
    x = list(range(len(SIZES)))  # evenly spaced = log2 scale for 128..2048
    _draw(ax, x, table, mappers, band)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("Number of neurons")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=14)


def legend_top(ax, ncol=2):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, frameon=False)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf",): # "png",
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}.png/.pdf")

def summarize_sizes(df: pd.DataFrame, pattern: str, metric: str, agg: str) -> pd.DataFrame:
    """Like summarize(), but keeps only the sizes present in the data."""
    sel = df[df["input_id"].str.contains(pattern, regex=True) & (df["metric"] == metric)]
    return sel.groupby(["N", "mapper"])["value"].agg(agg).unstack().sort_index()


def plot_lines_sizes(ax, table: pd.DataFrame, mappers, title=None, band=None, tick_labels=None):
    """tick_labels: optional dict number of neurons -> x tick label."""
    sizes = list(table.index)
    x = list(range(len(sizes)))
    _draw(ax, x, table, mappers, band)
    ax.set_xticks(list(x))
    ax.set_xticklabels([tick_labels.get(n, str(n)) if tick_labels else str(n) for n in sizes])
    ax.set_xlabel("Number of neurons")
    if title:
        ax.set_title(title, loc="left", pad=34, fontsize=13, fontweight="bold", color="#1F1F1F")

if __name__ == "__main__":
    csv_path = f"../../results/{RUN_NAME}/evaluation_results.csv"
    out_dir = f"../../results/{RUN_NAME}/clean_plots"

    df = load(csv_path)
    out = Path(out_dir)

    # ---------- 1) Mosaic: one plot per edge probability ----------
    mosaic_pat = {"0.5": r"^Mosaic_.*_0\.5p$", "1.0": r"^Mosaic_.*_1\.0p$"}
    mosaic = {p: summarize(df, pat, INC, "mean") for p, pat in mosaic_pat.items()}
    for p, table in mosaic.items():
        std = summarize(df, mosaic_pat[p], INC, "std")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_lines(ax, table, ORDER, band=band_std(table, std))
        ax.set_ylim(0, 0.9)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Inconsistencies")
        legend_top(ax)
        save(fig, out, f"1_mosaic_inconsistencies_p{p.replace('.', '')}")

    # ---------- 2) Noisy Mosaic: one plot per edge probability ----------
    noisy_pat = {"0.5": r"^Noisy_Mosaic_.*_0\.5p_", "1.0": r"^Noisy_Mosaic_.*_1\.0p_"}
    noisy = {p: summarize(df, pat, INC, "mean") for p, pat in noisy_pat.items()}
    # Upper bound on the optimal inconsistencies (computed separately), per number of neurons
    upper_bound = {
        "0.5": {128: 0.311, 256: 0.308, 512: 0.307, 1024: 0.466, 2048: 0.465},
    }
    for p, table in noisy.items():
        std = summarize(df, noisy_pat[p], INC, "std")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_lines(ax, table, ORDER, band=band_std(table, std))
        if p in upper_bound:
            ub = [upper_bound[p][n] for n in SIZES]
            ax.plot(range(len(SIZES)), ub, color="#1F1F1F", linestyle=":", linewidth=2,
                    marker="_", markersize=14, markeredgewidth=2,
                    label="Upper bound on optimum", zorder=1)
        ax.set_ylim(0, 0.9)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Inconsistencies")
        legend_top(ax, ncol=3)
        save(fig, out, f"2_noisy_mosaic_inconsistencies_p{p.replace('.', '')}")

    # ---------- 3) Runtime: median over seeds, Mosaic p = 0.5, log scale ----------
    rt_pat = r"^Mosaic_.*_0\.5p$"
    runtime = summarize(df, rt_pat, TIME, "median")
    q25 = summarize(df, rt_pat, TIME, lambda v: v.quantile(0.25))
    q75 = summarize(df, rt_pat, TIME, lambda v: v.quantile(0.75))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_lines(ax, runtime, ["KaMinPar_KL", "TheoreticalBoundsMapper", "Feedback_RandomClusterer_RandomAddressAssigner"],  # Random (< 1 ms) left out
               band=(q25, q75))
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Runtime in seconds (log scale)")
    legend_top(ax, ncol=3)
    save(fig, out, "3_runtime")

    # ---------- 4) sWTA ----------
    table = summarize_sizes(df, r"^sWTA_", INC, "mean")
    std = summarize_sizes(df, r"^sWTA_", INC, "std")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_lines_sizes(ax, table, SWTA_ORDER, band=band_std(table, std))
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Inconsistencies")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=11)
    save(fig, out, "4_swta_inconsistencies")
    print("\nsWTA:\n", (table * 100).round(1))

    # ---------- 5) Erdős–Rényi ----------
    # One input graph per size, with the edge probability given in ER_GRAPHS_05p
    er_pattern = "|".join(rf"^ER_{n}n_{re.escape(str(p))}p$" for n, p in ER_GRAPHS_05p.items())
    er_ticks = {n: f"{n}\np = {p:.3f}" for n, p in ER_GRAPHS_05p.items()}
    table = summarize_sizes(df, er_pattern, INC, "mean")
    std = summarize_sizes(df, er_pattern, INC, "std")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_lines_sizes(ax, table, ORDER, title="Erdős–Rényi", band=band_std(table, std),
                     tick_labels=er_ticks)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Inconsistencies")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False, fontsize=11)
    save(fig, out, "5_er_inconsistencies")
    print("\nErdős–Rényi:\n", (table * 100).round(1))


    # ---------- numbers used in the slide callouts ----------
    for p, table in mosaic.items():
        print(f"\nMosaic, p = {p}, mean inconsistencies:\n", (table * 100).round(1))
    for title, table in noisy.items():
        print(f"\nNoisy Mosaic, {title}:\n", (table * 100).round(1))
    print("\nRuntime (median, s):\n", runtime.round(2))
    fl, ts = runtime.loc[2048, "Feedback_RandomClusterer_RandomAddressAssigner"], runtime.loc[2048, "KaMinPar_KL"]
    print(f"\nSpeed-up Feedback Loop vs. Three-Step at 2048 neurons: {ts / fl:.0f}x")