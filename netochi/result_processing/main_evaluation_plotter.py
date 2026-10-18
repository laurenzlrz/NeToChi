import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# input ids look like
#   Mosaic_1024n_2R_4l_64N_0.5p
#   Noisy_Mosaic_2048n_2R_4l_128N_0.5p_0.1eps
INPUT_ID_RE = re.compile(r"^(?P<base>.*)_(?P<prob>[0-9.]+)p(?:_(?P<eps>[0-9.]+)eps)?$")

# mapper names look like 3stepHw_<Clusterer>_<AddressAssigner>_Opt
MAPPER_RE = re.compile(r"^3stepHw_(?P<clusterer>.+?)_(?:RandomAddressAssigner|[A-Za-z]+AddressAssigner)_Opt$")


def parse_input_id(input_id: str) -> pd.Series:
    """Split an input id into graph type, connection probability and noise level."""
    m = INPUT_ID_RE.match(str(input_id))
    if not m:
        return pd.Series([str(input_id), "Unknown", None])
    base, prob, eps = m.group("base"), m.group("prob"), m.group("eps")
    # noisy graphs with different eps are different graph types
    graph_type = f"{base}_{eps}eps" if eps is not None else base
    return pd.Series([graph_type, prob, eps])


def short_mapper_name(mapper: str) -> str:
    """3stepHw_KaMinPar_RandomAddressAssigner_Opt -> KaMinPar (unchanged if it doesn't match)."""
    m = MAPPER_RE.match(str(mapper))
    return m.group("clusterer") if m else str(mapper)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "metric" not in df.columns:  # file without header row
        df = pd.read_csv(path, header=None, names=["mapper", "input_id", "seed", "metric", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    missing = df["value"].isna().sum()
    if missing:
        print(f"Warning: dropping {missing} rows without a numeric value (e.g. unfinished runs)")
        df = df.dropna(subset=["value"])
    df[["graph_type", "prob", "eps"]] = df["input_id"].apply(parse_input_id)
    df["mapper_short"] = df["mapper"].apply(short_mapper_name)
    return df


def plot_metric(df: pd.DataFrame, metric: str, title: str, y_label: str, out_dir: Path,
                y_lim=None, show: bool = False) -> None:
    df_m = df[df["metric"] == metric]
    if df_m.empty:
        print(f"No rows for metric '{metric}', skipping")
        return

    mapper_order = sorted(df_m["mapper_short"].unique(), key=str.lower)
    safe_metric = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_")

    for graph in sorted(df_m["graph_type"].unique()):
        subset = df_m[df_m["graph_type"] == graph]
        prob_order = sorted(subset["prob"].unique(), key=lambda p: float(p) if p != "Unknown" else -1)
        n_seeds = subset.groupby(["mapper_short", "prob"])["value"].count().max()

        has_prob = (subset["prob"] != "Unknown").any()
        order = [m for m in mapper_order if m in set(subset["mapper_short"])]

        plt.figure(figsize=(10, 6))
        if has_prob:
            sns.barplot(data=subset, x="mapper_short", y="value",
                        hue="prob", hue_order=prob_order, palette="viridis",
                        order=order, errorbar="sd", capsize=0.1)
        else:
            sns.barplot(data=subset, x="mapper_short", y="value",
                        color=sns.color_palette("viridis", 3)[1],
                        order=order, errorbar="sd", capsize=0.1)

        plt.title(f"{title} for {graph}", fontsize=14, fontweight="bold")
        plt.xlabel("Mapper", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        if y_lim is not None:
            plt.ylim(*y_lim)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        if has_prob:
            plt.legend(title="Connection probability", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.figtext(0.01, 0.01, f"mean ± std over up to {n_seeds} seeds", fontsize=8, color="gray")
        plt.tight_layout()

        filename = out_dir / f"{safe_metric}_{graph}.pdf"
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.close()
        print(f"Saved {filename}")


if __name__ == "__main__":
    #RUN_NAME = "inconsistencies_evaluation"
    RUN_NAME = "inconsistencies_evaluation"
    results = Path(f"results/{RUN_NAME}/evaluation_results.csv")
    out_dir = Path(f"results/{RUN_NAME}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results)

    # metric in the csv -> (title, y-axis label, y-limits)
    METRICS = {
        #"Accuracy (Clustering)": ("Clustering Accuracy", "Accuracy", (0, 1.05)),
        #"Core Dist. Diff. (Clustering)": ("Difference in Core Distances", "Core distance difference", None),
        "Inconsistencies (Fraction)": ("Inconsistencies", "Inconsistencies (Fraction)", (0, 1.05)),
        "Time (s)": ("Runtime", "Time (s)", None),
    }
    for metric, (title, y_label, y_lim) in METRICS.items():
        plot_metric(df, metric, title, y_label, out_dir, y_lim=y_lim)