"""Turn the experiment CSVs into the report's figures and summary tables.

Example:
    python scripts/make_plots.py --results results --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"greedy": "Greedy", "dp": "DP", "bnb": "Branch & Bound", "cpsat": "CP-SAT (SOTA)"}
COLORS = {"greedy": "#4C72B0", "dp": "#DD8452", "bnb": "#55A868", "cpsat": "#C44E52"}
MARKERS = {"greedy": "o", "dp": "s", "bnb": "^", "cpsat": "D"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    return parser.parse_args()


def _line(ax, df: pd.DataFrame, x: str, y: str, logy: bool = False) -> None:
    """Plot mean of ``y`` against ``x`` for each solver."""
    for name, group in df.groupby("solver"):
        agg = group.groupby(x)[y].mean()
        ax.plot(
            agg.index,
            agg.values,
            marker=MARKERS[name],
            color=COLORS[name],
            label=LABELS[name],
        )
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def scalability_figures(df: pd.DataFrame, out: Path) -> None:
    """Running time and memory versus n, T and lambda."""
    sweeps = [("n", "n_apps", "number of candidate apps $n$"), ("T", "horizon", "horizon $T$")]
    sweeps.append(("lam", "lam", r"switching penalty $\lambda$"))
    for metric, ylabel, fname in [
        ("time_median_s", "median running time (s)", "time"),
        ("peak_python_mb", "peak Python heap (MB)", "memory"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (sweep, col, xlabel) in zip(axes, sweeps, strict=True):
            sub = df[df["sweep"] == sweep]
            if sub.empty:
                ax.set_visible(False)
                continue
            _line(ax, sub, col, metric, logy=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"scalability_{fname}.png", dpi=200)
        plt.close(fig)

    bnb = df[(df["solver"] == "bnb") & df["nodes_expanded"].notna()]
    bnb = bnb.assign(full_tree=bnb["L"].astype(float) ** bnb["horizon"])
    if not bnb.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, (sweep, col, xlabel) in zip(axes, sweeps[1:], strict=True):
            sub = bnb[bnb["sweep"] == sweep].groupby(col)
            expanded = sub["nodes_expanded"].mean()
            full_tree = sub["full_tree"].mean()
            ax.plot(expanded.index, expanded.values, marker="^", label="nodes expanded")
            ax.plot(full_tree.index, full_tree.values, ls="--", label="$L^T$ (no pruning)")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "bnb_pruning.png", dpi=200)
        plt.close(fig)


def real_figures(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Coverage / switching trade-off on LSApp; returns the summary table."""
    h_max = df["horizon"].max()
    greedy = df[df["solver"] == "greedy"]
    exact = df[(df["solver"] != "greedy") & (df["horizon"] == h_max)]
    sel = pd.concat([greedy, exact])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("coverage", "coverage of actual launches"),
            ("switches_per_day", "shortcut replacements per day"),
            ("planning_utility", "planning utility (normalised)"),
        ],
        strict=True,
    ):
        _line(ax, sel, "lam", metric)
        ax.set_xlabel(r"switching penalty $\lambda$")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8, title=f"exact solvers: H={h_max}")
    fig.tight_layout()
    fig.savefig(out / "real_tradeoff.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for horizon, group in df[df["solver"] == "dp"].groupby("horizon"):
        agg = group.groupby("lam")[["switches_per_day", "coverage"]].mean()
        ax.plot(agg["switches_per_day"], agg["coverage"], marker="o", label=f"DP, H={horizon}")
    agg = greedy.groupby("lam")[["switches_per_day", "coverage"]].mean()
    ax.plot(agg["switches_per_day"], agg["coverage"], marker="x", ls="--", label="Greedy")
    ax.set_xlabel("replacements per day")
    ax.set_ylabel("coverage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "real_pareto.png", dpi=200)
    plt.close(fig)

    metrics = [
        "coverage",
        "switches_per_day",
        "planning_utility",
        "realized_utility",
        "solve_ms_per_decision",
    ]
    return df.groupby(["lam", "horizon", "solver"])[metrics].mean().round(4).reset_index()


def main() -> None:
    """Generate every figure and table that has input data."""
    args = parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scal = results / "scalability.csv"
    if scal.exists():
        df = pd.read_csv(scal)
        scalability_figures(df, out)
        cols = ["time_median_s", "peak_python_mb", "gap_to_dp", "optimal"]
        table = df.groupby(["sweep", "n_apps", "horizon", "lam", "L", "solver"])[cols].mean()
        table.round(5).to_csv(out / "scalability_table.csv")
    real = results / "real.csv"
    if real.exists():
        real_figures(pd.read_csv(real), out).to_csv(out / "real_table.csv", index=False)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
