"""Experiment 2: running time and memory on synthetic instances.

Three sweeps (each over several random seeds):

* ``n``: number of candidate apps, fixed ``T`` and ``k``;
* ``T``: planning horizon, fixed ``n`` and ``k``;
* ``lam``: switching penalty, fixed ``n``, ``T``, ``k`` (affects pruning).

Branch and bound and CP-SAT get a ``--budget`` second time limit (the row's
``optimal`` column records whether optimality was proved). A solver is skipped
for larger sizes in a sweep once it exceeds the budget, so the sweep finishes
in reasonable time.

Example:
    python scripts/run_scalability.py --seeds 3 --budget 20
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd

from stableui.benchmark import measure
from stableui.data import synthetic_instance
from stableui.solvers import solve_branch_bound, solve_cpsat, solve_dp, solve_greedy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/scalability.csv")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=float, default=20.0, help="seconds per solve")
    parser.add_argument("--quick", action="store_true", help="small grid for smoke tests")
    return parser.parse_args()


def sweeps(quick: bool) -> list[tuple[str, list[dict]]]:
    """Return ``(sweep name, list of instance parameter dicts)`` pairs."""
    if quick:
        return [
            ("n", [{"n_apps": n, "horizon": 4, "k": 2, "lam": 0.2} for n in (4, 6, 8)]),
            ("T", [{"n_apps": 6, "horizon": t, "k": 2, "lam": 0.2} for t in (2, 4, 6)]),
            ("lam", [{"n_apps": 6, "horizon": 4, "k": 2, "lam": v} for v in (0.0, 0.5)]),
        ]
    return [
        (
            "n",
            [{"n_apps": n, "horizon": 6, "k": 3, "lam": 0.2} for n in (5, 8, 10, 15, 20, 30, 40)],
        ),
        (
            "T",
            [{"n_apps": 10, "horizon": t, "k": 3, "lam": 0.2} for t in (2, 4, 6, 8, 12, 24, 48)],
        ),
        (
            "lam",
            [
                {"n_apps": 10, "horizon": 6, "k": 3, "lam": v}
                for v in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
            ],
        ),
    ]


def main() -> None:
    """Run every sweep and write one CSV row per (instance, solver)."""
    args = parse_args()
    solvers = {
        "greedy": solve_greedy,
        "dp": solve_dp,
        "bnb": partial(solve_branch_bound, time_limit=args.budget),
        "cpsat": partial(solve_cpsat, time_limit=args.budget),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sweep, grid in sweeps(args.quick):
        over_budget: set[str] = set()
        for params in grid:
            for seed in range(args.seeds):
                inst = synthetic_instance(seed=seed, **params)
                optimum = None
                for name, solver in solvers.items():
                    if name in over_budget:
                        continue
                    row = measure(solver, inst, repeats=args.repeats, budget=args.budget)
                    if name == "dp":
                        optimum = row["utility"]
                    row.update(sweep=sweep, solver=name, seed=seed, L=inst.n_layouts, **params)
                    row["gap_to_dp"] = None if optimum is None else optimum - row["utility"]
                    rows.append(row)
                    print(
                        f"{sweep:>3} {params} seed={seed} {name:>6}: "
                        f"{row['time_median_s']:.4f}s util={row['utility']:.3f}"
                    )
                    if row["time_median_s"] >= args.budget or not (
                        row["optimal"] or name == "greedy"
                    ):
                        over_budget.add(name)
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
