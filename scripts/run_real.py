"""Experiment 1: receding-horizon simulation on LSApp users.

Example:
    python scripts/run_real.py --users 30 --lams 0 0.05 0.1 0.2 0.5 --horizons 1 3 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from stableui.data import build_user_data, dataset_summary, eligible_users, load_launches
from stableui.simulation import simulate, simulation_row
from stableui.solvers import SOLVERS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/lsapp.tsv")
    parser.add_argument("--out", default="results/real.csv")
    parser.add_argument("--users", type=int, default=30, help="max number of users (0 = all)")
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--period-minutes", type=int, default=60)
    parser.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS))
    parser.add_argument("--min-days", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    """Run the simulation grid and write one CSV row per configuration."""
    args = parse_args()
    launches = load_launches(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out.with_name("dataset_summary.json")
    summary_path.write_text(json.dumps(dataset_summary(launches), indent=2, default=str))

    users = eligible_users(launches, min_days=args.min_days, min_apps=args.n_candidates)
    if args.users:
        users = users[: args.users]
    print(f"{len(users)} eligible users")

    rows = []
    for i, user in enumerate(users, 1):
        data = build_user_data(
            launches, user, n_candidates=args.n_candidates, period_minutes=args.period_minutes
        )
        start = time.perf_counter()
        for lam in args.lams:
            for horizon in args.horizons:
                for name in args.solvers:
                    if name == "greedy" and horizon != args.horizons[0]:
                        continue  # greedy ignores the look-ahead window
                    res = simulate(data, SOLVERS[name], k=args.k, lam=lam, horizon=horizon)
                    rows.append(
                        {
                            "user_id": user,
                            "solver": name,
                            "lam": lam,
                            "horizon": horizon,
                            "k": args.k,
                            "n": args.n_candidates,
                            "test_days": data.test_counts.shape[0],
                            **simulation_row(res),
                        }
                    )
        print(f"[{i}/{len(users)}] user {user}: {time.perf_counter() - start:.1f}s")
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
