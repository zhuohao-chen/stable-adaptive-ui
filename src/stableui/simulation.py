"""Receding-horizon simulation of an adaptive shortcut bar on held-out data.

At every test period the planner receives predictions for the next
``horizon`` periods, solves the finite-horizon problem, *executes only the
first layout*, and moves on (model predictive control). The executed layout is
scored against the user's actual launches in that period.

With ``online_update=True`` the frequency predictor is refreshed at the end of
each test day with that day's observed launches.

Tie-breaking: frequency predictions contain many exact ties (e.g. zero usage
at night), so different exact solvers may return *different* optimal plans and
the simulated trajectories would diverge for reasons unrelated to the
algorithms. With ``tie_break=True`` predictions are rounded to ``1e-3`` and a
bias of ``1e-5 * (n - 1 - a)`` is added for app ``a`` (apps are ordered by
training frequency), which makes the optimum unique in practice so that all
exact solvers execute identical decisions.

Normalisation: users differ by orders of magnitude in launch volume, so with
``normalize=True`` all predictions (and hits) are divided by the mean training
prediction ``mean_{p,a} V[p, a]``. ``lambda = 1`` then means "one switch costs
as much as the value of an average candidate app for one period", and the same
``lambda`` grid is meaningful for every user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from stableui.data.lsapp import UserData
from stableui.problem import Instance, Layout, switching_cost
from stableui.solvers import Solver


@dataclass
class SimulationResult:
    """Aggregate metrics of one simulation run.

    Attributes:
        hits: Actual launches of apps that were displayed.
        launches: All actual launches in the test periods (any app).
        switches: Total shortcut replacements executed.
        predicted_value: Sum of predicted values of executed layouts.
        planning_utility: ``predicted_value - lam * switches``.
        solve_seconds: Total solver wall-clock time.
        n_decisions: Number of executed decisions (test periods).
        test_days: Number of simulated test days.
        layouts: Executed layout per test period.
        lam: Switching penalty used in the run.
        scale: Normalisation factor applied to values and hits.
    """

    lam: float = 0.0
    scale: float = 1.0
    hits: float = 0.0
    launches: float = 0.0
    switches: int = 0
    predicted_value: float = 0.0
    planning_utility: float = 0.0
    solve_seconds: float = 0.0
    n_decisions: int = 0
    test_days: int = 0
    layouts: list[Layout] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of actual launches covered by the shortcut bar."""
        return self.hits / self.launches if self.launches else 0.0

    @property
    def realized_utility(self) -> float:
        """Normalised actual hits minus the switching penalty."""
        return self.scale * self.hits - self.lam * self.switches


def simulate(
    data: UserData,
    solver: Solver,
    k: int,
    lam: float,
    horizon: int,
    online_update: bool = True,
    tie_break: bool = True,
    normalize: bool = True,
) -> SimulationResult:
    """Run the receding-horizon simulation over all test periods of one user.

    Args:
        data: Prepared user data.
        solver: Any solver from :mod:`stableui.solvers`.
        k: Number of shortcut slots.
        lam: Switching penalty.
        horizon: Look-ahead window (number of periods planned each step).
        online_update: Refresh predictions after every test day.
        tie_break: Apply the deterministic tie-break described above.
        normalize: Divide values by the mean training prediction.

    Returns:
        Aggregated metrics.
    """
    periods = data.periods_per_day
    counts_sum = data.train_counts.sum(axis=0)
    n_days = data.train_counts.shape[0]
    n_apps = data.train_counts.shape[2]
    bias = 1e-5 * np.arange(n_apps - 1, -1, -1) if tie_break else np.zeros(n_apps)
    mean_value = float(data.values.mean())
    scale = 1.0 / mean_value if normalize and mean_value > 0 else 1.0
    result = SimulationResult(lam=lam, scale=scale)
    prev: Layout | None = None

    result.test_days = data.test_counts.shape[0]
    for day in range(result.test_days):
        values = scale * counts_sum / n_days
        for p in range(periods):
            window = values[[(p + h) % periods for h in range(horizon)]]
            if tie_break:
                window = np.round(window, 3) + bias
            inst = Instance(values=window, k=k, lam=lam, initial=prev)
            start = time.perf_counter()
            layout = solver(inst).plan[0]
            result.solve_seconds += time.perf_counter() - start

            switches = switching_cost(prev, layout)
            value = float(values[p, list(layout)].sum())
            result.switches += switches
            result.predicted_value += value
            result.planning_utility += value - lam * switches
            result.hits += float(data.test_counts[day, p, list(layout)].sum())
            result.launches += float(data.test_totals[day, p])
            result.n_decisions += 1
            result.layouts.append(layout)
            prev = layout
        if online_update:
            counts_sum = counts_sum + data.test_counts[day]
            n_days += 1

    return result


def simulation_row(result: SimulationResult) -> dict:
    """Flatten a :class:`SimulationResult` into a dict for a results table."""
    return {
        "coverage": result.coverage,
        "hits": result.hits,
        "launches": result.launches,
        "switches": result.switches,
        "switches_per_day": result.switches / max(result.test_days, 1),
        "predicted_value": result.predicted_value,
        "planning_utility": result.planning_utility,
        "realized_utility": result.realized_utility,
        "solve_ms_per_decision": 1000 * result.solve_seconds / max(result.n_decisions, 1),
        "n_decisions": result.n_decisions,
    }
