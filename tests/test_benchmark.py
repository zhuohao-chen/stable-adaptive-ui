from stableui.benchmark import measure
from stableui.data import synthetic_instance
from stableui.solvers import solve_dp


def test_measure_reports_time_memory_and_utility():
    inst = synthetic_instance(horizon=3, n_apps=5, k=2, lam=0.1)
    row = measure(solve_dp, inst, repeats=2)
    assert row["time_median_s"] >= 0
    assert row["peak_python_mb"] >= 0
    assert row["optimal"]
    assert row["utility"] == solve_dp(inst).utility
