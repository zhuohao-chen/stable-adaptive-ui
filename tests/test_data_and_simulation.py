import numpy as np
import pandas as pd
import pytest

from stableui.data import build_user_data, eligible_users, launches_from_events, synthetic_instance
from stableui.simulation import simulate
from stableui.solvers import SOLVERS


def make_events():
    rows = []
    apps = ["A", "B", "C"]
    for day in range(10):
        for hour, app in [(8, "A"), (8, "A"), (12, "B"), (20, "C"), (20, "A")]:
            ts = pd.Timestamp("2018-01-01") + pd.Timedelta(days=day, hours=hour)
            if rows and rows[-1][2] == app and rows[-1][1] == ts:
                ts += pd.Timedelta(seconds=5)  # re-open within the merge window
            rows.append((0, ts, app, "Opened"))
            rows.append((0, ts, app, "Closed"))
    rows.append((1, pd.Timestamp("2018-01-01 09:00"), apps[0], "Opened"))
    return pd.DataFrame(rows, columns=["user_id", "timestamp", "app_name", "event_type"])


@pytest.fixture
def launches():
    return launches_from_events(make_events())


def test_reopens_are_merged(launches):
    user0 = launches[launches["user_id"] == 0]
    assert len(user0) == 10 * 4  # the two 8am opens of "A" become one launch
    assert set(launches.columns) == {"user_id", "timestamp", "app"}


def test_build_user_data_shapes_and_values(launches):
    data = build_user_data(launches, 0, n_candidates=3, period_minutes=60, train_frac=0.7)
    assert data.train_counts.shape == (7, 24, 3)
    assert data.test_counts.shape == (3, 24, 3)
    assert data.test_totals.sum() == 3 * 4
    a = data.apps.index("A")
    assert data.values[8, a] == pytest.approx(1.0)
    assert data.values[20, a] == pytest.approx(1.0)


def test_build_user_data_rejects_short_history(launches):
    with pytest.raises(ValueError):
        build_user_data(launches, 1, n_candidates=1)


def test_eligible_users(launches):
    assert eligible_users(launches, min_days=5, min_apps=3, min_launches=10) == [0]


def test_simulation_metrics(launches):
    data = build_user_data(launches, 0, n_candidates=3)
    for name, solver in SOLVERS.items():
        res = simulate(data, solver, k=2, lam=0.1, horizon=3)
        assert res.n_decisions == 3 * 24
        assert 0.0 <= res.coverage <= 1.0
        assert res.realized_utility == pytest.approx(res.scale * res.hits - 0.1 * res.switches), (
            name
        )


def test_exact_solvers_execute_identical_decisions(launches):
    data = build_user_data(launches, 0, n_candidates=3)
    layouts = {
        name: simulate(data, SOLVERS[name], k=2, lam=0.3, horizon=4).layouts
        for name in ("dp", "bnb", "cpsat")
    }
    assert layouts["dp"] == layouts["bnb"] == layouts["cpsat"]


def test_synthetic_instance_is_reproducible():
    a = synthetic_instance(horizon=5, n_apps=6, k=2, lam=0.1, seed=3)
    b = synthetic_instance(horizon=5, n_apps=6, k=2, lam=0.1, seed=3)
    assert np.array_equal(a.values, b.values)
    assert a.values.shape == (5, 6)
    assert (a.values >= 0).all()
