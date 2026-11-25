import numpy as np
import pytest

from stableui.problem import Instance, count_switches, total_utility
from stableui.solvers import (
    greedy_step,
    solve_branch_bound,
    solve_brute_force,
    solve_cpsat,
    solve_dp,
    solve_greedy,
)

EXACT = [solve_dp, solve_branch_bound, solve_cpsat]


def random_instance(seed, t_len=3, n=5, k=2, lam=0.5, with_initial=False):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 10, size=(t_len, n)) / 4.0  # exactly representable for CP-SAT
    initial = tuple(sorted(rng.choice(n, size=k, replace=False))) if with_initial else None
    return Instance(values=values, k=k, lam=lam, initial=initial)


def assert_valid(inst, sol):
    assert len(sol.plan) == inst.horizon
    for layout in sol.plan:
        assert len(layout) == inst.k == len(set(layout))
        assert list(layout) == sorted(layout)
        assert all(0 <= a < inst.n_apps for a in layout)
    assert sol.utility == pytest.approx(total_utility(inst, sol.plan))


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("with_initial", [False, True])
def test_exact_solvers_match_brute_force(seed, with_initial):
    lam = [0.0, 0.5, 1.5, 4.0][seed % 4]
    inst = random_instance(seed, lam=lam, with_initial=with_initial)
    oracle = solve_brute_force(inst)
    for solver in EXACT:
        sol = solver(inst)
        assert_valid(inst, sol)
        assert sol.utility == pytest.approx(oracle.utility), solver.__name__


@pytest.mark.parametrize("seed", range(10))
def test_greedy_never_beats_optimum(seed):
    inst = random_instance(seed, t_len=4, n=6, k=3, lam=1.0)
    greedy = solve_greedy(inst)
    assert_valid(inst, greedy)
    assert greedy.utility <= solve_dp(inst).utility + 1e-9


def test_greedy_step_is_one_step_optimal():
    inst = random_instance(3, t_len=1, n=6, k=2, lam=1.0, with_initial=True)
    assert solve_greedy(inst).utility == pytest.approx(solve_brute_force(inst).utility)


def test_greedy_step_prefers_keeping_previous_on_ties():
    values = np.array([1.0, 1.0, 1.0, 1.0])
    assert greedy_step(values, k=2, lam=0.0, prev=(2, 3)) == (2, 3)


def test_zero_penalty_picks_top_k_each_period():
    values = np.array([[5, 1, 3, 0], [0, 4, 1, 6]], dtype=float)
    inst = Instance(values=values, k=2, lam=0.0)
    for solver in [solve_greedy, *EXACT]:
        assert solver(inst).plan == [(0, 2), (1, 3)]


def test_large_penalty_freezes_layout():
    rng = np.random.default_rng(0)
    inst = Instance(values=rng.random((5, 6)), k=2, lam=100.0)
    for solver in EXACT:
        assert count_switches(solver(inst).plan) == 0


def test_lookahead_beats_greedy_on_crafted_instance():
    # Greedy switches to app 1 for a tiny gain at t=0 and must switch back.
    values = np.array([[1.0, 1.2], [1.0, 0.0], [1.0, 0.0]])
    # Switching gains 0.2 > lam now, but costs 2 * lam = 0.3 over the horizon.
    inst = Instance(values=values, k=1, lam=0.15, initial=(0,))
    greedy, optimal = solve_greedy(inst), solve_dp(inst)
    assert greedy.plan == [(1,), (0,), (0,)]
    assert optimal.plan == [(0,), (0,), (0,)]
    assert optimal.utility > greedy.utility


def test_dp_chunked_mode_matches_dense():
    inst = random_instance(7, t_len=4, n=7, k=3, lam=0.75)
    dense = solve_dp(inst)
    chunked = solve_dp(inst, max_dense_entries=50)
    assert not chunked.stats["dense"]
    assert chunked.utility == pytest.approx(dense.utility)


def test_branch_bound_node_limit_returns_feasible_plan():
    inst = random_instance(1, t_len=5, n=7, k=3, lam=0.3)
    sol = solve_branch_bound(inst, node_limit=3)
    assert_valid(inst, sol)
    assert sol.utility >= solve_greedy(inst).utility - 1e-9


def test_branch_bound_prunes():
    inst = random_instance(2, t_len=4, n=6, k=2, lam=0.5)
    stats = solve_branch_bound(inst).stats
    assert stats["nodes_pruned"] > 0
    assert stats["nodes_expanded"] < stats["n_layouts"] ** inst.horizon


def test_k_equals_n():
    inst = Instance(values=np.ones((3, 3)), k=3, lam=1.0)
    for solver in [solve_greedy, *EXACT]:
        assert solver(inst).plan == [(0, 1, 2)] * 3


def test_branch_bound_time_limit_returns_feasible_plan():
    inst = random_instance(4, t_len=8, n=8, k=3, lam=0.2)
    sol = solve_branch_bound(inst, time_limit=0.0)
    assert_valid(inst, sol)
    assert not sol.stats["optimal"]
