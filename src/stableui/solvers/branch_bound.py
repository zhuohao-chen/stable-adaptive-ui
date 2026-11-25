"""Depth-first branch and bound solver.

The search tree has one level per period; a node at depth ``t`` fixes the
layouts of periods ``0..t-1``. Children are the ``L = C(n, k)`` layouts for
period ``t``, explored in decreasing order of immediate utility (best-first
child ordering).

Upper bound for a node with accumulated utility ``acc`` at depth ``t``:

    UB = acc + sum_{s >= t} max_j val_s(j)

i.e. every remaining period shows its best layout with zero switching cost.
This bound is admissible because switching costs are non-negative. Since
children are sorted by immediate utility, once a child fails the bound test
all remaining siblings fail too, so the loop stops early.

The incumbent is initialised with the greedy solution, which typically gives a
strong lower bound from the start.

Complexity: worst case ``O(L^T)`` nodes, each costing ``O(L (n + log L))``
to score and sort its children; space ``O(T * L)`` for the recursion stack.
"""

from __future__ import annotations

import time

import numpy as np

from stableui.problem import (
    Instance,
    Layout,
    Solution,
    enumerate_layouts,
    layout_matrix,
    total_utility,
)
from stableui.solvers.greedy import solve_greedy

_EPS = 1e-9


class _Search:
    """Holds the state of one branch-and-bound run."""

    def __init__(self, inst: Instance, node_limit: int | None, time_limit: float | None) -> None:
        self.inst = inst
        self.layouts = enumerate_layouts(inst.n_apps, inst.k)
        self.member = layout_matrix(self.layouts, inst.n_apps).astype(np.float64)
        self.layout_values = inst.values @ self.member.T  # (T, L)
        best_per_period = self.layout_values.max(axis=1)
        # suffix[t] = sum of best layout values over periods t..T-1
        self.suffix = np.concatenate([np.cumsum(best_per_period[::-1])[::-1], [0.0]])
        self.node_limit = node_limit
        self.deadline = None if time_limit is None else time.perf_counter() + time_limit
        self.nodes = 0
        self.pruned = 0
        self.truncated = False

        greedy = solve_greedy(inst)
        self.best_utility = greedy.utility
        self.best_plan = [self.layouts.index(layout) for layout in greedy.plan]
        self.path: list[int] = []

    def _child_scores(self, t: int, prev_idx: int | None) -> np.ndarray:
        """Immediate utility of every layout in period ``t`` after ``prev_idx``."""
        vals = self.layout_values[t]
        if prev_idx is None:
            if self.inst.initial is None:
                return vals
            prev_vec = np.zeros(self.inst.n_apps)
            prev_vec[list(self.inst.initial)] = 1.0
        else:
            prev_vec = self.member[prev_idx]
        overlap = self.member @ prev_vec
        return vals - self.inst.lam * (self.inst.k - overlap)

    def _out_of_budget(self) -> bool:
        """Whether the node or time limit has been reached."""
        if self.truncated:
            return True
        if self.node_limit is not None and self.nodes >= self.node_limit:
            return True
        # Checking the clock is comparatively expensive, so do it every 1024 nodes.
        return (
            self.deadline is not None
            and self.nodes % 1024 == 0
            and time.perf_counter() > self.deadline
        )

    def run(self, t: int, prev_idx: int | None, acc: float) -> None:
        """Explore the subtree rooted at depth ``t``."""
        if t == self.inst.horizon:
            if acc > self.best_utility + _EPS:
                self.best_utility = acc
                self.best_plan = list(self.path)
            return
        scores = self._child_scores(t, prev_idx)
        order = np.argsort(-scores, kind="stable")
        for rank, j in enumerate(order):
            if self._out_of_budget():
                self.truncated = True
                return
            child_acc = acc + scores[j]
            if child_acc + self.suffix[t + 1] <= self.best_utility + _EPS:
                self.pruned += len(order) - rank
                break
            self.nodes += 1
            self.path.append(int(j))
            self.run(t + 1, int(j), child_acc)
            self.path.pop()


def solve_branch_bound(
    inst: Instance, node_limit: int | None = None, time_limit: float | None = None
) -> Solution:
    """Solve ``inst`` exactly with depth-first branch and bound.

    Args:
        inst: Problem instance.
        node_limit: Optional cap on the number of expanded nodes. If hit, the
            best plan found so far is returned and ``stats["optimal"]`` is
            ``False``.
        time_limit: Optional wall-clock limit in seconds, handled like
            ``node_limit``.

    Returns:
        The best solution found (optimal unless a limit was reached).
    """
    search = _Search(inst, node_limit, time_limit)
    search.run(0, None, 0.0)
    plan: list[Layout] = [search.layouts[i] for i in search.best_plan]
    return Solution(
        plan=plan,
        utility=total_utility(inst, plan),
        stats={
            "optimal": not search.truncated,
            "nodes_expanded": search.nodes,
            "nodes_pruned": search.pruned,
            "n_layouts": len(search.layouts),
        },
    )
