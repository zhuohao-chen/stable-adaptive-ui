"""Exhaustive search over all ``L^T`` plans; used only as a test oracle."""

from __future__ import annotations

from itertools import product

from stableui.problem import Instance, Solution, enumerate_layouts, total_utility

#: Refuse to enumerate more plans than this.
MAX_PLANS = 2_000_000


def solve_brute_force(inst: Instance, max_plans: int = MAX_PLANS) -> Solution:
    """Return an optimal plan by enumerating every layout sequence.

    Raises:
        ValueError: If ``L^T`` exceeds ``max_plans``.
    """
    layouts = enumerate_layouts(inst.n_apps, inst.k)
    if len(layouts) ** inst.horizon > max_plans:
        raise ValueError("instance too large for brute force")
    best_plan, best_utility = None, float("-inf")
    for plan in product(layouts, repeat=inst.horizon):
        utility = total_utility(inst, plan)
        if utility > best_utility:
            best_plan, best_utility = list(plan), utility
    return Solution(plan=best_plan, utility=best_utility, stats={"optimal": True})
