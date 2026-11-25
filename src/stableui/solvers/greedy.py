"""Greedy (myopic) solver.

At every period the greedy policy picks the layout with the highest *immediate*
utility given the layout chosen in the previous period, ignoring future
periods.

Because ``U_t = sum_{a in S} V[t, a] - lambda * (k - |S ∩ S_prev|)``, the
immediate utility decomposes over apps:

    U_t = sum_{a in S} (V[t, a] + lambda * [a in S_prev]) - lambda * k

so the one-step optimum is simply the top-``k`` apps by the adjusted score
``V[t, a] + lambda * [a in S_prev]``. No layout enumeration is needed.

Complexity: ``O(T * n log n)`` time (a sort per period; ``O(T * n)`` with
selection) and ``O(n + T * k)`` space.
"""

from __future__ import annotations

import numpy as np

from stableui.problem import Instance, Layout, Solution, total_utility


def greedy_step(values_t: np.ndarray, k: int, lam: float, prev: Layout | None) -> Layout:
    """Return the immediate-utility-maximising layout for a single period.

    Ties are broken towards keeping the previous layout and then towards the
    lower app index, which makes the result deterministic.

    Args:
        values_t: Predicted values for one period, shape ``(n,)``.
        k: Number of shortcut slots.
        lam: Switching penalty.
        prev: Layout shown in the previous period, or ``None``.

    Returns:
        The selected layout as a sorted tuple.
    """
    scores = np.array(values_t, dtype=float)
    kept = np.zeros(scores.shape[0], dtype=bool)
    if prev is not None:
        kept[list(prev)] = True
        scores = scores + lam * kept
    # lexsort sorts by the last key first: score (desc), then kept (desc), then index (asc).
    order = np.lexsort((np.arange(scores.shape[0]), ~kept, -scores))
    return tuple(sorted(int(a) for a in order[:k]))


def solve_greedy(inst: Instance) -> Solution:
    """Solve ``inst`` with the myopic greedy policy."""
    plan: list[Layout] = []
    prev = inst.initial
    for t in range(inst.horizon):
        cur = greedy_step(inst.values[t], inst.k, inst.lam, prev)
        plan.append(cur)
        prev = cur
    return Solution(plan=plan, utility=total_utility(inst, plan), stats={"optimal": False})
