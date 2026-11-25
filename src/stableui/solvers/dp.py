"""Dynamic programming solver (Bellman recursion over layouts).

State: the layout ``S_t`` shown in period ``t``. Let ``F_t(j)`` be the best
cumulative utility of periods ``0..t`` that ends with layout ``j``. Then

    F_0(j) = val_0(j) - lambda * sw(initial, j)
    F_t(j) = val_t(j) + max_i [F_{t-1}(i) - lambda * sw(i, j)]

where ``val_t(j)`` is the summed value of layout ``j`` and ``sw(i, j) = k -
|i ∩ j|``. The optimum is ``max_j F_{T-1}(j)``; the plan is recovered from
back-pointers.

Complexity with ``L = C(n, k)`` layouts:

* time ``O(T * L^2)`` for the transitions plus ``O(L^2 * n)`` once to build
  the overlap matrix (or ``O(T * L^2 * n)`` in the low-memory mode);
* space ``O(L^2 + T * L)`` (overlap matrix + back-pointers), or
  ``O(chunk * L + T * L)`` in the low-memory mode.
"""

from __future__ import annotations

import numpy as np

from stableui.problem import (
    Instance,
    Layout,
    Solution,
    enumerate_layouts,
    layout_matrix,
    total_utility,
)

#: Above this many entries the full L x L overlap matrix is not materialised.
MAX_DENSE_ENTRIES = 25_000_000


def solve_dp(inst: Instance, max_dense_entries: int = MAX_DENSE_ENTRIES) -> Solution:
    """Solve ``inst`` exactly with dynamic programming.

    Args:
        inst: Problem instance.
        max_dense_entries: Threshold on ``L * L`` above which the overlap
            matrix is recomputed in column chunks instead of stored densely.

    Returns:
        An optimal solution.
    """
    k, lam = inst.k, inst.lam
    layouts = enumerate_layouts(inst.n_apps, k)
    n_layouts = len(layouts)
    member = layout_matrix(layouts, inst.n_apps)
    member_f = member.astype(np.float64)
    layout_values = inst.values @ member_f.T  # (T, L)

    dense = n_layouts * n_layouts <= max_dense_entries
    penalty = lam * (k - member_f @ member_f.T) if dense else None  # (L, L)
    chunk = max(1, max_dense_entries // max(n_layouts, 1))

    if inst.initial is None:
        best = layout_values[0].copy()
    else:
        init_vec = np.zeros(inst.n_apps)
        init_vec[list(inst.initial)] = 1.0
        best = layout_values[0] - lam * (k - member_f @ init_vec)

    back = np.zeros((inst.horizon, n_layouts), dtype=np.int32)
    for t in range(1, inst.horizon):
        new_best = np.empty(n_layouts)
        for start in range(0, n_layouts, n_layouts if dense else chunk):
            stop = min(start + (n_layouts if dense else chunk), n_layouts)
            if dense:
                block = penalty[:, start:stop]
            else:
                block = lam * (k - member_f @ member_f[start:stop].T)
            cand = best[:, None] - block  # (L_prev, stop - start)
            arg = np.argmax(cand, axis=0)
            back[t, start:stop] = arg
            new_best[start:stop] = cand[arg, np.arange(stop - start)]
        best = new_best + layout_values[t]

    idx = int(np.argmax(best))
    indices = [idx]
    for t in range(inst.horizon - 1, 0, -1):
        idx = int(back[t, idx])
        indices.append(idx)
    plan: list[Layout] = [layouts[i] for i in reversed(indices)]
    return Solution(
        plan=plan,
        utility=total_utility(inst, plan),
        stats={"optimal": True, "n_layouts": n_layouts, "dense": dense},
    )
