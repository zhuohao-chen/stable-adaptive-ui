"""Problem definition for the stable adaptive shortcut-selection problem.

An adaptive mobile UI shows ``k`` shortcut slots chosen from ``n`` candidate
apps. For each of ``T`` planning periods we pick a *layout* -- an unordered set
of ``k`` distinct apps. The per-period utility is

    U_t = V_t - lambda * C_t

where ``V_t`` is the sum of the predicted usage values of the displayed apps and
``C_t`` is the number of shortcuts replaced relative to the previous layout
(``C_t = k - |S_t ∩ S_{t-1}|``). The goal is to maximise ``sum_t U_t``.

Layouts are represented as sorted tuples of app indices so that they are
hashable and have a canonical form.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np

Layout = tuple[int, ...]


@dataclass(frozen=True)
class Instance:
    """A finite-horizon shortcut-planning instance.

    Attributes:
        values: Array of shape ``(T, n)``; ``values[t, a]`` is the predicted
            usage value of showing app ``a`` during period ``t``.
        k: Number of shortcut slots.
        lam: Switching penalty ``lambda`` charged per replaced shortcut.
        initial: Layout shown before period 0. ``None`` means no previous
            layout, so the first period incurs no switching cost.
    """

    values: np.ndarray
    k: int
    lam: float
    initial: Layout | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must be a 2-D array of shape (T, n)")
        object.__setattr__(self, "values", values)
        t_len, n = values.shape
        if t_len < 1:
            raise ValueError("planning horizon T must be at least 1")
        if not 1 <= self.k <= n:
            raise ValueError(f"k must satisfy 1 <= k <= n (got k={self.k}, n={n})")
        if self.lam < 0:
            raise ValueError("switching penalty lam must be non-negative")
        if self.initial is not None:
            object.__setattr__(self, "initial", validate_layout(self.initial, n, self.k))

    @property
    def horizon(self) -> int:
        """Number of planning periods ``T``."""
        return self.values.shape[0]

    @property
    def n_apps(self) -> int:
        """Number of candidate apps ``n``."""
        return self.values.shape[1]

    @property
    def n_layouts(self) -> int:
        """Number of feasible layouts ``L = C(n, k)``."""
        return comb(self.n_apps, self.k)


@dataclass
class Solution:
    """Output of a solver.

    Attributes:
        plan: One layout per planning period.
        utility: Cumulative utility ``sum_t U_t`` of ``plan``.
        stats: Solver-specific diagnostics (e.g. nodes explored, status).
    """

    plan: list[Layout]
    utility: float
    stats: dict = field(default_factory=dict)


def validate_layout(layout: Sequence[int], n: int, k: int) -> Layout:
    """Return ``layout`` in canonical form, raising if it is infeasible."""
    canon = tuple(sorted(int(a) for a in layout))
    if len(canon) != k or len(set(canon)) != k:
        raise ValueError(f"layout must contain exactly {k} distinct apps: {layout}")
    if canon and (canon[0] < 0 or canon[-1] >= n):
        raise ValueError(f"layout contains an app index outside [0, {n}): {layout}")
    return canon


def enumerate_layouts(n: int, k: int) -> list[Layout]:
    """List all ``C(n, k)`` layouts in lexicographic order."""
    return list(combinations(range(n), k))


def layout_matrix(layouts: Sequence[Layout], n: int) -> np.ndarray:
    """Return a boolean incidence matrix ``M`` with ``M[i, a] = a in layouts[i]``."""
    mat = np.zeros((len(layouts), n), dtype=bool)
    for i, layout in enumerate(layouts):
        mat[i, list(layout)] = True
    return mat


def switching_cost(prev: Layout | None, cur: Layout) -> int:
    """Number of shortcuts replaced when moving from ``prev`` to ``cur``."""
    if prev is None:
        return 0
    return len(cur) - len(set(prev) & set(cur))


def period_utility(inst: Instance, t: int, prev: Layout | None, cur: Layout) -> float:
    """Utility ``U_t = V_t - lambda * C_t`` of showing ``cur`` in period ``t``."""
    value = float(inst.values[t, list(cur)].sum())
    return value - inst.lam * switching_cost(prev, cur)


def total_utility(inst: Instance, plan: Sequence[Layout]) -> float:
    """Cumulative utility of a full plan, starting from ``inst.initial``."""
    if len(plan) != inst.horizon:
        raise ValueError(f"plan has {len(plan)} periods, expected {inst.horizon}")
    prev = inst.initial
    total = 0.0
    for t, layout in enumerate(plan):
        cur = validate_layout(layout, inst.n_apps, inst.k)
        total += period_utility(inst, t, prev, cur)
        prev = cur
    return total


def count_switches(plan: Sequence[Layout], initial: Layout | None = None) -> int:
    """Total number of shortcut replacements over a plan."""
    prev = initial
    total = 0
    for layout in plan:
        total += switching_cost(prev, layout)
        prev = layout
    return total
