"""State-of-the-art baseline: integer programming solved with OR-Tools CP-SAT.

CP-SAT (Google OR-Tools, Apache-2.0 licence) is a lazy-clause-generation
constraint/integer-programming solver built on a SAT engine; it has won the
MiniZinc Challenge every year since 2018. We only *use* it as a library; the
model below is our own.

Model (binary variables ``x[t, a]`` = app ``a`` shown in period ``t``):

    maximise   sum_{t,a} V[t, a] * x[t, a]  -  lambda * sum_{t,a} s[t, a]
    subject to sum_a x[t, a] = k                       for all t
               s[t, a] >= x[t, a] - x[t-1, a]           for all t, a
               x, s binary

``s[t, a] = 1`` marks an app *added* in period ``t``. Because every layout has
exactly ``k`` apps, additions equal removals, so ``sum_a s[t, a] = C_t``.

The model has ``O(T * n)`` variables and constraints -- linear in ``n``, in
contrast to the ``C(n, k)`` layout states used by DP and branch and bound.
General integer programming is NP-hard, so CP-SAT gives no polynomial
worst-case guarantee.

CP-SAT works with integer coefficients, so values are scaled by ``scale`` and
rounded; the returned utility is recomputed exactly in floating point.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from stableui.problem import Instance, Layout, Solution, total_utility


def solve_cpsat(
    inst: Instance,
    time_limit: float | None = 60.0,
    num_workers: int = 1,
    scale: int = 100_000,
) -> Solution:
    """Solve ``inst`` with the CP-SAT integer programming model.

    Args:
        inst: Problem instance.
        time_limit: Wall-clock limit in seconds (``None`` for no limit).
        num_workers: Number of CP-SAT search workers. ``1`` keeps the
            comparison with the single-threaded Python solvers fair.
        scale: Multiplier applied to values and penalty before rounding.

    Returns:
        The best solution found; ``stats["optimal"]`` reports whether CP-SAT
        proved optimality (up to coefficient rounding).
    """
    t_len, n = inst.values.shape
    model = cp_model.CpModel()
    x = [[model.new_bool_var(f"x_{t}_{a}") for a in range(n)] for t in range(t_len)]
    adds = []
    for t in range(t_len):
        model.add(sum(x[t]) == inst.k)
        for a in range(n):
            if t == 0 and inst.initial is None:
                continue
            prev = (1 if a in inst.initial else 0) if t == 0 else x[t - 1][a]
            s = model.new_bool_var(f"s_{t}_{a}")
            model.add(s >= x[t][a] - prev)
            adds.append(s)

    value_terms = [
        int(round(inst.values[t, a] * scale)) * x[t][a] for t in range(t_len) for a in range(n)
    ]
    penalty = int(round(inst.lam * scale))
    model.maximize(sum(value_terms) - penalty * sum(adds))

    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = num_workers
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed with status {solver.status_name(status)}")

    plan: list[Layout] = [
        tuple(a for a in range(n) if solver.boolean_value(x[t][a])) for t in range(t_len)
    ]
    return Solution(
        plan=plan,
        utility=total_utility(inst, plan),
        stats={
            "optimal": status == cp_model.OPTIMAL,
            "status": solver.status_name(status),
            "branches": solver.num_branches,
            "conflicts": solver.num_conflicts,
            "solver_wall_time": solver.wall_time,
        },
    )
