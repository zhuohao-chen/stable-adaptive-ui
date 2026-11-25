"""Solvers for the stable shortcut-selection problem.

Every solver has the signature ``solve(inst: Instance) -> Solution``.
"""

from collections.abc import Callable

from stableui.problem import Instance, Solution
from stableui.solvers.branch_bound import solve_branch_bound
from stableui.solvers.brute_force import solve_brute_force
from stableui.solvers.cpsat import solve_cpsat
from stableui.solvers.dp import solve_dp
from stableui.solvers.greedy import greedy_step, solve_greedy

Solver = Callable[[Instance], Solution]

#: Solvers compared in the experiments, keyed by short name.
SOLVERS: dict[str, Solver] = {
    "greedy": solve_greedy,
    "dp": solve_dp,
    "bnb": solve_branch_bound,
    "cpsat": solve_cpsat,
}

__all__ = [
    "SOLVERS",
    "Solver",
    "greedy_step",
    "solve_branch_bound",
    "solve_brute_force",
    "solve_cpsat",
    "solve_dp",
    "solve_greedy",
]
