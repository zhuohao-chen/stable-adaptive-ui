"""Timing and memory measurement helpers.

* Time: ``time.perf_counter`` over ``repeats`` runs; the median is reported.
* Memory: peak Python-heap allocation during one extra run measured with
  ``tracemalloc``. NumPy buffers are tracked; memory allocated inside native
  libraries that bypass the Python allocator (notably OR-Tools' C++ core) is
  *not*, so CP-SAT memory is additionally sampled as the process RSS increase.
  The timed runs are executed without ``tracemalloc`` because tracing slows
  allocation-heavy code.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
import tracemalloc

try:  # Unix only; on Windows the RSS column is reported as NaN.
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

from stableui.problem import Instance, Solution
from stableui.solvers import Solver


def _max_rss_bytes() -> float:
    """Peak resident set size of this process in bytes (NaN where unavailable)."""
    if resource is None:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def measure(
    solver: Solver,
    inst: Instance,
    repeats: int = 3,
    track_memory: bool = True,
    budget: float | None = None,
) -> dict:
    """Run ``solver`` on ``inst`` and collect performance statistics.

    Args:
        solver: Solver function.
        inst: Problem instance.
        repeats: Number of timed runs.
        track_memory: Also run once under ``tracemalloc``.
        budget: If a run takes longer than this many seconds, skip the
            remaining repeats and the memory run.

    Returns:
        Dict with ``time_median_s``, ``time_min_s``, ``peak_python_mb``,
        ``rss_growth_mb``, ``utility``, ``optimal`` and solver stats.
    """
    times: list[float] = []
    sol: Solution | None = None
    rss_before = _max_rss_bytes()
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        sol = solver(inst)
        times.append(time.perf_counter() - start)
        if budget is not None and times[-1] > budget:
            track_memory = False
            break
    rss_growth = _max_rss_bytes() - rss_before
    rss_growth = max(0.0, rss_growth) if rss_growth == rss_growth else rss_growth

    peak = float("nan")
    if track_memory:
        gc.collect()
        tracemalloc.start()
        solver(inst)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = peak_bytes / 2**20

    row = {
        "time_median_s": statistics.median(times),
        "time_min_s": min(times),
        "peak_python_mb": peak,
        "rss_growth_mb": rss_growth / 2**20,
        "utility": sol.utility,
        "optimal": bool(sol.stats.get("optimal", False)),
    }
    for key in ("nodes_expanded", "nodes_pruned", "n_layouts", "status", "branches"):
        if key in sol.stats:
            row[key] = sol.stats[key]
    return row