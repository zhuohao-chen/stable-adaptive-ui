"""Synthetic instances for scalability experiments.

Values mimic app-usage predictions: each app has a Zipf-distributed base
popularity and a daily cycle with a random phase, plus multiplicative noise.
Values are rounded to ``1e-4`` so that CP-SAT's integer scaling is exact.
"""

from __future__ import annotations

import numpy as np

from stableui.problem import Instance


def synthetic_instance(
    horizon: int,
    n_apps: int,
    k: int,
    lam: float,
    seed: int = 0,
    zipf_s: float = 1.0,
    periods_per_day: int = 24,
    noise: float = 0.3,
) -> Instance:
    """Generate a random planning instance.

    Args:
        horizon: Number of planning periods ``T``.
        n_apps: Number of candidate apps ``n``.
        k: Number of shortcut slots.
        lam: Switching penalty.
        seed: Random seed.
        zipf_s: Exponent of the Zipf popularity law.
        periods_per_day: Length of the daily usage cycle in periods.
        noise: Standard deviation of the log-normal noise.

    Returns:
        An :class:`~stableui.problem.Instance` with no initial layout.
    """
    rng = np.random.default_rng(seed)
    base = 1.0 / np.arange(1, n_apps + 1) ** zipf_s
    rng.shuffle(base)
    phase = rng.uniform(0, 2 * np.pi, size=n_apps)
    t = np.arange(horizon)[:, None]
    cycle = 1.0 + 0.8 * np.sin(2 * np.pi * t / periods_per_day + phase)
    values = base * cycle * rng.lognormal(0.0, noise, size=(horizon, n_apps))
    return Instance(values=np.round(values, 4), k=k, lam=lam)
