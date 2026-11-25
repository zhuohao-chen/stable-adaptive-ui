# Benchmarking Optimization Algorithms for Stable Adaptive Mobile UIs

COSC 520 (UBC Okanagan) course project — Cisco Chen, Iverson Yuan.

An adaptive home screen shows `k` shortcut slots chosen from `n` candidate apps.
Each period it should show the apps the user is likely to open, but replacing
shortcuts too often makes the interface unpredictable. We plan layouts over `T`
periods to maximise

```
sum_t U_t,   U_t = V_t - λ·C_t
```

where `V_t` is the predicted usage value of the displayed apps and
`C_t = k − |S_t ∩ S_{t−1}|` is the number of replaced shortcuts. We compare four algorithms:

| Solver | Module | Optimal? | Time | Space |
|---|---|---|---|---|
| Greedy (myopic) | `solvers/greedy.py` | per period only | `O(T·n log n)` | `O(n + T·k)` |
| Dynamic programming | `solvers/dp.py` | yes | `O(T·L²)` | `O(L² + T·L)` |
| Branch & bound | `solvers/branch_bound.py` | yes | `O(L^T)` worst case | `O(T·L)` |
| CP-SAT integer program (SOTA) | `solvers/cpsat.py` | yes (proved by solver) | NP-hard in general | `O(T·n)` model |

with `L = C(n, k)` feasible layouts. A brute-force oracle (`solvers/brute_force.py`)
is used only in tests.

## Installation

Requires Python ≥ 3.10.

```bash
git clone <this repo> && cd stable-adaptive-ui
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Dataset

We use [LSApp](https://github.com/aliannejadi/LSApp) (Aliannejadi et al., ACM TOIS 2021).
It is not redistributed here. Download and extract it with:

```bash
python -c "from stableui.data import download_lsapp; download_lsapp('data/lsapp.tsv')"
```

Note that `lsapp.tsv.gz` in the LSApp repository is a gzipped *tar* archive. The helper
handles this; if you download it manually, use `tar -xzf lsapp.tsv.gz -C data/`.

## Usage

```bash
pytest                                   # unit tests (exact solvers are checked against brute force)
python scripts/run_real.py --users 30    # Experiment 1: receding-horizon simulation on LSApp
python scripts/run_scalability.py        # Experiment 2: time/memory on synthetic instances
python scripts/make_plots.py             # figures and tables -> figures/
```

`run_scalability.py --quick` runs a small grid in a few seconds. The full grid takes roughly
10–20 minutes; branch and bound and CP-SAT get a per-solve time limit (`--budget`).

Minimal API example:

```python
import numpy as np
from stableui import Instance
from stableui.solvers import SOLVERS

inst = Instance(values=np.random.rand(6, 10), k=3, lam=0.2)
for name, solve in SOLVERS.items():
    print(name, solve(inst).utility)
```

## Repository layout

```
src/stableui/
  problem.py          Instance, utility, switching cost, layout enumeration
  solvers/            greedy, dp, branch_bound, cpsat, brute_force
  data/lsapp.py       LSApp loading, launch de-duplication, train/test arrays
  data/synthetic.py   synthetic instances for scalability tests
  simulation.py       receding-horizon (MPC) simulation and metrics
  benchmark.py        timing (perf_counter) and memory (tracemalloc) helpers
scripts/              experiment and plotting entry points
tests/                pytest suite
```

## Experimental protocol and design notes

- **Launches.** Only `Opened` events are used. Re-opens of the same app within 60 s
  are merged into one launch, because the raw log often records several open/close
  pairs within seconds.
- **Prediction.** `V[p, a]` = mean launches of app `a` in hour-of-day `p` per training
  day (first 70% of each user's days). Predictions are refreshed after each test day.
  Values are normalised per user (divided by their mean), so `λ` means the same thing
  for every user.
- **Receding horizon.** At each test hour the solver plans `H` hours ahead and only the
  first layout is executed. With `H = 1` all exact solvers coincide with Greedy.
- **Tie-breaking.** Frequency predictions have many exact ties (e.g. zero usage at night).
  To keep different optimal solvers from executing different but equally optimal layouts,
  predictions are rounded to `1e-3` and a tiny deterministic bias favours more frequent
  apps. With this, DP, B&B and CP-SAT execute identical decisions (verified in the tests).

## Known quirks / threats to validity

- `tracemalloc` measures only Python-heap allocations (including NumPy). CP-SAT allocates
  memory inside the OR-Tools C++ core, so its reported Python peak underestimates its real
  memory; `rss_growth_mb` gives a coarse process-level complement.
- CP-SAT runs single-threaded (`num_workers=1`) for a fair comparison with the Python
  solvers; with more workers it is faster. Its integer coefficients are scaled by 1e5 and
  rounded.
- DP and B&B are written in Python/NumPy while CP-SAT is compiled C++, so constant
  factors are not comparable; compare growth rates rather than absolute times.
- Timings depend on the machine; report the hardware used.

## Third-party code and licences

- **OR-Tools** (Google, Apache License 2.0) is used as a library for the CP-SAT solver.
  We do not modify or redistribute it; the integer programming model is our own.
- **LSApp** dataset © its authors; cite Aliannejadi et al. (2021) when using it.
- All other code in this repository was written by the authors.

## Licence

MIT — see [LICENSE](LICENSE).
