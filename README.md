# Preferendus ACO

Multi-objective route optimisation using **Ant Colony Optimisation (ACO)** and the
**IMAP/Preferendus** preference-aggregation method.

---

## What is this?

The **Preferendus** method evaluates design alternatives by converting raw objective
values (distance, slope, cost, …) into *preference scores* via stakeholder-defined
preference functions, then aggregating those scores using a population-relative
z-score normalisation (`a_fine_aggregator`).  The key property: **the same solution
can receive a different score in a different iteration**, because scoring is relative
to the current population — not absolute.

This package provides a `PreferendusACO` decorator that wraps any graph-evaluation
function and optimises the path through the graph using ACO, following exactly the
same interface as a GA-based Preferendus decorator.  Any directed graph problem can
be solved by:

1. defining preference functions for each objective
2. passing a graph dict with edges and edge attributes
3. writing a simulation function that returns `(None, None, obj1, obj2, ...)`
4. decorating it with `@PreferendusACO(...)`

---

## Repository structure

```
IMAP-SI/
├── src/
│   └── preferendus/
│       └── optimization/
│           └── aco/
│               ├── __init__.py        # exposes PreferendusACO
│               ├── modo.py            # PreferendusACO decorator
│               ├── optimization.py    # ACO engine (run_aco_optimization)
│               └── _aggregator.py     # a_fine_aggregator (no external deps)
├── examples/
│   ├── route_optimisation_ACO.ipynb   # demo notebook
│   └── utils/
│       └── route_problem.py           # demo-specific grid helpers
└── pyproject.toml
```

---

## Installation

```bash
pip install -e .
```

Runtime dependency: `numpy`.

Optional extras:

```bash
# notebook / plotting examples
pip install -e .[examples]

# local development checks (tests)
pip install -e .[dev]
```

---

## Quick start

```python
from preferendus.optimization.aco.modo import PreferendusACO
from scipy.interpolate import PchipInterpolator
import numpy as np

# 1. Define preference functions
stakeholder_preferences = {
    1: {
        "1_distance": lambda d: float(np.clip(PchipInterpolator([min_d, max_d], [100, 0])(d), 0, 100)),
        "2_slope":    lambda s: float(np.clip(PchipInterpolator([min_s, max_s], [100, 0])(s), 0, 100)),
    }
}

objective_weights = {1: {"1_distance": 0.5, "2_slope": 0.5}}

# 2. Build the graph
graph = {
    "edges":          edges,          # list: edges[i] = array of neighbour indices
    "idx_start":      idx_start,
    "idx_goal":       idx_goal,
    "edge_scores":    edge_abs_slopes,   # heuristic attribute (lower = more attractive)
    "edge_distances": edge_distances,
}

# 3. Decorate the simulation function
@PreferendusACO(
    preference_functions=stakeholder_preferences,
    stakeholder_weights=[1.0],
    objective_weights=objective_weights,
    graph=graph,
    optimization_options={"num_ants": 200, "num_iterations": 20, "seed": 42},
)
def evaluate_route(path):
    total_distance  = sum(edge_distances[(path[i], path[i+1])] for i in range(len(path)-1))
    total_abs_slope = sum(abs(edge_slopes[(path[i], path[i+1])]) for i in range(len(path)-1))
    return None, None, total_distance, total_abs_slope

# 4. Run
result = evaluate_route()
print(result["x_opt"])    # best path (list of node indices)
print(result["utility"])  # final utility score (0–100)
```

---

## `@PreferendusACO` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `preference_functions` | `dict` | `{stakeholder_id: {obj_name: callable}}` — maps objective values to preference scores (0–100) |
| `stakeholder_weights` | `list[float]` | Relative importance of each stakeholder (normalised internally) |
| `objective_weights` | `dict` | `{stakeholder_id: {obj_name: float}}` — per-stakeholder objective weights; defaults to equal |
| `graph` | `dict` | Graph with keys `edges`, `idx_start`, `idx_goal`, `edge_scores`, `edge_distances` |
| `optimization_options` | `dict` | ACO algorithm settings (see below) |

### ACO options

| Key | Default | Meaning |
|-----|---------|---------|
| `num_ants` | 100 | Ant paths constructed per iteration |
| `num_iterations` | 20 | Number of iterations |
| `alpha` | 1.0 | Pheromone influence exponent |
| `beta` | 0.75 | Heuristic influence exponent |
| `rho` | 0.05 | Pheromone evaporation rate |
| `elite_size` | 50 | Best paths kept as hall-of-fame across iterations |
| `tau_min` | 0.01 | Minimum pheromone level (clamping) |
| `tau_max` | 10.0 | Maximum pheromone level (clamping) |
| `seed` | `None` | Random seed for reproducibility |

---

## Result dict

| Key | Type | Description |
|-----|------|-------------|
| `x_opt` | `list` | Best path found (node index sequence) |
| `utility` | `float` | Final utility score (0–100) |
| `simulation_results` | `tuple` | Raw return value of your simulation function |
| `history` | `list` | Per-iteration `{iteration, best_afine, num_valid_paths}` |
| `elite_solutions` | `list` | Final elite pool of `(path, pref_scores)` tuples |

---

## Demo notebook

`examples/route_optimisation_ACO.ipynb` demonstrates the full workflow on a
directed grid graph with two objectives: **route distance** and **absolute slope**.

Preference-curve bounds are derived automatically by two single-objective
Dijkstra calibration runs before the ACO is executed — no manual tuning needed.

---

## Preference curve calibration pattern

To set principled bounds for preference functions, run two Dijkstra searches:

```python
# Run 1: minimise |slope|
min_abs_slope, pure_slope_path = dijkstra(edge_abs_slopes, ...)
# Run 2: minimise distance
min_dist, pure_dist_path = dijkstra(edge_distances, ...)

slope_pref_best  = min_abs_slope                          # -> pref 100
slope_pref_worst = path_total(pure_dist_path, abs_slopes) * 1.2  # -> pref 0
dist_pref_best   = min_dist                               # -> pref 100
dist_pref_worst  = path_total(pure_slope_path, distances) * 1.2  # -> pref 0
```

This guarantees the ACO result always lies within the defined preference range.

---

## Development checks

```bash
# basic syntax validation
python -m compileall src examples/utils

# test suite
pytest
```

If you're using the workspace virtual environment on Windows:

```bash
.venv\Scripts\python.exe -m compileall src examples/utils
.venv\Scripts\python.exe -m pytest
```
