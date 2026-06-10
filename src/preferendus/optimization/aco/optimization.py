"""ACO optimization engine for graph-based multi-objective problems.

Mirrors the structure of the GA optimization module but replaces the
genetic algorithm loop with an Ant Colony Optimisation loop.

The key property (shared with the GA variant) is that solution quality is
evaluated *population-wide* each iteration via a_fine_aggregator — meaning
the same path can receive a different score in a different iteration depending
on the other paths it competes against.  This matches exactly the GA
evaluation principle used in the Preferendus decorator.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from preferendus.optimization.aco._aggregator import a_fine_aggregator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _construct_ant_path(
    edges: list,
    pheromones: dict,
    heuristics: dict,
    idx_start: int,
    idx_goal: int,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> Optional[list]:
    """Construct one ant's path from start to goal.

    Parameters
    ----------
    edges      : list of arrays; edges[i] = reachable node indices from node i
    pheromones : dict {(i, j): pheromone level}
    heuristics : dict {(i, j): heuristic attractiveness}
    idx_start  : start node index
    idx_goal   : goal node index
    alpha      : pheromone influence exponent
    beta       : heuristic influence exponent
    rng        : numpy random generator

    Returns
    -------
    list of node indices forming a complete path, or None if no path found
    """
    current = idx_start
    path = [current]
    visited = {current}

    while current != idx_goal:
        neighbours = [n for n in edges[current] if n not in visited]
        if not neighbours:
            return None

        raw_probs = np.array(
            [
                (pheromones.get((current, n), 1e-4) ** alpha)
                * (heuristics.get((current, n), 1e-4) ** beta)
                for n in neighbours
            ],
            dtype=float,
        )
        total = raw_probs.sum()
        probs = raw_probs / total if total > 0 else np.ones(len(raw_probs)) / len(raw_probs)

        next_node = int(rng.choice(neighbours, p=probs))
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


def _score_path(
    path: list,
    simulation_fn: Callable,
    preference_functions: Dict,
    stakeholder_weights: List[float],
    objective_weights: Dict,
    objective_total_weights: Dict,
    all_objectives: List[str],
) -> List[float]:
    """Run the simulation function on *path* and return per-objective preference scores.

    The preference scores are weighted sums across all stakeholders, normalised by
    the total weight per objective — exactly the same aggregation used in the GA.

    Parameters
    ----------
    path                   : list of node indices
    simulation_fn          : callable(path) -> (None, None, obj1, obj2, ...)
    preference_functions   : {stakeholder_id: {obj_name: callable}}
    stakeholder_weights    : list of normalised stakeholder weights
    objective_weights      : {stakeholder_id: {obj_name: weight}}
    objective_total_weights: {obj_name: total combined weight}
    all_objectives         : sorted list of objective names

    Returns
    -------
    list of float — one preference score per objective in all_objectives order
    """
    n_prefix = 2  # (env, objects) prefix — same convention as the GA
    sim_results = simulation_fn(path)
    objective_values = sim_results[n_prefix:]

    objective_dict = {
        name: objective_values[i]
        for i, name in enumerate(all_objectives)
        if i < len(objective_values)
    }

    objective_scores = dict.fromkeys(all_objectives, 0.0)
    for s_idx, s_id in enumerate(sorted(preference_functions.keys())):
        s_prefs = preference_functions[s_id]
        s_weight = stakeholder_weights[s_idx]
        obj_weights_s = objective_weights[s_id]

        for obj in all_objectives:
            if obj in s_prefs:
                pref_score = s_prefs[obj](objective_dict[obj])
                obj_weight = obj_weights_s.get(obj, 1.0)
                objective_scores[obj] += s_weight * obj_weight * pref_score

    for obj in all_objectives:
        if objective_total_weights[obj] > 0:
            objective_scores[obj] /= objective_total_weights[obj]

    return [objective_scores[obj] for obj in all_objectives]


# ---------------------------------------------------------------------------
# Public optimisation entry-point
# ---------------------------------------------------------------------------

def run_aco_optimization(
    simulation_fn: Callable,
    preference_functions: Dict,
    stakeholder_weights: List[float],
    objective_weights: Dict,
    objective_total_weights: Dict,
    all_objectives: List[str],
    weights_list: List[float],
    graph: Dict,
    options: Optional[Dict] = None,
) -> Optional[Dict]:
    """Run ACO optimisation for a graph-based multi-objective problem.

    The core evaluation loop follows the population-relative scoring principle:
    all ant paths (plus elite survivors) are scored together via
    ``a_fine_aggregator`` so that rankings are relative to the current
    population — not absolute.  This is the same principle used in the GA
    variant of Preferendus.

    Parameters
    ----------
    simulation_fn          : callable(path) -> (None, None, obj1, obj2, ...)
        Evaluates a complete path and returns objective values.  The first two
        return values are ignored (env / objects, following the GA convention).
    preference_functions   : {stakeholder_id: {obj_name: callable}}
    stakeholder_weights    : normalised stakeholder importance weights
    objective_weights      : {stakeholder_id: {obj_name: weight}}
    objective_total_weights: {obj_name: combined weight across stakeholders}
    all_objectives         : sorted list of objective names
    weights_list           : normalised per-objective weights for a_fine_aggregator
    graph : dict with keys
        - ``edges``          : list of arrays (edges[i] = neighbour node indices)
        - ``idx_start``      : start node index
        - ``idx_goal``       : goal node index
        - ``edge_scores``    : dict {(i,j): score value}   (used for heuristic)
        - ``edge_distances`` : dict {(i,j): distance value} (used for heuristic)
        - ``heuristics``     : dict {(i,j): heuristic value} (optional override)
    options : dict with ACO parameters
        - ``num_ants``       : int   (default 100)
        - ``num_iterations`` : int   (default 20)
        - ``alpha``          : float (default 1.0)  — pheromone influence
        - ``beta``           : float (default 0.75) — heuristic influence
        - ``rho``            : float (default 0.05) — evaporation rate
        - ``elite_size``     : int   (default 50)
        - ``tau_min``        : float (default 0.01)
        - ``tau_max``        : float (default 10.0)
        - ``seed``           : int or None (default None)

    Returns
    -------
    dict or None
        ``{'best_path': list, 'history': list, 'elite_solutions': list}``
        or None if no valid path was ever found.
    """
    if options is None:
        options = {}

    edges = graph["edges"]
    edge_scores = graph.get("edge_scores", {})
    edge_distances = graph.get("edge_distances", {})
    idx_start = graph["idx_start"]
    idx_goal = graph["idx_goal"]

    num_ants = options.get("num_ants", 100)
    num_iterations = options.get("num_iterations", 20)
    alpha = options.get("alpha", 1.0)
    beta = options.get("beta", 0.75)
    rho = options.get("rho", 0.05)
    elite_size = options.get("elite_size", 50)
    tau_min = options.get("tau_min", 0.01)
    tau_max = options.get("tau_max", 10.0)
    seed = options.get("seed", None)

    rng = np.random.default_rng(seed)

    all_edge_keys = set(edge_scores.keys()) | set(edge_distances.keys())
    pheromones = {e: 1.0 for e in all_edge_keys}

    # Build per-edge heuristics — can be overridden via graph['heuristics']
    heuristics: dict = graph.get("heuristics") or {}
    if not heuristics:
        w0 = weights_list[0] if len(weights_list) > 0 else 0.5
        w1 = weights_list[1] if len(weights_list) > 1 else 0.5
        for e in all_edge_keys:
            h = 1.0
            if e in edge_scores:
                h *= (1.0 / (edge_scores[e] + 1.0)) ** w0
            if e in edge_distances:
                h *= (1.0 / (edge_distances[e] + 1e-9)) ** w1
            heuristics[e] = max(h, 1e-4)

    # Elite pool carries over across iterations (simulates hall-of-fame)
    elite_solutions: list = []  # list of (path, pref_scores)
    best_path: Optional[list] = None
    best_afine: float = -float("inf")
    history: list = []

    for iteration in range(num_iterations):
        ant_paths: list = []

        for _ in range(num_ants):
            path = _construct_ant_path(
                edges, pheromones, heuristics,
                idx_start, idx_goal,
                alpha, beta, rng,
            )
            if path is not None:
                ant_paths.append(path)

        if not ant_paths:
            print(f"Iteration {iteration + 1}/{num_iterations}: no valid paths found")
            continue

        # Combine fresh ant paths with elite survivors (population-relative scoring)
        all_paths = ant_paths + [ep[0] for ep in elite_solutions]

        # Evaluate every path with the simulation function + preference functions
        all_pref_scores = [
            _score_path(
                path, simulation_fn,
                preference_functions, stakeholder_weights,
                objective_weights, objective_total_weights, all_objectives,
            )
            for path in all_paths
        ]

        # Population-wide aggregation via a_fine_aggregator
        # p_array shape: (n_objectives, n_solutions)
        if len(all_paths) >= 2:
            p_array = np.array(all_pref_scores).T
            afine_scores = a_fine_aggregator(weights_list, p_array)
        else:
            afine_scores = np.array([50.0] * len(all_paths))

        # Rank best to worst
        candidates = sorted(
            zip(afine_scores, all_paths, all_pref_scores),
            key=lambda x: -x[0],
        )

        elite_solutions = [(c[1], c[2]) for c in candidates[:elite_size]]
        best_path = candidates[0][1]
        best_afine = float(candidates[0][0])

        history.append({
            "iteration": iteration + 1,
            "best_afine": best_afine,
            "num_valid_paths": len(ant_paths),
        })

        print(
            f"Iteration {iteration + 1}/{num_iterations}  "
            f"a_fine={best_afine:.1f}  "
            f"valid_ants={len(ant_paths)}"
        )

        # Pheromone evaporation
        for e in pheromones:
            pheromones[e] *= (1.0 - rho)

        # Deposit pheromone proportional to afine score from elite solutions
        for afine_val, path, _ in candidates[:elite_size]:
            deposit = float(afine_val + 50.0) / 100.0  # range ~[0.5, 1.5]
            for step in range(len(path) - 1):
                e = (path[step], path[step + 1])
                if e in pheromones:
                    pheromones[e] += deposit

        # Clamp pheromone levels
        for e in pheromones:
            pheromones[e] = float(np.clip(pheromones[e], tau_min, tau_max))

    if best_path is None:
        return None

    return {
        "best_path": best_path,
        "history": history,
        "elite_solutions": elite_solutions,
    }
