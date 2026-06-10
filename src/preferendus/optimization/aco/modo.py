"""PreferendusACO — decorator for graph-based multi-objective optimisation.

Drop-in counterpart to a GA-based Preferendus decorator, but uses Ant Colony
Optimisation (ACO) instead of a Genetic Algorithm.

Usage
-----
::

    from preferendus.optimization.aco.modo import PreferendusACO

    stakeholder_preferences = {
        1: {
            "1_distance": lambda d: ...,   # preference function for distance
            "2_slope":    lambda s: ...,   # preference function for slope
        },
    }

    objective_weights = {
        1: {"1_distance": 0.5, "2_slope": 0.5},
    }

    # graph data obtained from your problem setup
    graph = {
        "edges":          edges,        # list: edges[i] = array of neighbour indices
        "idx_start":      idx_start,    # int: start node
        "idx_goal":       idx_goal,     # int: goal node
        "edge_scores":    scores,       # dict {(i,j): float} — heuristic attribute 1
        "edge_distances": distances,    # dict {(i,j): float} — heuristic attribute 2
    }

    @PreferendusACO(
        preference_functions=stakeholder_preferences,
        stakeholder_weights=[1.0],
        objective_weights=objective_weights,
        graph=graph,
        optimization_options={"num_ants": 200, "num_iterations": 20},
    )
    def evaluate_path(path):
        total_distance = sum(distances[(path[i], path[i+1])] for i in range(len(path)-1))
        total_slope    = sum(abs(slopes[(path[i], path[i+1])]) for i in range(len(path)-1))
        return None, None, total_distance, total_slope

    result = evaluate_path()
    print(result["x_opt"])     # best path (list of node indices)
    print(result["utility"])   # final utility score
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

from preferendus.optimization.aco.optimization import run_aco_optimization


class PreferendusACO:
    """Optimisation Engine decorator for graph-based simulations using ACO.

    This decorator wraps a *path evaluation* function and optimises the choice
    of path through a graph by maximising stakeholder utility across multiple
    objectives.

    The key property (shared with the GA variant) is that preference scores
    are **population-relative**: the same path may receive a different score in
    a different iteration because ``a_fine_aggregator`` normalises scores
    within the current population.

    Parameters
    ----------
    preference_functions : Dict[int, Dict[str, Callable]]
        Nested dict mapping stakeholder IDs to objective preference functions.
        Each callable accepts a single numeric objective value and returns a
        numeric preference score (higher = better).

    stakeholder_weights : List[float]
        Relative importance of each stakeholder (normalised internally).

    objective_weights : Dict[int, Dict[str, float]], optional
        Per-stakeholder objective weights.  Defaults to equal weights.

    graph : Dict
        Graph data required by the ACO solver:

        - ``edges``          : list — ``edges[i]`` = array of neighbour indices
        - ``idx_start``      : int  — start node
        - ``idx_goal``       : int  — goal node
        - ``edge_scores``    : dict {(i,j): float} — heuristic attribute 1
        - ``edge_distances`` : dict {(i,j): float} — heuristic attribute 2
        - ``heuristics``     : dict {(i,j): float} — optional pre-computed
          heuristics (overrides automatic computation)

    optimization_options : Dict, optional
        ACO algorithm parameters:

        - ``num_ants``       : int   (default 100)
        - ``num_iterations`` : int   (default 20)
        - ``alpha``          : float (default 1.0)  — pheromone influence
        - ``beta``           : float (default 0.75) — heuristic influence
        - ``rho``            : float (default 0.05) — evaporation rate
        - ``elite_size``     : int   (default 50)
        - ``tau_min``        : float (default 0.01)
        - ``tau_max``        : float (default 10.0)
        - ``seed``           : int   (default None)
        - ``suppress_warnings`` : bool (default True)
        - ``suppress_logging``  : bool (default True)

    Returns
    -------
    Callable
        A decorated function that, when called (no arguments required), runs
        the ACO optimisation and returns a result dict:

        - ``x_opt``             : list  — best path (node indices)
        - ``utility``           : float — final utility score
        - ``simulation_results``: tuple — raw output of ``simulation_fn(best_path)``
        - ``history``           : list  — per-iteration metrics
        - ``elite_solutions``   : list  — final elite pool
    """

    def __init__(
        self,
        preference_functions: Dict[int, Dict[str, Callable]],
        stakeholder_weights: List[float],
        graph: Dict,
        objective_weights: Optional[Dict[int, Dict[str, float]]] = None,
        optimization_options: Optional[Dict] = None,
    ):
        if graph is None:
            raise ValueError("graph must be provided.")
        if "edges" not in graph or "idx_start" not in graph or "idx_goal" not in graph:
            raise ValueError("graph must contain 'edges', 'idx_start', and 'idx_goal'.")

        self.preference_functions = preference_functions

        total_weight = sum(stakeholder_weights)
        self.stakeholder_weights = [w / total_weight for w in stakeholder_weights]

        # Objective weights — default to equal per stakeholder
        self.objective_weights: Dict = {}
        if objective_weights is None:
            for s_id in self.preference_functions:
                objectives = list(self.preference_functions[s_id].keys())
                w = 1.0 / len(objectives)
                self.objective_weights[s_id] = dict.fromkeys(objectives, w)
        else:
            for s_id, weights in objective_weights.items():
                total = sum(weights.values())
                self.objective_weights[s_id] = {obj: w / total for obj, w in weights.items()}

        self.graph = graph

        self.optimization_options: Dict = {
            "num_ants": 100,
            "num_iterations": 20,
            "alpha": 1.0,
            "beta": 0.75,
            "rho": 0.05,
            "elite_size": 50,
            "tau_min": 0.01,
            "tau_max": 10.0,
            "seed": None,
            "suppress_warnings": True,
            "suppress_logging": True,
        }
        if optimization_options:
            self.optimization_options.update(optimization_options)

    def __call__(self, simulation_fn: Callable) -> Callable:
        """Wrap *simulation_fn* so that calling it triggers ACO optimisation.

        Parameters
        ----------
        simulation_fn : Callable
            Function with signature ``(path: list) -> tuple``.
            ``path`` is a list of node indices.  The return value must follow
            the same convention as the GA simulation function::

                return None, None, objective_1, objective_2, ...

            where the objectives correspond to the keys in
            ``preference_functions`` (sorted alphabetically).

        Returns
        -------
        Callable
            Wrapped function; call with no arguments to run the optimisation.
        """

        @functools.wraps(simulation_fn)
        def wrapper(*args, **kwargs):
            suppress_warnings = self.optimization_options.get("suppress_warnings", True)
            suppress_logging = self.optimization_options.get("suppress_logging", True)

            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.simplefilter("ignore")

                root_logger = logging.getLogger()
                old_level = root_logger.level
                if suppress_logging:
                    root_logger.setLevel(logging.ERROR)

                try:
                    all_objectives = sorted(
                        {obj for prefs in self.preference_functions.values() for obj in prefs}
                    )

                    objective_total_weights: Dict = dict.fromkeys(all_objectives, 0.0)
                    for s_idx, s_id in enumerate(sorted(self.preference_functions.keys())):
                        s_weight = self.stakeholder_weights[s_idx]
                        obj_weights_s = self.objective_weights[s_id]
                        for obj in all_objectives:
                            if obj in self.preference_functions[s_id]:
                                objective_total_weights[obj] += (
                                    s_weight * obj_weights_s.get(obj, 1.0)
                                )

                    total_w = sum(objective_total_weights.values())
                    norm_objective_weights = {
                        obj: w / total_w for obj, w in objective_total_weights.items()
                    }
                    weights_list = [norm_objective_weights[obj] for obj in all_objectives]

                    result = run_aco_optimization(
                        simulation_fn=simulation_fn,
                        preference_functions=self.preference_functions,
                        stakeholder_weights=self.stakeholder_weights,
                        objective_weights=self.objective_weights,
                        objective_total_weights=objective_total_weights,
                        all_objectives=all_objectives,
                        weights_list=weights_list,
                        graph=self.graph,
                        options=self.optimization_options,
                    )

                    if result is None:
                        print("ACO optimisation found no valid solution.")
                        return None

                    best_path = result["best_path"]
                    final_results = simulation_fn(best_path)

                    # Compute final score (same formula as GA decorator)
                    n_prefix = 2
                    objective_values = final_results[n_prefix:]
                    objective_dict = {
                        name: objective_values[i]
                        for i, name in enumerate(all_objectives)
                        if i < len(objective_values)
                    }

                    final_scores = []
                    for obj in all_objectives:
                        obj_score = 0.0
                        obj_weight_total = 0.0
                        for s_idx, s_id in enumerate(sorted(self.preference_functions.keys())):
                            if obj in self.preference_functions[s_id]:
                                s_weight = self.stakeholder_weights[s_idx]
                                obj_weight = self.objective_weights[s_id].get(obj, 1.0)
                                pref_score = self.preference_functions[s_id][obj](
                                    objective_dict[obj]
                                )
                                obj_score += s_weight * obj_weight * pref_score
                                obj_weight_total += s_weight * obj_weight
                        if obj_weight_total > 0:
                            final_scores.append(obj_score / obj_weight_total)

                    final_utility = sum(w * s for w, s in zip(weights_list, final_scores))

                    print(f"Final objective values : {objective_dict}")
                    print(f"Final preference scores: {final_scores}")
                    print(f"Final weights          : {weights_list}")
                    print(f"Final utility          : {final_utility}")
                    print(f"Best path length       : {len(best_path)} nodes")

                    return {
                        "x_opt": best_path,
                        "utility": final_utility,
                        "simulation_results": final_results,
                        "history": result.get("history", []),
                        "elite_solutions": result.get("elite_solutions", []),
                    }

                finally:
                    if suppress_logging:
                        root_logger.setLevel(old_level)

        return wrapper
