"""Standalone a_fine_aggregator — no external dependencies.

Aggregates multi-objective preference scores using z-score normalisation
and a weighted sum, then rescales to [0, 100].  Higher is better.

This is the core aggregation mechanism shared by all Preferendus optimisers
(both GA and ACO variants).
"""

from __future__ import annotations

from typing import Union

import numpy as np


def a_fine_aggregator(
    w: Union[list, np.ndarray],
    p: Union[list, np.ndarray],
) -> np.ndarray:
    """Aggregate multi-objective scores using z-score normalisation + weighted sum.

    Parameters
    ----------
    w : weights per objective (normalised to sum=1 internally)
    p : 2-D array, shape (n_objectives, n_solutions) — higher values = better

    Returns
    -------
    np.ndarray of shape (n_solutions,) scaled to [0, 100] — higher = better
    """
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if total < 1e-9:
        raise ValueError("All weights are zero.")
    w = w / total

    assert len(w) == len(p), (
        f"Number of weights ({len(w)}) != number of objectives ({len(p)})."
    )

    p_t = np.array(p, dtype=float).T  # (n_solutions, n_objectives)

    std = np.std(p_t, axis=0)
    std[std == 0] = 1e-6

    z = (p_t - np.mean(p_t, axis=0)) / std
    p_star = np.sum(w * z, axis=1)

    if len(np.unique(p_star)) == 1:
        return np.full(len(p_t), 50.0, dtype=float)

    return (p_star - p_star.min()) / (p_star.max() - p_star.min()) * 100.0
