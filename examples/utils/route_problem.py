"""Route problem setup and visualisation utilities for the ACO notebook example.

Provides the grid graph used in the route optimisation demo:
- 11 x 4 intermediate nodes on a regular 10m grid
- start node at (0, 30), goal node at (120, 30)
- directed edges connecting each node only to the next column

Functions
---------
build_grid()                    -- build node coordinates and edge list
generate_slopes_distances()     -- Gaussian edge slopes + Euclidean distances
plot_route()                    -- render grid + best route as a base64 PNG
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Fixed problem constants
# ---------------------------------------------------------------------------

GRID_SIZE    = 10
GRID_X_RANGE = np.arange(10, 120, GRID_SIZE)   # 10, 20, ..., 110
GRID_Y_RANGE = np.arange(10, 50,  GRID_SIZE)   # 10, 20, 30, 40
START_COORDS = [0,   30]
GOAL_COORDS  = [120, 30]
SCORES_SEED  = 42


def build_grid() -> tuple:
    """Build the grid of node coordinates and the directed edge list.

    Returns
    -------
    grid_coords : np.ndarray  shape (N, 2)
    X, Y        : meshgrid arrays (for plotting)
    idx_start   : int
    idx_goal    : int
    edges       : list[np.ndarray]  — edges[i] = reachable node indices from i
    """
    grid_y, grid_x = np.meshgrid(GRID_Y_RANGE, GRID_X_RANGE)
    X = grid_x
    Y = grid_y
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Prepend start, append goal
    grid_coords = np.concatenate(
        [np.array([START_COORDS]), grid_coords, np.array([GOAL_COORDS])],
        axis=0,
    )

    idx_start = int(np.where((grid_coords == START_COORDS).all(axis=1))[0][0])
    idx_goal  = int(np.where((grid_coords == GOAL_COORDS ).all(axis=1))[0][0])

    # Directed edges: each node connects to nodes one column to the right
    num_coords = len(grid_coords)
    edges = []
    for idx in range(num_coords):
        x, _y = grid_coords[idx]
        reachable = np.where(
            (grid_coords[:, 0] > x) & (grid_coords[:, 0] <= x + GRID_SIZE)
        )[0]
        edges.append(reachable)

    return grid_coords, X, Y, idx_start, idx_goal, edges


def generate_slopes_distances(
    grid_coords: np.ndarray,
    edges: list,
    seed: int = SCORES_SEED,
) -> tuple[dict, dict]:
    """Generate Gaussian slope values and Euclidean distances for every edge.

    Slopes are drawn from N(0, 0.3) and clipped to [-1, 1], where -1 means
    90 degrees downhill and +1 means 90 degrees uphill.  The distribution is
    centred at 0 (mostly flat terrain) with occasional steep sections.

    When computing a route's slope score, use the *absolute* value of each
    edge slope — negative slope is still terrain effort, not a reward.

    Returns
    -------
    slopes    : dict {(i, j): float}  — slope in [-1, 1]
    distances : dict {(i, j): float}  — Euclidean distance in metres
    """
    rng = np.random.default_rng(seed=seed)

    slopes:    dict = {}
    distances: dict = {}

    for i in range(len(grid_coords)):
        x_i, y_i = grid_coords[i]
        for j in edges[i]:
            x_j, y_j = grid_coords[j]
            slopes[(i, j)]    = float(np.clip(rng.normal(0.0, 0.3), -1.0, 1.0))
            distances[(i, j)] = float(np.sqrt((y_j - y_i) ** 2 + (x_j - x_i) ** 2))

    return slopes, distances


def plot_route(
    grid_coords: np.ndarray,
    edge_attr: dict,
    X: np.ndarray,
    Y: np.ndarray,
    path: list[int],
    title: str = "ACO Best Route",
) -> str:
    """Render the grid and best route as a base64-encoded PNG string.

    Edge line width encodes the absolute value of *edge_attr* (wider = larger
    absolute value = more terrain effort).  Works with slope values (floats
    in [-1, 1]) or any other numeric edge attribute.

    Path is drawn in navy blue; start in green; goal in red.

    Returns
    -------
    str — base64-encoded PNG
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    abs_vals = {e: abs(v) for e, v in edge_attr.items()}
    max_abs  = max(abs_vals.values()) if abs_vals else 1.0
    min_width, max_width = 0.5, 5.0

    for (i, j), abs_val in abs_vals.items():
        x1, y1 = grid_coords[i]
        x2, y2 = grid_coords[j]
        width = min_width + (abs_val / max_abs) * (max_width - min_width)
        ax.plot([x1, x2], [y1, y2], color="#c1c9d0", linewidth=width, alpha=0.6)

    ax.plot(X.ravel(), Y.ravel(), "o", c="#c1c9d0", markersize=4, zorder=2)

    if path:
        path_x = [grid_coords[p][0] for p in path]
        path_y = [grid_coords[p][1] for p in path]
        ax.plot(path_x, path_y, color="#11294e", linewidth=3, zorder=3, label="Route")

    ax.plot(START_COORDS[0], START_COORDS[1], "o", c="#2e7d32", markersize=10,
            zorder=4, label="Start")
    ax.plot(GOAL_COORDS[0],  GOAL_COORDS[1],  "o", c="#b71c1c", markersize=10,
            zorder=4, label="Goal")

    ax.set_title(title, fontsize=11, fontweight="bold", color="#11294e")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_facecolor("#f7f9fb")
    fig.patch.set_facecolor("#f7f9fb")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
