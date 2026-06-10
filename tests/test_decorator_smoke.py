from preferendus.optimization.aco.modo import PreferendusACO


def test_preferendus_aco_smoke_run() -> None:
    edges = [[1], [2], []]
    graph = {
        "edges": edges,
        "idx_start": 0,
        "idx_goal": 2,
        "edge_scores": {(0, 1): 1.0, (1, 2): 1.0},
        "edge_distances": {(0, 1): 2.0, (1, 2): 3.0},
    }

    preference_functions = {
        1: {
            "distance": lambda d: max(0.0, 100.0 - float(d) * 10.0),
        }
    }
    objective_weights = {1: {"distance": 1.0}}

    @PreferendusACO(
        preference_functions=preference_functions,
        stakeholder_weights=[1.0],
        objective_weights=objective_weights,
        graph=graph,
        optimization_options={"num_ants": 4, "num_iterations": 2, "seed": 7},
    )
    def evaluate(path):
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += graph["edge_distances"][(path[i], path[i + 1])]
        return None, None, total_distance

    result = evaluate()

    assert result is not None
    assert result["x_opt"] == [0, 1, 2]
    assert isinstance(result["utility"], float)
    assert "history" in result
