import numpy as np
import pytest

from preferendus.optimization.aco._aggregator import a_fine_aggregator


def test_aggregator_returns_midpoint_for_identical_scores() -> None:
    weights = [0.5, 0.5]
    preferences = np.array([[10.0, 10.0], [20.0, 20.0]])

    scores = a_fine_aggregator(weights, preferences)

    assert np.allclose(scores, np.array([50.0, 50.0]))


def test_aggregator_raises_when_all_weights_zero() -> None:
    with pytest.raises(ValueError, match="All weights are zero"):
        a_fine_aggregator([0.0, 0.0], np.array([[1.0, 2.0], [3.0, 4.0]]))
