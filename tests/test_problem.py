import numpy as np
import pytest

from stableui.problem import (
    Instance,
    count_switches,
    enumerate_layouts,
    layout_matrix,
    switching_cost,
    total_utility,
    validate_layout,
)


def test_enumerate_layouts_count():
    assert len(enumerate_layouts(6, 2)) == 15
    assert enumerate_layouts(3, 2) == [(0, 1), (0, 2), (1, 2)]


def test_layout_matrix():
    mat = layout_matrix([(0, 2)], 3)
    assert mat.tolist() == [[True, False, True]]


def test_switching_cost():
    assert switching_cost(None, (0, 1)) == 0
    assert switching_cost((0, 1), (0, 1)) == 0
    assert switching_cost((0, 1), (1, 2)) == 1
    assert switching_cost((0, 1), (2, 3)) == 2


def test_total_utility_hand_computed():
    values = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]])
    inst = Instance(values=values, k=2, lam=1.5)
    # t0: {0,2} -> 5, no previous layout; t1: {1,2} -> 6 - 1.5 * 1
    assert total_utility(inst, [(0, 2), (1, 2)]) == pytest.approx(9.5)


def test_total_utility_with_initial_layout():
    inst = Instance(values=np.array([[1.0, 2.0]]), k=1, lam=0.5, initial=(0,))
    assert total_utility(inst, [(1,)]) == pytest.approx(1.5)


def test_count_switches():
    assert count_switches([(0, 1), (0, 2), (3, 4)]) == 3
    assert count_switches([(0, 1)], initial=(2, 3)) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": np.ones(3), "k": 1, "lam": 0.0},
        {"values": np.ones((2, 3)), "k": 4, "lam": 0.0},
        {"values": np.ones((2, 3)), "k": 1, "lam": -1.0},
        {"values": np.ones((2, 3)), "k": 2, "lam": 0.0, "initial": (0, 0)},
    ],
)
def test_instance_validation(kwargs):
    with pytest.raises(ValueError):
        Instance(**kwargs)


def test_validate_layout_canonicalises():
    assert validate_layout([2, 0], n=3, k=2) == (0, 2)
    with pytest.raises(ValueError):
        validate_layout([0, 3], n=3, k=2)
