import pytest
from pytest import approx
import numpy as np

from autostep.strategies import dx_min_cut, dx_relative_evaluation_order
from autostep.numerical_derivatives import df_dx_numerical


@pytest.fixture
def x_range():
    x_initial = 0.0
    x_final = 5.0
    num_of_points = 20
    return np.linspace(x_initial, x_final, num_of_points)


def f_increasing_monotonic(x):
    return x * x * x - x * x + x - 5


def f_decreasing_monotonic(x):
    return -x * x * x - x * x + x - 5


def f_non_monotonic(x):
    return np.sin(x) * np.exp(-x)


def f_nearly_constant(x):
    return np.sin(x) + 1e14


def dfdx_increasing_monotonic(x):
    return 3 * x * x - 2 * x + 1


def dfdx_decreasing_monotonic(x):
    return -3 * x * x - 2 * x + 1


def dfdx_non_monotonic(x):
    return np.exp(-x) * (np.cos(x) - np.sin(x))


def dfdx_nearly_constant(x):
    return np.cos(x)


@pytest.mark.parametrize(
    "f, dfdx_exact, rel_error",
    [
        [f_increasing_monotonic, dfdx_increasing_monotonic, 1e-3],
        [f_decreasing_monotonic, dfdx_decreasing_monotonic, 1e-3],
        [f_non_monotonic, dfdx_non_monotonic, 1e-3],
        pytest.param(
            f_nearly_constant,
            dfdx_nearly_constant,
            1e-3,
            marks=pytest.mark.xfail(reason="Corner case to be solved"),
        ),
    ],
)
def test_dx_min_cut(f, dfdx_exact, rel_error, x_range):
    f_evaluations = f(x_range)
    dx = dx_min_cut(f_evaluations)
    estimated_dfdx = df_dx_numerical(f, x_range, dx)
    relative_error_l2 = np.linalg.norm(estimated_dfdx - dfdx_exact(x_range)) / np.linalg.norm(
        dfdx_exact(x_range)
    )
    assert relative_error_l2 < 5e-3


@pytest.mark.parametrize(
    "f, dfdx_exact, rel_error",
    [
        [f_increasing_monotonic, dfdx_increasing_monotonic, 1e-3],
        [f_decreasing_monotonic, dfdx_decreasing_monotonic, 0.05],
        [f_non_monotonic, dfdx_non_monotonic, 1e-3],
        pytest.param(
            f_nearly_constant,
            dfdx_nearly_constant,
            1e-2,
            marks=pytest.mark.xfail(reason="Corner case to be solved"),
        ),
    ],
)
def test_dx_relative_evaluation_order(f, dfdx_exact, rel_error, x_range):
    f_evaluations = f(x_range)
    dx = dx_relative_evaluation_order(f_evaluations)
    estimated_dfdx = df_dx_numerical(f, x_range, dx)
    relative_error_l2 = np.linalg.norm(estimated_dfdx - dfdx_exact(x_range)) / np.linalg.norm(
        dfdx_exact(x_range)
    )
    assert relative_error_l2 < rel_error
