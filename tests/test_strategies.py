import pytest
from pytest import approx
import numpy as np

from autostep.strategies import dx_min_cut, dx_relative_evaluation_order, _calculate_min_gap
from autostep.numerical_derivatives import forward_diff
from tests.benchmarks import (
    f_increasing_monotonic,
    f_decreasing_monotonic,
    f_non_monotonic,
    f_nearly_constant,
    f_constant,
)
from tests.benchmarks import (
    dfdx_increasing_monotonic,
    dfdx_decreasing_monotonic,
    dfdx_non_monotonic,
    dfdx_nearly_constant,
    dfdx_constant,
)


@pytest.fixture
def x_range():
    x_initial = 0.0
    x_final = 5.0
    num_of_points = 20
    return np.linspace(x_initial, x_final, num_of_points)


@pytest.mark.parametrize(
    "f, dfdx_exact, rel_error",
    [
        [f_increasing_monotonic, dfdx_increasing_monotonic, 2e-3],
        [f_decreasing_monotonic, dfdx_decreasing_monotonic, 3e-3],
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
    estimated_dfdx = forward_diff(f, x_range, dx)
    relative_error_l2 = np.linalg.norm(estimated_dfdx - dfdx_exact(x_range)) / np.linalg.norm(
        dfdx_exact(x_range)
    )
    assert relative_error_l2 < rel_error


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
    estimated_dfdx = forward_diff(f, x_range, dx)
    relative_error_l2 = np.linalg.norm(estimated_dfdx - dfdx_exact(x_range)) / np.linalg.norm(
        dfdx_exact(x_range)
    )
    assert relative_error_l2 < rel_error


@pytest.mark.xfail(reason="Corner case: constant functions derivatives.")
def test_dx_relative_evaluation_order_for_constant_function(x_range):
    f_evaluations = f_constant(x_range)
    dx = dx_relative_evaluation_order(f_evaluations)
    estimated_dfdx = forward_diff(f_constant, x_range, dx)
    relative_error_l2 = np.linalg.norm(estimated_dfdx - dfdx_exact(x_range)) / np.linalg.norm(
        dfdx_exact(x_range)
    )
    assert relative_error_l2 < rel_error


@pytest.mark.xfail(reason="Corner case: constant functions derivatives.")
def test_dx_min_cut_for_constant_function(x_range):
    f_evaluations = f_constant(x_range)
    dx = dx_min_cut(f_evaluations)
    estimated_dfdx = forward_diff(f_constant, x_range, dx)
    relative_error_l2 = np.linalg.norm(estimated_dfdx - dfdx_exact(x_range)) / np.linalg.norm(
        dfdx_exact(x_range)
    )
    assert relative_error_l2 < rel_error


@pytest.mark.parametrize(
    "f", [f_increasing_monotonic, f_decreasing_monotonic, f_non_monotonic, f_nearly_constant,],
)
def test_min_cut_estimation_with_too_high_tolerance(f, x_range):
    tol = 1e1
    f_evaluations = f(x_range)
    with pytest.raises(RuntimeError, match=r"No valid step can be calculated."):
        dx_min_cut(f_evaluations, tol=tol)


def test_failing_scaling_factor_for_constant_function(x_range):
    f_sorted_evaluations = f_constant(x_range)
    with pytest.raises(RuntimeError, match=r"No valid max-min gap can be calculated."):
        _calculate_min_gap(f_sorted_evaluations)
