"""
This module provides strategies to automatically estimate step size to use in numerical derivatives calculations.
"""

import numpy as np


def dx_relative_evaluation_order(evaluations: np.ndarray) -> np.float64:
    """
    TODO: docs

    :param np.ndarray evaluations:
    :return:
    :rtype: np.float64
    """
    evaluations_without_zeros = evaluations[np.abs(evaluations) > 0]
    evaluations_absolute_values = np.abs(
        evaluations_without_zeros
    )  # this is necessary for decreasing derivative cases
    evaluations_absolute_values_sorted = np.sort(evaluations_absolute_values)
    scaling_factor = _calculate_min_gap(evaluations_absolute_values_sorted)
    return scaling_factor * np.linalg.norm(np.min(evaluations_absolute_values))


def dx_min_cut(evaluations: np.ndarray, tol: float = 1e-16) -> np.float64:
    """
    TODO: docs

    :param np.ndarray evaluations:
    :param np.float64 tol:
    :return:
    :rtype: np.float64
    """
    evaluations_without_zeros = evaluations[np.abs(evaluations) > 0]
    sorted_evaluations = np.sort(np.abs(evaluations_without_zeros))  # increasing order
    scaling_factor = _calculate_min_gap(sorted_evaluations)

    dx = 0.0
    next_index = 1
    num_of_evaluations = len(sorted_evaluations)
    while dx < tol:
        if next_index == num_of_evaluations:
            raise RuntimeError("No valid step can be calculated.")

        # dx = scaling_factor * sorted_evaluations[0]
        dx = (
            scaling_factor
            * (sorted_evaluations[next_index] - sorted_evaluations[0])
            * sorted_evaluations[0]
        )
        # dx = scaling_factor * (sorted_evaluations[next_index] - sorted_evaluations[0])
        # dx = scaling_factor
        next_index += 1

    return dx


def dx_min_cut_scaled(x: np.ndarray, evaluations: np.ndarray, tol: float = 1e-16) -> np.float64:
    """
    TODO: docs

    :param np.ndarray x:
    :param np.ndarray evaluations:
    :param np.float64 tol:
    :return:
    :rtype: np.float64
    """
    evaluations_without_zeros = evaluations[np.abs(evaluations) > 0]
    sorted_evaluations = np.sort(np.abs(evaluations_without_zeros))  # increasing order
    scaling_factor = _calculate_min_gap(evaluations_without_zeros)
    delta_x = np.abs((x.max() - x.min()) * x.mean())
    delta_y = np.abs((evaluations.max() - evaluations.min()) * evaluations.mean())
    rescaling_evaluations_to_x = delta_x / delta_y

    dx = 0.0
    next_index = 0
    num_of_evaluations = len(sorted_evaluations)
    while dx < tol:
        if next_index == num_of_evaluations:
            raise RuntimeError("No valid step can be calculated.")

        # dx = scaling_factor * sorted_evaluations[0]  # this one behaves as dx_relative_evaluation_order
        dx = scaling_factor * sorted_evaluations[next_index] * rescaling_evaluations_to_x
        next_index += 1

    return dx


def _calculate_min_gap(evaluations: np.ndarray) -> np.float64:
    """
    TODO: docs

    :param np.ndarray evaluations:
    :return:
    :rtype: np.float64
    """
    index_max = len(evaluations)
    min_gap_values = np.array([])
    for i in range(index_max - 1):
        gap_value = np.abs(evaluations[i + 1] - evaluations[i])
        if gap_value > 0:
            min_gap_values = np.append(min_gap_values, gap_value)

    min_gap = np.min(min_gap_values)
    if min_gap > 1:
        min_gap = 1 / min_gap
    return min_gap
