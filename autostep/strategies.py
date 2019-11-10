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
    scaling_factor = _calculate_scaling_factor(evaluations_absolute_values_sorted)
    return scaling_factor * np.linalg.norm(np.min(evaluations_absolute_values))


def dx_min_cut(evaluations: np.ndarray, tol: float = 1e-12) -> np.float64:
    """
    TODO: docs

    :param np.ndarray evaluations:
    :param np.float64 tol:
    :return:
    :rtype: np.float64
    """
    evaluations_without_zeros = evaluations[np.abs(evaluations) > 0]
    sorted_evaluations = np.sort(np.abs(evaluations_without_zeros))  # increasing order
    scaling_factor = _calculate_scaling_factor(sorted_evaluations)

    dx = 0.0
    next_index = 1
    num_of_evaluations = len(sorted_evaluations)
    while dx < tol:
        #         print(f"{num_of_evaluations}\t{next_index}\t{dx}\t{scaling_factor}")
        if next_index == num_of_evaluations:
            raise RuntimeError("No valid step can be calculated.")

        dx = scaling_factor * np.abs(sorted_evaluations[next_index] - sorted_evaluations[0])
        next_index += 1

    if dx < tol:
        raise RuntimeError("No significant step can be estimated.")

    return dx


def _calculate_scaling_factor(sorted_evaluations: np.ndarray) -> np.float64:
    """
    TODO: docs

    :param np.ndarray sorted_evaluations:
    :return:
    :rtype: np.float64
    """
    index_max = len(sorted_evaluations) - 1
    max_min_gap = 0.0
    while max_min_gap <= 0:
        if index_max < 0:
            raise RuntimeError("No valid max-min gap can be calculated.")

        max_min_gap = sorted_evaluations[index_max] - sorted_evaluations[0]
        index_max -= 1

    min_value = sorted_evaluations[0]
    scaling_factor = max_min_gap / min_value
    if scaling_factor > 1:
        scaling_factor = 1 / scaling_factor
    return scaling_factor
