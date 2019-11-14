"""
In this modules, simple numerical derivatives are provided based on exact definition of derivatives in a given point.
"""

import numpy as np
import types


def forward_diff(f: types.FunctionType, x: np.ndarray, dx: np.float64) -> np.ndarray:
    """
    TODO: docs

    :param types.FunctionType f:
    :param np.ndarray | float x:
    :param np.float64 dx:
    :return:
    :rtype: np.ndarray | float
    """
    return (f(x + dx) - f(x)) / dx
