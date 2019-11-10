import numpy as np


def f_increasing_monotonic(x):
    return x * x * x - x * x + x - 5


def f_decreasing_monotonic(x):
    return -x * x * x - x * x + x - 5


def f_non_monotonic(x):
    return np.sin(x) * np.exp(-x)


def f_nearly_constant(x):
    return np.sin(x) + 1e14


def f_constant(x):
    return np.ones(len(x))


def dfdx_increasing_monotonic(x):
    return 3 * x * x - 2 * x + 1


def dfdx_decreasing_monotonic(x):
    return -3 * x * x - 2 * x + 1


def dfdx_non_monotonic(x):
    return np.exp(-x) * (np.cos(x) - np.sin(x))


def dfdx_nearly_constant(x):
    return np.cos(x)


def dfdx_constant(x):
    return np.zeros(len(x))
