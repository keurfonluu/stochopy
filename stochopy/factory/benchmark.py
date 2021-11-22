import numpy as np

__all__ = [
    "ackley",
    "griewank",
    "quartic",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "styblinski_tang",
]


def ackley(x):
    """
    The Ackley function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Ackley function is to be computed.

    Returns
    -------
    float
        The value of the Ackley function.

    """
    x = np.asarray(x)
    ndim = x.size
    e = 2.7182818284590451
    sum1 = np.sqrt(1.0 / ndim * np.square(x).sum())
    sum2 = 1.0 / ndim * np.cos(2.0 * np.pi * x).sum()
    return 20.0 + e - 20.0 * np.exp(-0.2 * sum1) - np.exp(sum2)


def griewank(x):
    """
    The Griewank function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Griewank function is to be computed.

    Returns
    -------
    float
        The value of the Griewank function.

    """
    x = np.asarray(x)
    ndim = x.size
    sum1 = np.square(x).sum() / 4000.0
    prod1 = np.prod(np.cos(x / np.sqrt(np.arange(1, ndim + 1))))
    return 1.0 + sum1 - prod1


def quartic(x):
    """
    The Quartic function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Quartic function is to be computed.

    Returns
    -------
    float
        The value of the Quartic function.

    """
    x = np.asarray(x)
    ndim = x.size
    return (np.arange(1, ndim + 1) * np.power(x, 4)).sum()


def rastrigin(x):
    """
    The Rastrigin function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rastrigin function is to be computed.

    Returns
    -------
    float
        The value of the Rastrigin function.

    """
    x = np.asarray(x)
    ndim = x.size
    sum1 = (np.square(x) - 10.0 * np.cos(2.0 * np.pi * x)).sum()
    return 10.0 * ndim + sum1


def rosenbrock(x):
    """
    The Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    float
        The value of the Rosenbrock function.

    """
    x = np.asarray(x)
    sum1 = ((x[1:] - x[:-1] ** 2) ** 2).sum()
    sum2 = np.square(1.0 - x[:-1]).sum()
    return 100.0 * sum1 + sum2


def sphere(x):
    """
    The Sphere function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Sphere function is to be computed.

    Returns
    -------
    float
        The value of the Sphere function.

    """
    return np.square(x).sum()


def styblinski_tang(x):
    """
    The Styblinski-Tang function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Styblinski-Tang function is to be computed.

    Returns
    -------
    float
        The value of the Styblinski-Tang function.

    """
    x = np.asarray(x)
    sum1 = (np.power(x, 4) - 16.0 * np.square(x) + 5.0 * x).sum()
    return 0.5 * sum1 + 39.16599 * x.size
