import numpy

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
    x = numpy.asarray(x)
    ndim = x.size
    e = 2.7182818284590451
    sum1 = numpy.sqrt(1.0 / ndim * numpy.square(x).sum())
    sum2 = 1.0 / ndim * numpy.cos(2.0 * numpy.pi * x).sum()
    return 20.0 + e - 20.0 * numpy.exp(-0.2 * sum1) - numpy.exp(sum2)


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
    x = numpy.asarray(x)
    ndim = x.size
    sum1 = numpy.square(x).sum() / 4000.0
    prod1 = numpy.prod(numpy.cos(x / numpy.sqrt(numpy.arange(1, ndim + 1))))
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
    x = numpy.asarray(x)
    ndim = x.size
    return (numpy.arange(1, ndim + 1) * numpy.power(x, 4)).sum()


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
    x = numpy.asarray(x)
    ndim = x.size
    sum1 = (numpy.square(x) - 10.0 * numpy.cos(2.0 * numpy.pi * x)).sum()
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
    x = numpy.asarray(x)
    sum1 = ((x[1:] - x[:-1] ** 2) ** 2).sum()
    sum2 = numpy.square(1.0 - x[:-1]).sum()
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
    return numpy.square(x).sum()


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
    x = numpy.asarray(x)
    sum1 = (numpy.power(x, 4) - 16.0 * numpy.square(x) + 5.0 * x).sum()
    return 0.5 * sum1 + 39.16599 * x.size
