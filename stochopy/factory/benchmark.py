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
    x = numpy.asarray(x)
    ndim = x.size
    e = 2.7182818284590451
    sum1 = numpy.sqrt(1.0 / ndim * numpy.square(x).sum())
    sum2 = 1.0 / ndim * numpy.cos(2.0 * numpy.pi * x).sum()
    return 20.0 + e - 20.0 * numpy.exp(-0.2 * sum1) - numpy.exp(sum2)


def griewank(x):
    x = numpy.asarray(x)
    ndim = x.size
    sum1 = numpy.square(x).sum() / 4000.0
    prod1 = numpy.prod(numpy.cos(x / numpy.sqrt(numpy.arange(1, ndim + 1))))
    return 1.0 + sum1 - prod1


def quartic(x):
    x = numpy.asarray(x)
    ndim = x.size
    return (numpy.arange(1, ndim + 1) * numpy.power(x, 4)).sum()


def rastrigin(x):
    x = numpy.asarray(x)
    ndim = x.size
    sum1 = (numpy.square(x) - 10.0 * numpy.cos(2.0 * numpy.pi * x)).sum()
    return 10.0 * ndim + sum1


def rosenbrock(x):
    x = numpy.asarray(x)
    sum1 = ((x[1:] - x[:-1] ** 2) ** 2).sum()
    sum2 = numpy.square(1.0 - x[:-1]).sum()
    return 100.0 * sum1 + sum2


def sphere(x):
    return numpy.square(x).sum()


def styblinski_tang(x):
    x = numpy.asarray(x)
    sum1 = (numpy.power(x, 4) - 16.0 * numpy.square(x) + 5.0 * x).sum()
    return 0.5 * sum1 + 39.16599 * x.size
