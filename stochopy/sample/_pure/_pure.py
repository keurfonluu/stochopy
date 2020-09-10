import numpy

from .._helpers import register, SampleResult

__all__ = [
    "sample",
]


def sample(fun, bounds, args=(), options={}):
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if numpy.ndim(bounds) != 2:
        raise ValueError()

    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Options
    _options = {
        "maxiter": 100,
    }
    _options.update(options)

    maxiter = _options["maxiter"]

    # Generate random models
    xall = numpy.random.uniform(lower, upper, (maxiter, ndim))
    funall = numpy.array([fun(x, *args) for x in xall])

    idx = numpy.argmin(funall)
    return SampleResult(
        x=xall[idx],
        fun=funall[idx],
        nfev=maxiter,
        nit=maxiter,
        accept_ratio=1.0,
        xall=xall,
        funall=funall,
    )


register("pure", sample)
