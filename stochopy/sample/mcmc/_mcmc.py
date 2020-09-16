import numpy

from .._common import in_search_space
from .._helpers import register, SampleResult

__all__ = [
    "sample",
]


def sample(
    fun,
    bounds,
    x0=None,
    args=(),
    maxiter=100,
    stepsize=0.1,
    perc=1.0,
    constraints=None,
):
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if numpy.ndim(bounds) != 2:
        raise ValueError()

    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Initial guess x0
    if x0 is not None and len(x0) != ndim:
        raise ValueError()

    # Step size
    if numpy.ndim(stepsize) == 0:
        stepsize = numpy.full(ndim, stepsize)

    if len(stepsize) != ndim:
        raise ValueError()

    stepsize *= 0.5 * (upper - lower)
    
    # Number of dimensions to perturb per iteration
    if not 0.0 <= perc <= 1.0:
        raise ValueError()

    ndim_per_iter = max(1, int(perc * ndim))
    
    # Initialize arrays
    xall = numpy.empty((maxiter, ndim))
    funall = numpy.empty(maxiter)
    xall[0] = x0 if x0 is not None else numpy.random.uniform(lower, upper)
    funall[0] = fun(xall[0], *args)

    # Metropolis-Hastings algorithm
    i = 1
    n_accepted = 0
    while i < maxiter:
        for j in numpy.arange(0, ndim, ndim_per_iter):
            jmax = min(ndim, j + ndim_per_iter - 1)
            perturbation = numpy.random.randn(jmax - j + 1) * stepsize[j : jmax + 1]

            xall[i] = numpy.copy(xall[i - 1])
            xall[i, j : jmax + 1] += perturbation

            accept = False
            if in_search_space(xall[i], lower, upper, constraints):
                funall[i] = fun(xall[i], *args)
                log_alpha = min(0.0, funall[i - 1] - funall[i])
                accept = log_alpha > numpy.log(numpy.random.rand())

            if accept:
                n_accepted += 1
            else:
                xall[i] = xall[i - 1]
                funall[i] = funall[i - 1]

            i += 1
            if i == maxiter:
                break

    idx = numpy.argmin(funall)
    return SampleResult(
        x=xall[idx],
        fun=funall[idx],
        nfev=maxiter,
        nit=maxiter,
        accept_ratio=n_accepted / maxiter,
        xall=xall,
        funall=funall,
    )


register("mcmc", sample)
