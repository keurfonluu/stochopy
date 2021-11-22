import numpy as np

from .._common import in_search_space
from .._helpers import SampleResult, register

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
    seed=None,
    constraints=None,
    return_all=True,
    callback=None,
):
    """
    Sample the variable space using the Metropolis-Hastings algorithm.

    Parameters
    ----------
    fun : callable
        The objective function to be sampled. Must be in the form ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
    bounds : array_like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the sampling argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    x0 : array_like or None, optional, default None
        Initial sample. Array of real elements of size (``ndim``,), where ``ndim`` is the number of independent variables.
    args : tuple, optional, default None
        Extra arguments passed to the objective function.
    maxiter : int, optional, default 100
        Total number of samples to generate.
    stepsize : scalar or array_like, optional, default 0.1
        Standard deviation of Gaussian perturbation (as a fraction of feasible space defined by ``bounds``).
    perc : scalar, optional, default 1.0
        Number of dimensions to perturb at each iteration (as a fraction of total number of variables).
    seed : int or None, optional, default None
        Seed for random number generator.
    constraints : str or None, optional, default None
        Constraints definition:

         - None: no constraint
         - 'Reject': infeasible solutions are always rejected

    return_all : bool, optional, default True
        Set to True to return an array with shape (``maxiter``, ``ndim``) of all the samples.
    callback : callable or None, optional, default None
        Called after each iteration. It is a callable with the signature ``callback(xk, SampleResult state)``, where ``xk`` is the current population and ``state`` is a :class:`stochopy.sample.SampleResult` object with the same fields as the ones from the return.

    Returns
    -------
    :class:`stochopy.sample.SampleResult`
        The sampling result represented as a :class:`stochopy.sample.SampleResult`. Important attributes are:

         - ``x``: the best sample array
         - ``fun``: the best sample function value
         - ``xall``: the samples array
         - '`funall``: the samples' function value array

    """
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if np.ndim(bounds) != 2:
        raise ValueError()

    ndim = len(bounds)
    lower, upper = np.transpose(bounds)

    # Initial guess x0
    if x0 is not None and len(x0) != ndim:
        raise ValueError()

    # Step size
    if np.ndim(stepsize) == 0:
        stepsize = np.full(ndim, stepsize)

    if len(stepsize) != ndim:
        raise ValueError()

    stepsize *= 0.5 * (upper - lower)

    # Number of dimensions to perturb per iteration
    if not 0.0 <= perc <= 1.0:
        raise ValueError()

    ndim_per_iter = max(1, int(perc * ndim))

    # Seed
    if seed is not None:
        np.random.seed(seed)

    # Callback
    if callback is not None and not hasattr(callback, "__call__"):
        raise ValueError()

    # Initialize arrays
    xall = np.empty((maxiter, ndim))
    funall = np.empty(maxiter)
    xall[0] = x0 if x0 is not None else np.random.uniform(lower, upper)
    funall[0] = fun(xall[0], *args)

    # First iteration for callback
    if callback is not None:
        res = SampleResult(x=xall[0], fun=funall[0], nit=1, accept_ratio=1.0)
        if return_all:
            res.update({"xall": xall[:1], "funall": funall[:1]})

        callback(xall[0], res)

    # Metropolis-Hastings algorithm
    i = 1
    n_accepted = 0
    imin, fmin = 0, np.Inf
    while i < maxiter:
        for j in np.arange(0, ndim, ndim_per_iter):
            jmax = min(ndim, j + ndim_per_iter - 1)
            perturbation = np.random.randn(jmax - j + 1) * stepsize[j : jmax + 1]

            xall[i] = xall[i - 1].copy()
            xall[i, j : jmax + 1] += perturbation

            accept = False
            if in_search_space(xall[i], lower, upper, constraints):
                funall[i] = fun(xall[i], *args)
                log_alpha = min(0.0, funall[i - 1] - funall[i])
                accept = log_alpha > np.log(np.random.rand())

            if accept:
                n_accepted += 1
                if funall[i] < fmin:
                    imin, fmin = i, funall[i]
            else:
                xall[i] = xall[i - 1]
                funall[i] = funall[i - 1]

            i += 1
            if callback is not None:
                res = SampleResult(
                    x=xall[imin], fun=funall[imin], nit=i, accept_ratio=n_accepted / i,
                )
                if return_all:
                    res.update({"xall": xall[: i - 1], "funall": funall[: i - 1]})

                callback(xall[i - 1], res)

            if i == maxiter:
                break

    res = SampleResult(
        x=xall[imin], fun=fmin, nit=maxiter, accept_ratio=n_accepted / maxiter,
    )
    if return_all:
        res.update({"xall": xall, "funall": funall})

    return res


register("mcmc", sample)
