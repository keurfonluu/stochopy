import numpy

from .._common import lhs, messages, optimizer, selection_sync
from .._helpers import OptimizeResult, register

__all__ = [
    "minimize",
]


def minimize(
    fun,
    bounds,
    x0=None,
    args=(),
    maxiter=100,
    popsize=10,
    nrperc=0.5,
    seed=None,
    xtol=1.0e-8,
    ftol=1.0e-8,
    workers=1,
    backend=None,
    return_all=False,
):
    """
    Minimize an objective function using Neighborhood Algorithm (NA).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
    bounds : array_like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    x0 : array_like or None, optional, default None
        Initial population. Array of real elements with shape (``popsize``, ``ndim``), where ``ndim`` is the number of independent variables. If ``x0`` is not specified, the population is initialized using Latin Hypercube sampling.
    args : tuple, optional, default None
        Extra arguments passed to the objective function.
    maxiter : int, optional, default 100
        The maximum number of generations over which the entire population is evolved.
    popsize : int, optional, default 10
        Total population size.
    nrperc : scalar, optional, default 0.5
        Number of resamplings (as a fraction of total population size).
    seed : int or None, optional, default None
        Seed for random number generator.
    xtol : scalar, optional, default 1.0e-8
        Solution tolerance for termination.
    ftol : scalar, optional, default 1.0e-8
        Objective function value tolerance for termination.
    workers : int, optional, default 1
        The population is subdivided into workers sections and evaluated in parallel (uses :class:`joblib.Parallel`). Supply -1 to use all available CPU cores.
    backend : str {'loky', 'threading', 'mpi'}, optional, default 'threading'
        Parallel backend to use when ``workers`` is not ``0`` or ``1``:

         - 'loky': disable threading
         - 'threading': enable threading
         - 'mpi': use MPI (uses :mod:`mpi4py`)

    return_all : bool, optional, default False
        Set to True to return an array with shape (``nit``, ``popsize``, ``ndim``) of all the solutions at each iteration.

    Returns
    -------
    :class:`stochopy.optimize.OptimizeResult`
        The optimization result represented as a :class:`stochopy.optimize.OptimizeResult`. Important attributes are:

         - ``x``: the solution array
         - ``fun``: the solution function value
         - ``success``: a Boolean flag indicating if the optimizer exited successfully
         - ``message``: a string which describes the cause of the termination

    References
    ----------
    .. [1] M. Sambridge, *Geophysical inversion with a neighbourhood algorithm - I. Searching a parameter space*, Geophysical Journal International, 1999, 138(2): 479–494

    """
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if numpy.ndim(bounds) != 2:
        raise ValueError()

    # Initial guess x0
    if x0 is not None:
        if numpy.ndim(x0) != 2 or numpy.shape(x0)[1] != len(bounds):
            raise ValueError()

    # Population size
    if popsize < 2:
        raise ValueError()

    if x0 is not None and len(x0) != popsize:
        raise ValueError()

    # NA parameters
    if not 0.0 < nrperc <= 1.0:
        raise ValueError()

    # Seed
    if seed is not None:
        numpy.random.seed(seed)

    # Run in serial or parallel
    optargs = (
        bounds,
        x0,
        maxiter,
        popsize,
        nrperc,
        xtol,
        ftol,
        return_all,
    )
    res = na(fun, args, True, workers, backend, *optargs)

    return res


@optimizer
def na(
    fun,
    args,
    sync,
    workers,
    backend,
    bounds,
    x0,
    maxiter,
    popsize,
    nrperc,
    xtol,
    ftol,
    return_all,
):
    """Optimize with Neighborhood Algorithm."""
    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Number of resampling
    nr = max(2, int(nrperc * popsize))

    # Initial population
    X = x0 if x0 is not None else lhs(popsize, ndim, bounds)
    pbest = X.copy()

    # Evaluate initial population
    pfit = fun(X)
    pbestfit = pfit.copy()

    # Initial best solution
    gbidx = numpy.argmin(pbestfit)
    gfit = pbestfit[gbidx]
    gbest = X[gbidx].copy()

    # Store and rank all models sampled
    Xall, Xallfit = stack_and_sort(X, numpy.empty((0, ndim)), pfit, [])

    # Initialize arrays
    if return_all:
        xall = numpy.empty((maxiter, popsize, ndim))
        funall = numpy.empty((maxiter, popsize))
        xall[0] = X.copy()
        funall[0] = pfit.copy()

    # Iterate until one of the termination criterion is satisfied
    it = 1
    converged = False
    while not converged:
        it += 1

        # Mutation
        X = mutation(X, Xall, popsize, ndim, nr, lower, upper)

        # Selection
        gbest, gfit, pfit, status = selection_sync(
            it, X, gbest, pbest, pbestfit, maxiter, xtol, ftol, fun
        )
        Xall, Xallfit = stack_and_sort(X, Xall, pfit, Xallfit)

        if return_all:
            xall[it - 1] = X.copy()
            funall[it - 1] = pfit.copy()

        converged = status is not None

    res = OptimizeResult(
        x=gbest,
        success=status >= 0,
        status=status,
        message=messages[status],
        fun=gfit,
        nfev=it * popsize,
        nit=it,
    )
    if return_all:
        res["xall"] = xall[:it]
        res["funall"] = funall[:it]

    return res


def mutation(X, Xall, popsize, ndim, nr, lower, upper):
    """
    Update population.
    
    Note
    ----
    Code adapted from <https://github.com/keithfma/neighborhood/blob/master/neighborhood/search.py>

    """
    Xall = (Xall - lower) / (upper - lower)

    for i in range(popsize):
        k = i % nr
        U = numpy.delete(Xall, k, axis=0)
        V = Xall[k].copy()

        d1 = 0.0
        d2 = ((U[:, 1:] - V[1:]) ** 2).sum(axis=1)

        for j in range(ndim):
            lim = 0.5 * (Xall[k, j] + U[:, j] + (d1 - d2) / (Xall[k, j] - U[:, j]))

            try:
                low = max(0.0, lim[lim <= V[j]].max())
            except ValueError:
                low = 0.0

            try:
                high = min(1.0, lim[lim >= V[j]].min())
            except ValueError:
                high = 1.0

            V[j] = numpy.random.uniform(low, high)

            if j < ndim - 1:
                d1 += (Xall[k, j] - V[j]) ** 2 - (Xall[k, j + 1] - V[j + 1]) ** 2
                d2 += (U[:, j] - V[j]) ** 2 - (U[:, j + 1] - V[j + 1]) ** 2

        X[i] = V * (upper - lower) + lower

    return X


def stack_and_sort(X, Xall, pfit, Xallfit):
    """Store and rank all models sampled."""
    Xall = numpy.vstack((X, Xall))
    Xallfit = numpy.concatenate((pfit, Xallfit))
    idx = Xallfit.argsort()

    return Xall[idx], Xallfit[idx]


register("na", minimize)
