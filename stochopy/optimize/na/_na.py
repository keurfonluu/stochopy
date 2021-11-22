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
    callback=True,
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
    callback : callable or None, optional, default None
        Called after each iteration. It is a callable with the signature ``callback(X, OptimizeResult state)``, where ``X`` is the current population and ``state`` is a partial :class:`stochopy.optimize.OptimizeResult` object with the same fields as the ones from the return (except ``"success"``, ``"status"`` and ``"message"``).

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

    # Callback
    if callback is not None and not hasattr(callback, "__call__"):
        raise ValueError()

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
        callback,
    )
    res = na(fun, args, True, workers, backend, *optargs)

    return res


@optimizer
def na(
    funnorm,
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
    callback,
):
    """Optimize with Neighborhood Algorithm."""
    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Normalize and unnormalize
    normalize = lambda x: (x - lower) / (upper - lower)
    unnormalize = lambda x: x * (upper - lower) + lower

    fun = lambda x: funnorm(unnormalize(x))

    # Number of resampling
    nr = max(1, int(nrperc * popsize))

    # Initial population
    X = x0 if x0 is not None else lhs(popsize, ndim, bounds)
    X = normalize(X)
    pbest = X.copy()

    # Evaluate initial population
    pfit = fun(X)
    pbestfit = pfit.copy()

    # Initial best solution
    gbidx = numpy.argmin(pbestfit)
    gfit = pbestfit[gbidx]
    gbest = X[gbidx].copy()

    # Store all models sampled
    Xall = X.copy()
    Xallfit = pfit.copy()

    # Initialize arrays
    if return_all:
        xall = numpy.empty((maxiter, popsize, ndim))
        funall = numpy.empty((maxiter, popsize))
        xall[0] = unnormalize(X)
        funall[0] = pfit.copy()

    # First iteration for callback
    if callback is not None:
        res = OptimizeResult(
            x=unnormalize(gbest),
            fun=gfit,
            nfev=popsize,
            nit=1,
        )
        if return_all:
            res.update({"xall": xall[:1], "funall": funall[:1]})
        
        callback(unnormalize(X), res)

    # Iterate until one of the termination criterion is satisfied
    it = 1
    converged = False
    while not converged:
        it += 1

        # Mutation
        X = mutation(Xall, Xallfit, popsize, ndim, nr)

        # Selection
        gbest, gfit, pfit, status = selection_sync(
            it, X, gbest, pbest, pbestfit, maxiter, xtol, ftol, fun
        )
        Xall = numpy.vstack((X, Xall))
        Xallfit = numpy.concatenate((pfit, Xallfit))

        if return_all:
            xall[it - 1] = unnormalize(X)
            funall[it - 1] = pfit.copy()

        converged = status is not None

        if callback is not None:
            res = OptimizeResult(
                x=unnormalize(gbest),
                fun=gfit,
                nfev=it * popsize,
                nit=it,
            )
            if return_all:
                res.update({"xall": xall[:it], "funall": funall[:it]})
            
            callback(unnormalize(X), res)

    res = OptimizeResult(
        x=unnormalize(gbest),
        success=status >= 0,
        status=status,
        message=messages[status],
        fun=gfit,
        nfev=it * popsize,
        nit=it,
    )
    if return_all:
        res.update({"xall": xall[:it], "funall": funall[:it]})

    return res


def mutation(Xall, Xallfit, popsize, ndim, nr):
    """
    Update population.

    Note
    ----
    Code adapted from <https://github.com/keithfma/neighborhood/blob/master/neighborhood/search.py>

    """
    X = numpy.empty((popsize, ndim))

    ix = Xallfit.argsort()[:nr]
    for i in range(popsize):
        k = ix[i % nr]
        X[i] = Xall[k].copy()
        U = numpy.delete(Xall, k, axis=0)

        d1 = 0.0
        d2 = ((U[:, 1:] - X[i, 1:]) ** 2).sum(axis=1)

        for j in range(ndim):
            lim = 0.5 * (Xall[k, j] + U[:, j] + (d1 - d2) / (Xall[k, j] - U[:, j]))

            idx = lim <= X[i, j]
            low = max(lim[idx].max(), 0.0) if idx.sum() else 0.0

            idx = lim >= X[i, j]
            high = min(lim[idx].min(), 1.0) if idx.sum() else 1.0

            X[i, j] = numpy.random.uniform(low, high)

            if j < ndim - 1:
                d1 += (Xall[k, j] - X[i, j]) ** 2 - (Xall[k, j + 1] - X[i, j + 1]) ** 2
                d2 += (U[:, j] - X[i, j]) ** 2 - (U[:, j + 1] - X[i, j + 1]) ** 2

    return X


register("na", minimize)
