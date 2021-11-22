import numpy as np

from .._common import lhs, messages, optimizer, selection_async, selection_sync
from .._helpers import OptimizeResult, register
from ._constraints import _constraints_map
from ._strategy import _strategy_map

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
    mutation=0.5,
    recombination=0.9,
    strategy="best1bin",
    seed=None,
    xtol=1.0e-8,
    ftol=1.0e-8,
    constraints=None,
    updating="immediate",
    workers=1,
    backend=None,
    return_all=False,
    callback=None,
):
    """
    Minimize an objective function using Differential Evolution (DE).

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
    mutation : scalar, optional, default 0.5
        The mutation constant. In the literature this is also known as differential weight, being denoted by F. It should be in the range [0, 2]. Increasing the mutation constant increases the search radius, but will slow down convergence.
    recombination : scalar, optional, default 0.9
        The recombination constant, should be in the range [0, 1]. In the literature this is also known as the crossover probability, being denoted by CR. Increasing this value allows a larger number of mutants to progress into the next generation, but at the risk of population stability.
    strategy : str, optional, default 'best1bin'
        The differential evolution strategy to use. Should be one of:

         - 'rand1bin'
         - 'rand2bin'
         - 'best1bin'
         - 'best2bin'

    seed : int or None, optional, default None
        Seed for random number generator.
    xtol : scalar, optional, default 1.0e-8
        Solution tolerance for termination.
    ftol : scalar, optional, default 1.0e-8
        Objective function value tolerance for termination.
    constraints : str or None, optional, default None
        Constraints definition:

         - None: no constraint
         - 'Random': infeasible solutions are resampled in the feasible space defined by `bounds`

    updating : str {'immediate', 'deferred'}, optional, default 'immediate'
        If ``'immediate'``, the best solution vector is continuously updated within a single generation. This can lead to faster convergence as candidate solutions can take advantage of continuous improvements in the best solution. With ``'deferred'``, the best solution vector is updated once per generation. Only ``'deferred'`` is compatible with parallelization, and is overridden when ``workers`` is not ``0`` or ``1`` or ``backend == 'mpi'``.
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
    .. [1] R. Storn and K. Price, *Differential Evolution - A Simple and Efficient Heuristic for global Optimization over Continuous Spaces*, Journal of Global Optimization, 1997, 11(4): 341-359

    """
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if np.ndim(bounds) != 2:
        raise ValueError()

    # Initial guess x0
    if x0 is not None:
        if np.ndim(x0) != 2 or np.shape(x0)[1] != len(bounds):
            raise ValueError()

    # Population size
    if popsize < 2:
        raise ValueError()

    if x0 is not None and len(x0) != popsize:
        raise ValueError()

    # DE parameters
    if not 0.0 <= mutation <= 2.0:
        raise ValueError()

    if not 0.0 <= recombination <= 1.0:
        raise ValueError()

    if updating not in {"immediate", "deferred"}:
        raise ValueError()

    F = mutation
    CR = recombination
    mut = _strategy_map[strategy]

    # Synchronize
    sync = updating == "deferred"
    sync = sync or workers not in {0, 1}
    sync = sync or backend == "mpi"

    # Seed
    if seed is not None:
        np.random.seed(seed)

    # Callback
    if callback is not None and not hasattr(callback, "__call__"):
        raise ValueError()

    # Run in serial or parallel
    optargs = (
        bounds,
        x0,
        maxiter,
        popsize,
        F,
        CR,
        mut,
        constraints,
        xtol,
        ftol,
        return_all,
        callback,
    )
    res = de(fun, args, sync, workers, backend, *optargs)

    return res


@optimizer
def de(
    fun,
    args,
    sync,
    workers,
    backend,
    bounds,
    x0,
    maxiter,
    popsize,
    F,
    CR,
    mut,
    constraints,
    xtol,
    ftol,
    return_all,
    callback,
):
    """Optimize with DE."""
    ndim = len(bounds)
    lower, upper = np.transpose(bounds)

    # Constraints
    cons = _constraints_map[constraints](lower, upper)

    # Iteration
    de_iter = de_sync if sync else de_async

    # Initial population
    X = x0 if x0 is not None else lhs(popsize, ndim, bounds)
    U = np.empty((popsize, ndim))

    # Evaluate initial population
    pfit = fun(X)
    pbestfit = pfit.copy()

    # Initial best solution
    gbidx = np.argmin(pbestfit)
    gfit = pbestfit[gbidx]
    gbest = X[gbidx].copy()

    # Initialize arrays
    if return_all:
        xall = np.empty((maxiter, popsize, ndim))
        funall = np.empty((maxiter, popsize))
        xall[0] = X.copy()
        funall[0] = pfit.copy()

    # First iteration for callback
    if callback is not None:
        res = OptimizeResult(x=gbest, fun=gfit, nfev=popsize, nit=1,)
        if return_all:
            res.update({"xall": xall[:1], "funall": funall[:1]})

        callback(X, res)

    # Iterate until one of the termination criterion is satisfied
    it = 1
    converged = False
    while not converged:
        it += 1

        r1 = np.random.rand(popsize, ndim)
        X, gbest, pbestfit, gfit, pfit, status = de_iter(
            it,
            X,
            U,
            gbest,
            pbestfit,
            gfit,
            pfit,
            F,
            CR,
            r1,
            maxiter,
            xtol,
            ftol,
            fun,
            mut,
            cons,
        )

        if return_all:
            xall[it - 1] = X.copy()
            funall[it - 1] = pbestfit.copy()

        converged = status is not None

        if callback is not None:
            res = OptimizeResult(x=gbest, fun=gfit, nfev=it * popsize, nit=it,)
            if return_all:
                res.update({"xall": xall[:it], "funall": funall[:it]})

            callback(X, res)

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
        res.update({"xall": xall[:it], "funall": funall[:it]})

    return res


def delete_shuffle_sync(popsize):
    """Delete current solution from population for mutation (synchronous)."""
    return np.transpose([delete_shuffle_async(i, popsize) for i in range(popsize)])


def delete_shuffle_async(i, popsize):
    """Delete current solution from population for mutation (asynchronous)."""
    return np.random.permutation(np.delete(np.arange(popsize), i))


def de_sync(
    it,
    X,
    U,
    gbest,
    pbestfit,
    gfit,
    pfit,
    F,
    CR,
    r1,
    maxiter,
    xtol,
    ftol,
    fun,
    mut,
    cons,
):
    """Synchronous DE."""
    popsize, ndim = X.shape

    # Mutation
    V = mut(delete_shuffle_sync(popsize), F, X, gbest)

    # Recombination
    mask = np.zeros_like(r1, dtype=bool)
    irand = np.random.randint(ndim, size=popsize)
    for i in range(popsize):
        mask[i, irand[i]] = True

    U[:] = cons(np.where(np.logical_or(mask, r1 <= CR), V, X))

    # Selection
    gbest, gfit, pfit, status = selection_sync(
        it, U, gbest, X, pbestfit, maxiter, xtol, ftol, fun
    )

    return X, gbest, pbestfit, gfit, pfit, status


def de_async(
    it,
    X,
    U,
    gbest,
    pbestfit,
    gfit,
    pfit,
    F,
    CR,
    r1,
    maxiter,
    xtol,
    ftol,
    fun,
    mut,
    cons,
):
    """Asynchronous DE."""
    popsize, ndim = X.shape

    for i in range(popsize):
        # Mutation
        V = mut(delete_shuffle_async(i, popsize), F, X, gbest)

        # Recombination
        mask = np.zeros(ndim, dtype=bool)
        irand = np.random.randint(ndim)
        mask[irand] = True
        U[i] = cons(np.where(np.logical_or(mask, r1[i] <= CR), V, X[i]))

        # Selection
        gbest, gfit, pfit[i], status = selection_async(
            it, U, gbest, gfit, X, pbestfit, maxiter, xtol, ftol, fun, i
        )

    # Stop if maximum iteration is reached
    if status is None and it >= maxiter:
        status = -1

    return X, gbest, pbestfit, gfit, pfit, status


register("de", minimize)
