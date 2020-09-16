import numpy

from ._constraints import _constraints_map
from ._strategy import _strategy_map
from .._common import messages, parallelize, selection_sync, selection_async
from .._helpers import register, OptimizeResult

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
    recombination=0.1,
    strategy="rand1bin",
    seed=None,
    xtol=1.0e-8,
    ftol=1.0e-8,
    constraints=None,
    updating="deferred",
    workers=1,
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
    if x0 is not None:
        if numpy.ndim(x0) != 2 or numpy.shape(x0)[1] != ndim:
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
    sync = updating == "deferred"

    # Constraints
    cons = _constraints_map[constraints](lower, upper)

    # Synchronize
    de_iter = de_sync if sync else de_async

    # Parallel
    fun = parallelize(fun, args, sync, workers)

    # Seed
    if seed is not None:
        numpy.random.seed(seed)

    # Initial population
    X = (
        x0
        if x0 is not None
        else numpy.random.uniform(lower, upper, (popsize, ndim)) 
    )
    U = numpy.empty((popsize, ndim))

    # Evaluate initial population
    pfit = fun(X) if sync else numpy.array([fun(xx) for xx in X])
    pbestfit = numpy.array(pfit)

    # Initial best solution
    gbidx = numpy.argmin(pbestfit)
    gfit = pbestfit[gbidx]
    gbest = numpy.copy(X[gbidx])

    # Initialize arrays
    xall = numpy.empty((popsize, ndim, maxiter))
    funall = numpy.empty((popsize, maxiter))
    xall[:, :, 0] = numpy.copy(X)
    funall[:, 0] = numpy.copy(pfit)

    # Iterate until one of the termination criterion is satisfied
    it = 1
    converged = False
    while not converged:
        it += 1

        r1 = numpy.random.rand(popsize, ndim)
        X, gbest, pbestfit, gfit, pfit, status = de_iter(it, X, U, gbest, pbestfit, gfit, pfit, F, CR, r1, maxiter, xtol, ftol, fun, mut, cons)

        xall[:, :, it - 1] = numpy.copy(X)
        funall[:, it - 1] = numpy.copy(pbestfit)

        converged = status is not None

    return OptimizeResult(
        x=gbest,
        success=status >= 0,
        status=status,
        message=messages[status],
        fun=gfit,
        nfev=it * popsize,
        nit=it,
        xall=xall[:, :, :it],
        funall=funall[:, :it],
    )


def delete_shuffle_sync(popsize):
    return numpy.transpose([
        delete_shuffle_async(i, popsize) for i in range(popsize)
    ])


def delete_shuffle_async(i, popsize):
    return numpy.random.permutation(numpy.delete(numpy.arange(popsize), i))


def de_sync(it, X, U, gbest, pbestfit, gfit, pfit, F, CR, r1, maxiter, xtol, ftol, fun, mut, cons):
    popsize, ndim = X.shape

    # Mutation
    V = mut(delete_shuffle_sync(popsize), F, X, gbest)

    # Recombination
    mask = numpy.zeros_like(r1, dtype=bool)
    irand = numpy.random.randint(ndim, size=popsize)
    for i in range(popsize):
        mask[i, irand[i]] = True

    U[:] = cons(numpy.where(numpy.logical_or(mask, r1 <= CR), V, X))

    # Selection
    gbest, gfit, pfit, status = selection_sync(it, U, gbest, X, pbestfit, maxiter, xtol, ftol, fun)

    return X, gbest, pbestfit, gfit, pfit, status


def de_async(it, X, U, gbest, pbestfit, gfit, pfit, F, CR, r1, maxiter, xtol, ftol, fun, mut, cons):
    popsize, ndim = X.shape

    for i in range(popsize):
        # Mutation
        V = mut(delete_shuffle_async(i, popsize), F, X, gbest)

        # Recombination
        mask = numpy.zeros(ndim, dtype=bool)
        irand = numpy.random.randint(ndim)
        mask[irand] = True
        U[i] = cons(numpy.where(numpy.logical_or(mask, r1[i] <= CR), V, X[i]))

        # Selection
        gbest, gfit, pfit[i], status = selection_async(it, U, gbest, gfit, X, pbestfit, maxiter, xtol, ftol, fun, i)

    # Stop if maximum iteration is reached
    if status is None and it >= maxiter:
        status = -1

    return X, gbest, pbestfit, gfit, pfit, status


register("de", minimize)
