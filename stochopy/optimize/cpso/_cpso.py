import numpy

from ._constraints import _constraints_map
from .._common import messages, lhs, parallelize, selection_sync, selection_async
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
    inertia=0.7298,
    cognitivity=1.49618,
    sociability=1.49618,
    competitivity=1.0,
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

    # CPSO parameters
    if not 0.0 <= inertia <= 1.0:
        raise ValueError()

    if not 0.0 <= cognitivity <= 4.0:
        raise ValueError()

    if not 0.0 <= sociability <= 4.0:
        raise ValueError()

    if competitivity is not None and not 0.0 <= competitivity <= 2.0:
        raise ValueError()

    if updating not in {"immediate", "deferred"}:
        raise ValueError()

    w = inertia
    c1 = cognitivity
    c2 = sociability
    gamma = competitivity
    sync = updating == "deferred"

    # Swarm maximum radius
    if gamma:
        delta = numpy.log(1.0 + 0.003 * popsize) / numpy.max((0.2, numpy.log(0.01 * maxiter)))

    # Constraints
    cons = _constraints_map[constraints](lower, upper, sync)

    # Synchronize
    pso_iter = pso_sync if sync else pso_async

    # Parallel
    fun = parallelize(fun, args, sync, workers)

    # Seed
    if seed is not None:
        numpy.random.seed(seed)

    # Initial population
    X = (
        x0
        if x0 is not None
        else lhs(popsize, ndim, bounds)
    )
    V = numpy.zeros((popsize, ndim))
    pbest = numpy.copy(X)

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
        r2 = numpy.random.rand(popsize, ndim)
        X, V, pbest, gbest, pbestfit, gfit, pfit, status = pso_iter(it, X, V, pbest, gbest, pbestfit, gfit, pfit, w, c1, c2, r1, r2, maxiter, xtol, ftol, fun, cons)

        xall[:, :, it - 1] = numpy.copy(X)
        funall[:, it - 1] = numpy.copy(pfit)

        converged = status is not None

        if not converged and gamma:
            X, V, pbest, pbestfit = restart(it, X, V, pbest, gbest, pbestfit, lower, upper, gamma, delta, maxiter)

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


def mutation(X, V, pbest, gbest, w, c1, c2, r1, r2, cons):
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
    X, V = cons(X, V)

    return X, V


def pso_sync(it, X, V, pbest, gbest, pbestfit, gfit, pfit, w, c1, c2, r1, r2, maxiter, xtol, ftol, fun, cons):
    # Mutation
    X, V = mutation(X, V, pbest, gbest, w, c1, c2, r1, r2, cons)

    # Selection
    gbest, gfit, pfit, status = selection_sync(it, X, gbest, pbest, pbestfit, maxiter, xtol, ftol, fun)

    return X, V, pbest, gbest, pbestfit, gfit, pfit, status


def pso_async(it, X, V, pbest, gbest, pbestfit, gfit, pfit, w, c1, c2, r1, r2, maxiter, xtol, ftol, fun, cons):
    for i in range(len(X)):
        # Mutation
        X[i], V[i] = mutation(X[i], V[i], pbest[i], gbest, w, c1, c2, r1[i], r2[i], cons)

        # Selection
        gbest, gfit, pfit[i], status = selection_async(it, X, gbest, gfit, pbest, pbestfit, maxiter, xtol, ftol, fun, i)
                    
    # Stop if maximum iteration is reached
    if status is None and it >= maxiter:
        status = -1

    return X, V, pbest, gbest, pbestfit, gfit, pfit, status


def restart(it, X, V, pbest, gbest, pbestfit, lower, upper, gamma, delta, maxiter):
    popsize, ndim = X.shape

    # Evaluate swarm size
    swarm_radius = numpy.max([
        numpy.linalg.norm(X[i] - gbest) for i in range(popsize)
    ])
    swarm_radius /= numpy.sqrt(4.0 * ndim)

    # Restart particles if swarm size is lower than threshold
    if swarm_radius < delta:
        inorm = it / maxiter
        nw = int((popsize - 1.0) / (1.0 + numpy.exp(1.0 / 0.09 * (inorm - gamma + 0.5))))
        
        # Reset positions, velocities and personal bests
        if nw > 0:
            idx = pbestfit.argsort()[:-nw - 1:-1]
            V[idx] = numpy.zeros((nw, ndim))
            X[idx] = numpy.random.uniform(lower, upper, (nw, ndim))
            pbest[idx] = numpy.copy(X[idx])
            pbestfit[idx] = numpy.full(nw, 1.0e30)

    return X, V, pbest, pbestfit


register("cpso", minimize)
