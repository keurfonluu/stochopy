import numpy

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
    options={},
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

    # Options
    _options = {
        "maxiter": 100,
        "popsize": 10,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
        "gamma": 1.0,
        "xtol": 1.0e-8,
        "ftol": 1.0e-8,
        "constraints": False,
        "sync": True,
        "parallel": False,
    }
    _options.update(options)

    maxiter = _options["maxiter"]
    popsize = _options["popsize"]
    w = _options["w"]
    c1 = _options["c1"]
    c2 = _options["c2"]
    gamma = _options["gamma"]
    xtol = _options["xtol"]
    ftol = _options["ftol"]
    constraints = _options["constraints"]
    sync = _options["sync"]
    parallel = _options["parallel"]

    # Population size
    if popsize < 2:
        raise ValueError()

    if x0 is not None and len(x0) != popsize:
        raise ValueError()

    # PSO parameters
    if not 0.0 <= w <= 1.0:
        raise ValueError()

    if not 0.0 <= c1 <= 4.0:
        raise ValueError()

    if not 0.0 <= c2 <= 4.0:
        raise ValueError()

    if gamma is not None and not 0.0 <= gamma <= 2.0:
        raise ValueError()

    # Swarm maximum radius
    if gamma:
        delta = numpy.log(1.0 + 0.003 * popsize) / numpy.max((0.2, numpy.log(0.01 * maxiter)))

    # Constraints
    cons = constrain(constraints, lower, upper)

    # Synchronize
    pso_iter = pso_sync if sync else pso_async

    # Parallel
    fun = parallelize(fun, args, sync, parallel)

    # Initialize arrays
    xall = numpy.empty((popsize, ndim, maxiter))
    funall = numpy.empty((popsize, maxiter))

    # Initial population
    X = (
        x0
        if x0 is not None
        else numpy.random.uniform(lower, upper, (popsize, ndim)) 
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

    # Iterate until one of the termination criterion is satisfied
    it = 1
    converged = False
    while not converged:
        it += 1

        r1 = numpy.random.rand(popsize, ndim)
        r2 = numpy.random.rand(popsize, ndim)
        X, V, pbest, gbest, pbestfit, gfit, status = pso_iter(it, X, V, pbest, gbest, pbestfit, gfit, w, c1, c2, r1, r2, maxiter, xtol, ftol, fun, cons)

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
    

def constrain(constraints, lower, upper):
    if constraints:
        def shrink(X, V):
            Xold = X
            X = X + V

            maskl = X < lower
            masku = X > upper
            condl = maskl.any()
            condu = masku.any()

            if condl and condu:
                bl = ((Xold[maskl] - lower) / (Xold[maskl] - X[maskl])).min()
                bu = ((Xold[masku] - upper) / (Xold[masku] - X[masku])).min()
                beta = min(bl, bu)

            elif condl and not condu:
                beta = ((Xold[maskl] - lower) / (Xold[maskl] - X[maskl])).min()

            elif not condl and condu:
                beta = ((Xold[masku] - upper) / (Xold[masku] - X[masku])).min()

            else:
                return X

            return Xold + beta * (X - Xold)

        return shrink

    else:
        return lambda X, V: X + V


def mutation(X, V, pbest, gbest, w, c1, c2, r1, r2, cons):
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
    X = cons(X, V)

    return X, V


def pso_sync(it, X, V, pbest, gbest, pbestfit, gfit, w, c1, c2, r1, r2, maxiter, xtol, ftol, fun, cons):
    # Mutation
    X, V = mutation(X, V, pbest, gbest, w, c1, c2, r1, r2, cons)

    # Selection
    gbest, gfit, status = selection_sync(it, X, gbest, pbest, pbestfit, maxiter, xtol, ftol, fun)

    return X, V, pbest, gbest, pbestfit, gfit, status


def pso_async(it, X, V, pbest, gbest, pbestfit, gfit, w, c1, c2, r1, r2, maxiter, xtol, ftol, fun, cons):
    for i in range(len(X)):
        # Mutation
        X[i], V[i] = mutation(X[i], V[i], pbest[i], gbest, w, c1, c2, r1[i], r2[i], cons)

        # Selection
        gbest, gfit, status = selection_async(it, X, gbest, gfit, pbest, pbestfit, maxiter, xtol, ftol, fun, i)
                    
    # Stop if maximum iteration is reached
    if status is None and it >= maxiter:
        status = -1

    return X, V, pbest, gbest, pbestfit, gfit, status


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
