import numpy

from .._common import lhs, messages, run, selection_async, selection_sync
from .._helpers import OptimizeResult, register
from ._constraints import _constraints_map

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
    updating="immediate",
    workers=1,
    backend=None,
    return_all=False,
):
    """
    Minimize an objective function using Competitive Particle Swarm Optimization (CPSO).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form `f(x, *args)`, where `x` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
    bounds : array_like
        Bounds for variables. `(min, max)` pairs for each element in `x`, defining the finite lower and upper bounds for the optimizing argument of `fun`. It is required to have `len(bounds) == len(x)`. `len(bounds)` is used to determine the number of parameters in `x`.
    x0 : array_like or None, optional, default None
        Initial population. Array of real elements with shape (`popsize`, `ndim`), where `ndim` is the number of independent variables. If `x0` is not specified, the population is initialized using Latin Hypercube sampling.
    args : tuple, optional, default None
        Extra arguments passed to the objective function.
    maxiter : int, optional, default 100
        The maximum number of generations over which the entire population is evolved.
    popsize : int, optional, default 10
        Total population size.
    inertia : scalar, optional, default 0.7298
        Inertial weight, denoted by w in the literature. It should be in the range [0, 1].
    cognitivity : scalar, optional, default 1.49618
        Cognition parameter, denoted by c1 in the literature. It should be in the range [0, 4].
    sociability: scalar, optional, default 1.49618
        Sociability parameter, denoted by c2 in the literature. It should be in the range [0, 4].
    competitivity : scalar, optional, default 1.
        Competitivity parameter, denoted by gamma in the literature. It should be in the range [0, 2].
    seed : int or None, optional, default None
        Seed for random number generator.
    xtol : scalar, optional, default 1.0e-8
        Solution tolerance for termination.
    ftol : scalar, optional, default 1.0e-8
        Objective function value tolerance for termination.
    constraints : str or None, optional, default None
        Constraints definition:
         - None: no constraint
         - 'Shrink': infeasible solutions are repaired by shrinking particles' velocity vector
    updating : str {'immediate', 'deferred'}, optional, default 'immediate'
        If `'immediate'`, the best solution vector is continuously updated within a single generation. This can lead to faster convergence as candidate solutions can take advantage of continuous improvements in the best solution. With `'deferred'`, the best solution vector is updated once per generation. Only `'deferred'` is compatible with parallelization, and is overridden when `workers` is not `0` or `1` or `backend == 'mpi'`.
    workers : int, optional, default 1
        The population is subdivided into workers sections and evaluated in parallel (uses :class:`joblib.Parallel`). Supply -1 to use all available CPU cores.
    backend : str {'loky', 'threading', 'mpi'}, optional, default 'threading'
        Parallel backend to use when `workers` is not `0` or `1`:
         - 'loky': disable threading
         - 'threading': enable threading
         - 'mpi': use MPI (uses :mod:`mpi4py`)
    return_all : bool, optional, default False
        Set to True to return an array with shape (nit, popsize, ndim) of all the solutions at each iteration.

    Returns
    -------
    :class:`stochopy.optimize.OptimizeResult`
        The optimization result represented as a :class:`stochopy.optimize.OptimizeResult`. Important attributes are:

         - `x`: the solution array
         - `fun`: the solution function value
         - `success`: a Boolean flag indicating if the optimizer exited successfully
         - `message`: a string which describes the cause of the termination

    References
    ----------
    .. [1] J. Kennedy and R. Eberhart, *Particle swarm optimization*, Proceedings of ICNN'95 - International Conference on Neural Networks, 1995, 4: 1942-1948
    .. [2] F. Van Den Bergh, *An analysis of particle swarm optimizers*, University of Pretoria, 2001
    .. [3] K. Luu, M. Noble, A. Gesret, N. Belayouni and P.-F. Roux, *A parallel competitive Particle Swarm Optimization for non-linear first arrival traveltime tomography and uncertainty quantification*, Computers & Geosciences, 2018, 113: 81-93

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

    # Synchronize
    sync = updating == "deferred"
    sync = sync or workers not in {0, 1}
    sync = sync or backend == "mpi"

    # Seed
    if seed is not None:
        numpy.random.seed(seed)

    # Run in serial or parallel
    optargs = (
        bounds,
        x0,
        maxiter,
        popsize,
        w,
        c1,
        c2,
        gamma,
        constraints,
        sync,
        xtol,
        ftol,
        return_all,
    )
    res = run(cpso, fun, args, sync, workers, backend, optargs)

    return res


def cpso(
    fun,
    bounds,
    x0,
    maxiter,
    popsize,
    w,
    c1,
    c2,
    gamma,
    constraints,
    sync,
    xtol,
    ftol,
    return_all,
):
    """Optimize with CPSO."""
    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Constraints
    cons = _constraints_map[constraints](lower, upper, sync)

    # Iteration
    pso_iter = pso_sync if sync else pso_async

    # Swarm maximum radius
    if gamma:
        delta = numpy.log(1.0 + 0.003 * popsize) / numpy.max(
            (0.2, numpy.log(0.01 * maxiter))
        )

    # Initial population
    X = x0 if x0 is not None else lhs(popsize, ndim, bounds)
    V = numpy.zeros((popsize, ndim))
    pbest = X.copy()

    # Evaluate initial population
    pfit = fun(X)
    pbestfit = pfit.copy()

    # Initial best solution
    gbidx = numpy.argmin(pbestfit)
    gfit = pbestfit[gbidx]
    gbest = X[gbidx].copy()

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

        r1 = numpy.random.rand(popsize, ndim)
        r2 = numpy.random.rand(popsize, ndim)
        X, V, pbest, gbest, pbestfit, gfit, pfit, status = pso_iter(
            it,
            X,
            V,
            pbest,
            gbest,
            pbestfit,
            gfit,
            pfit,
            w,
            c1,
            c2,
            r1,
            r2,
            maxiter,
            xtol,
            ftol,
            fun,
            cons,
        )

        if return_all:
            xall[it - 1] = X.copy()
            funall[it - 1] = pfit.copy()

        converged = status is not None

        if not converged and gamma:
            X, V, pbest, pbestfit = restart(
                it, X, V, pbest, gbest, pbestfit, lower, upper, gamma, delta, maxiter
            )

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


def mutation(X, V, pbest, gbest, w, c1, c2, r1, r2, cons):
    """Update position and velocity vectors."""
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
    X, V = cons(X, V)

    return X, V


def pso_sync(
    it,
    X,
    V,
    pbest,
    gbest,
    pbestfit,
    gfit,
    pfit,
    w,
    c1,
    c2,
    r1,
    r2,
    maxiter,
    xtol,
    ftol,
    fun,
    cons,
):
    """Synchronous PSO."""
    # Mutation
    X, V = mutation(X, V, pbest, gbest, w, c1, c2, r1, r2, cons)

    # Selection
    gbest, gfit, pfit, status = selection_sync(
        it, X, gbest, pbest, pbestfit, maxiter, xtol, ftol, fun
    )

    return X, V, pbest, gbest, pbestfit, gfit, pfit, status


def pso_async(
    it,
    X,
    V,
    pbest,
    gbest,
    pbestfit,
    gfit,
    pfit,
    w,
    c1,
    c2,
    r1,
    r2,
    maxiter,
    xtol,
    ftol,
    fun,
    cons,
):
    """Asynchronous PSO."""
    popsize = len(X)

    for i in range(popsize):
        # Mutation
        X[i], V[i] = mutation(
            X[i], V[i], pbest[i], gbest, w, c1, c2, r1[i], r2[i], cons
        )

        # Selection
        gbest, gfit, pfit[i], status = selection_async(
            it, X, gbest, gfit, pbest, pbestfit, maxiter, xtol, ftol, fun, i
        )

    # Stop if maximum iteration is reached
    if status is None and it >= maxiter:
        status = -1

    return X, V, pbest, gbest, pbestfit, gfit, pfit, status


def restart(it, X, V, pbest, gbest, pbestfit, lower, upper, gamma, delta, maxiter):
    """Competitive PSO algorithm."""
    popsize, ndim = X.shape

    # Evaluate swarm size
    swarm_radius = numpy.max([numpy.linalg.norm(X[i] - gbest) for i in range(popsize)])
    swarm_radius /= numpy.sqrt(4.0 * ndim)

    # Restart particles if swarm size is lower than threshold
    if swarm_radius < delta:
        inorm = it / maxiter
        nw = int(
            (popsize - 1.0) / (1.0 + numpy.exp(1.0 / 0.09 * (inorm - gamma + 0.5)))
        )

        # Reset positions, velocities and personal bests
        if nw > 0:
            idx = pbestfit.argsort()[: -nw - 1 : -1]
            V[idx] = numpy.zeros((nw, ndim))
            X[idx] = numpy.random.uniform(lower, upper, (nw, ndim))
            pbest[idx] = X[idx].copy()
            pbestfit[idx] = numpy.full(nw, 1.0e30)

    return X, V, pbest, pbestfit


register("cpso", minimize)
