from .. import cpso
from .._helpers import register

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
    stochopy.optimize.OptimizeResult
        The optimization result represented as a :class:`stochopy.optimize.OptimizeResult`. Important attributes are:
        - `x`: the solution array
        - `fun`: the solution function value
        - `success`: a Boolean flag indicating if the optimizer exited successfully
        - `message`: a string which describes the cause of the termination

    References
    ----------
    .. [1] J. Kennedy and R. Eberhart, *Particle swarm optimization*, Proceedings of ICNN'95 - International Conference on Neural Networks, 1995, 4: 1942-1948
    .. [2] F. Van Den Bergh, *An analysis of particle swarm optimizers*, University of Pretoria, 2001

    """
    competitivity = None

    return cpso(
        fun,
        bounds,
        x0,
        args,
        maxiter,
        popsize,
        inertia,
        cognitivity,
        sociability,
        competitivity,
        seed,
        xtol,
        ftol,
        constraints,
        updating,
        workers,
        backend,
        return_all,
    )


register("pso", minimize)
