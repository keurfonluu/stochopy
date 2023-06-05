import numpy as np

from .._common import messages, optimizer
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
    sigma=0.1,
    muperc=0.5,
    seed=None,
    xtol=1.0e-8,
    ftol=1.0e-8,
    constraints=None,
    workers=1,
    backend=None,
    return_all=False,
    verbosity=1.0,
    callback=None,
):
    """
    Minimize an objective function using Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
    bounds : array_like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    x0 : array_like or None, optional, default None
        Initial mean. Array of real elements of size (``ndim``,), where ``ndim`` is the number of independent variables.
    args : tuple, optional, default None
        Extra arguments passed to the objective function.
    maxiter : int, optional, default 100
        The maximum number of generations over which the entire population is evolved.
    popsize : int, optional, default 10
        Total population size.
    sigma : scalar or array_like, optional, default 0.1
        Initial standard deviation (as a fraction of feasible space defined by ``bounds``).
    muperc : scalar, optional, default 0.5
        Number of parents (as a fraction of total population size).
    seed : int or None, optional, default None
        Seed for random number generator.
    xtol : scalar, optional, default 1.0e-8
        Solution tolerance for termination.
    ftol : scalar, optional, default 1.0e-8
        Objective function value tolerance for termination.
    constraints : str or None, optional, default None
        Constraints definition:

         - None: no constraint
         - 'Penalize': infeasible solutions are repaired and their function values are penalized

    workers : int, optional, default 1
        The population is subdivided into workers sections and evaluated in parallel (uses :class:`joblib.Parallel`). Supply -1 to use all available CPU cores.
    backend : str {'loky', 'threading', 'mpi'}, optional, default 'threading'
        Parallel backend to use when ``workers`` is not ``0`` or ``1``:

         - 'loky': disable threading
         - 'threading': enable threading
         - 'mpi': use MPI (uses :mod:`mpi4py`)

    return_all : bool, optional, default False
        Set to True to return an array with shape (``nit``, ``verbosity`` * ``popsize``, ``ndim``) of all the solutions at each iteration.
    verbosity : float, optional, default 1.0
        Fraction of population to consider in `return_all`. If 0.0, returns the best solution at each iteration.
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
    .. [1] N. Hansen, *The CMA evolution strategy: A tutorial*, Inria, Université Paris-Saclay, LRI, 2011, 102: 1-34

    """
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    # Dimensionality and search space
    if np.ndim(bounds) != 2:
        raise ValueError()

    # Initial guess x0
    if x0 is not None:
        if np.ndim(x0) != 1 or len(x0) != len(bounds):
            raise ValueError()

    # CMA-ES parameters
    if sigma <= 0.0:
        raise ValueError()

    if not 0.0 < muperc <= 1.0:
        raise ValueError()

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
        sigma,
        muperc,
        constraints,
        xtol,
        ftol,
        return_all,
        verbosity,
        callback,
    )
    res = cmaes(fun, args, True, workers, backend, *optargs)

    return res


@optimizer
def cmaes(
    funstd,
    args,
    sync,
    workers,
    backend,
    bounds,
    x0,
    maxiter,
    popsize,
    sigma,
    muperc,
    constraints,
    xtol,
    ftol,
    return_all,
    verbosity,
    callback,
):
    """Optimize with CMA-ES."""
    ndim = len(bounds)
    lower, upper = np.transpose(bounds)

    # Standardize and unstandardize
    xm = 0.5 * (upper + lower)
    xstd = 0.5 * (upper - lower)
    standardize = lambda x: (x - xm) / xstd
    unstandardize = lambda x: x * xstd + xm

    fun = lambda x: funstd(unstandardize(x))

    # Constraints
    if constraints is not None:
        cons = _constraints_map[constraints]

    # Initial mean
    xmean = np.random.uniform(-1.0, 1.0, ndim) if x0 is None else standardize(x0)
    xold = np.empty(ndim)

    # Number of parents
    mu = int(muperc * popsize)

    # Strategy parameter setting: Selection
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = weights.sum() ** 2 / np.square(weights).sum()

    # Strategy parameter setting: Adaptation
    cc = (4.0 + mueff / ndim) / (ndim + 4.0 + 2.0 * mueff / ndim)
    cs = (mueff + 2.0) / (ndim + mueff + 5.0)
    c1 = 2.0 / ((ndim + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((ndim + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (ndim + 1.0)) - 1.0) + cs

    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(ndim)
    ps = np.zeros(ndim)
    B = np.eye(ndim)
    D = np.ones(ndim)
    C = np.eye(ndim)
    invsqrtC = np.eye(ndim)
    chind = np.sqrt(ndim) * (1.0 - 1.0 / (4.0 * ndim) + 1.0 / (21.0 * ndim ** 2))

    # Initialize boundaries weights
    bnd_weights = np.zeros(ndim)
    dfithist = np.ones(1)

    # Initialize arrays
    if return_all:
        nout = int(np.ceil(verbosity * popsize))
        xall = np.empty((maxiter, max(1, nout), ndim))
        funall = np.empty((maxiter, max(1, nout)))

    # (mu, lambda)-CMA-ES
    nfev = 0
    eigeneval = 0
    arbestfitness = np.zeros(maxiter)
    ilim = int(10.0 + 30.0 * ndim / popsize)
    insigma = sigma
    validfitval = False
    iniphase = True

    it = 0
    converged = False
    while not converged:
        it += 1

        # Generate lambda offsprings
        arx = np.array(
            [
                xmean + sigma * np.dot(B, D * np.random.randn(ndim))
                for i in range(popsize)
            ]
        )
        arxvalid = arx.copy()

        # Evaluate fitness
        if constraints == "Penalize":
            arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase = cons(
                arxvalid,
                arx,
                xmean,
                xold,
                sigma,
                np.diag(C),
                mueff,
                it,
                bnd_weights,
                dfithist,
                validfitval,
                iniphase,
                fun,
            )
        else:
            arfitness = fun(arxvalid)
        nfev += popsize

        if return_all:
            if nout > 0:
                xall[it - 1] = unstandardize(arxvalid[:nout])
                funall[it - 1] = arfitness[:nout].copy()

            else:
                idx = arfitness.argmin()
                xall[it - 1] = unstandardize(arxvalid[idx])
                funall[it - 1] = arfitness[idx].copy()

        # Sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(arfitness)
        xold = xmean.copy()
        xmean = np.dot(weights, arx[arindex[:mu], :])

        # Save best fitness
        arbestfitness[it - 1] = arfitness[arindex[0]].copy()

        # Cumulation
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * np.dot(
            invsqrtC, xmean - xold
        ) / sigma
        cond = np.linalg.norm(ps) / np.sqrt(
            1.0 - (1.0 - cs) ** (2.0 * nfev / popsize)
        ) / chind < 1.4 + 2.0 / (ndim + 1.0)
        pc *= 1.0 - cc
        pc += np.sqrt(cc * (2.0 - cc) * mueff) * (xmean - xold) / sigma if cond else 0.0

        # Adapt covariance matrix C
        artmp = (arx[arindex[:mu], :] - np.tile(xold, (mu, 1))) / sigma
        tmp = 0.0 if cond else c1 * cc * (2.0 - cc) * C
        C *= 1.0 - c1 - cmu
        C += cmu * np.dot(np.dot(artmp.T, np.diag(weights)), artmp)
        C += c1 * np.outer(pc, pc)
        C += tmp

        # Adapt step size sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chind - 1.0))

        # Diagonalization of C
        if nfev - eigeneval > popsize / (c1 + cmu) / ndim / 10.0:
            eigeneval = nfev
            C = np.triu(C) + np.triu(C, 1).T
            D, B = np.linalg.eigh(C)
            idx = np.argsort(D)
            D = D[idx]
            B = B[:, idx]
            D = np.sqrt(D)
            invsqrtC = np.dot(np.dot(B, np.diag(1.0 / D)), B.T)

        # Check convergence
        status = converge(
            it,
            ndim,
            maxiter,
            xmean,
            xold,
            arbestfitness,
            arfitness,
            arindex,
            sigma,
            insigma,
            ilim,
            pc,
            xtol,
            ftol,
            np.diag(C),
            B,
            D,
        )
        converged = status is not None

        if callback is not None:
            res = OptimizeResult(
                x=unstandardize(arxvalid[arindex[0]]),
                fun=arfitness[arindex[0]],
                nfev=nfev,
                nit=it,
            )
            if return_all:
                res.update({"xall": xall[:it], "funall": funall[:it]})

            callback(unstandardize(arxvalid), res)

    res = OptimizeResult(
        x=unstandardize(arxvalid[arindex[0]]),
        success=status >= 0,
        status=status,
        message=messages[status],
        fun=arfitness[arindex[0]],
        nfev=nfev,
        nit=it,
    )
    if return_all:
        res.update({"xall": xall[:it], "funall": funall[:it]})

    return res


def converge(
    it,
    ndim,
    maxiter,
    xmean,
    xold,
    arbestfitness,
    arfitness,
    arindex,
    sigma,
    insigma,
    ilim,
    pc,
    xtol,
    ftol,
    diagC,
    B=None,
    D=None,
):
    """Check convergence status at the end of an iteration."""
    status = None
    i = int(np.floor(np.mod(it, ndim)))
    sqdiagC = np.sqrt(diagC)

    # Stop if maximum iteration is reached
    if it >= maxiter:
        status = -1

    # Stop if mean position changes less than xtol
    elif np.linalg.norm(xold - xmean) <= xtol and arfitness[arindex[0]] < ftol:
        status = 0

    # Stop if fitness is less than ftol
    elif arfitness[arindex[0]] <= ftol:
        status = 1

    # NoEffectAxis: stop if numerical precision problem
    elif B is not None and (np.abs(0.1 * sigma * B[:, i] * D[i]) < 1.0e-10).all():
        status = -2

    # NoEffectCoord: stop if too low coordinate axis deviations
    elif (0.2 * sigma * sqdiagC < 1.0e-10).any():
        status = -3

    # ConditionCov: stop if the condition number exceeds 1e14
    elif D is not None and D.max() > 1.0e7 * D.min():
        status = -4

    # EqualFunValues: stop if the range of fitness values is zero
    elif (
        it >= ilim
        and arbestfitness[it - ilim : it + 1].max()
        - arbestfitness[it - ilim : it + 1].min()
        < 1.0e-10
    ):
        status = -5

    # TolXUp: stop if x-changes larger than 1e3 times initial sigma
    elif (sigma * sqdiagC > 1.0e3 * insigma).any():
        status = -6

    # TolFun: stop if fun-changes smaller than 1e-12
    elif (
        it > 2
        and np.append(arfitness, arbestfitness).max()
        - np.append(arfitness, arbestfitness).min()
        < 1.0e-12
    ):
        status = -7

    # TolX: stop if x-changes smaller than 1e-11 times initial sigma
    elif (sigma * np.append(np.abs(pc), sqdiagC.max()) < 1.0e-11 * insigma).all():
        status = -8

    return status


register("cmaes", minimize)
