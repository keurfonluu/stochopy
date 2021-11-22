from functools import wraps

import numpy as np
from joblib import Parallel, delayed

prefer = {
    "loky": "processes",
    "threading": "threads",
}


# Failed if < 0, success otherwise
messages = {
    -8: "TolX",
    -7: "TolFun",
    -6: "TolXUp",
    -5: "EqualFunValues",
    -4: "ConditionCov",
    -3: "NoEffectCoord",
    -2: "NoEffectAxis",
    -1: "maximum number of iterations is reached",
    0: "best solution changes less than xtol",
    1: "best solution value is lower than ftol",
}


def optimizer(optfun):
    """Decorate an optimizer."""

    @wraps(optfun)
    def decorator(fun, args, sync, workers, backend, *optargs):
        """Run an optimizer with selected backend."""

        def wrapper(fun, args, sync, backend, parallel):
            """Wrap an objective function."""
            if sync:
                if parallel:
                    if backend in prefer:
                        fun = delayed(fun)

                        def wrapper(x):
                            f = parallel(fun(xx, *args) for xx in x)
                            return np.array(f)

                    elif backend == "mpi":
                        try:
                            from mpi4py import MPI
                        except ImportError:
                            raise ImportError(
                                "parallelization using MPI requires mpi4py to be installed"
                            )

                        mpi_comm = MPI.COMM_WORLD
                        mpi_rank = mpi_comm.Get_rank()
                        mpi_size = mpi_comm.Get_size()
                        mpi_double = MPI.DOUBLE

                        def wrapper(x):
                            popsize = len(x)
                            f = np.zeros(popsize)
                            fmpi = np.zeros(popsize)

                            mpi_comm.Bcast([x, mpi_double], root=0)
                            for i in np.arange(mpi_rank, popsize, mpi_size):
                                fmpi[i] = fun(x[i], *args)
                            mpi_comm.Barrier()

                            mpi_comm.Allreduce(
                                [fmpi, mpi_double], [f, mpi_double], op=MPI.SUM
                            )

                            return f

                    else:
                        raise ValueError(f"unknown backend '{backend}'")

                else:

                    def wrapper(x):
                        return np.array([fun(xx, *args) for xx in x])

            else:

                def wrapper(x):
                    if x.ndim == 2:
                        return np.array([fun(xx, *args) for xx in x])
                    else:
                        return fun(x, *args)

            return wrapper

        backend = backend if backend else "threading"

        if backend in prefer and workers not in {0, 1}:
            with Parallel(n_jobs=workers, prefer=prefer[backend]) as parallel:
                fun = wrapper(fun, args, sync, backend, parallel)
                res = optfun(fun, args, sync, workers, backend, *optargs)

        else:
            parallel = backend == "mpi"
            fun = wrapper(fun, args, sync, backend, parallel)
            res = optfun(fun, args, sync, workers, backend, *optargs)

        return res

    return decorator


def lhs(popsize, ndim, bounds=None):
    """Latin Hypercube sampling."""
    x = np.random.uniform(size=(popsize, ndim)) / popsize
    x += np.linspace(-1.0, 1.0, popsize, endpoint=False)[:, None]
    pop = np.transpose([x[np.random.permutation(popsize), i] for i in range(ndim)])

    if bounds is not None:
        lower, upper = np.transpose(bounds)
        pop *= 0.5 * (upper - lower)
        pop += 0.5 * (upper + lower)

    return pop


def selection_sync(it, cand, xbest, x, xfun, maxiter, xtol, ftol, fun):
    """Synchronous selection."""
    # Selection
    candfun = fun(cand)
    idx = candfun < xfun
    xfun[idx] = candfun[idx].copy()
    x[idx] = cand[idx].copy()

    # Best solution index
    idx = np.argmin(xfun)

    # Stop if best solution changes less than xtol
    cond1 = np.linalg.norm(xbest - x[idx]) <= xtol
    cond2 = xfun[idx] <= ftol
    if cond1 and cond2:
        xbest = x[idx].copy()
        xbestfun = xfun[idx]
        status = 0

    # Stop if best solution value is less than ftol
    elif xfun[idx] <= ftol:
        xbest = x[idx].copy()
        xbestfun = xfun[idx]
        status = 1

    # Stop if maximum iteration is reached
    elif it >= maxiter:
        xbest = x[idx].copy()
        xbestfun = xfun[idx]
        status = -1

    # Otherwise, update best solution
    else:
        xbest = x[idx].copy()
        xbestfun = xfun[idx]
        status = None

    return xbest, xbestfun, candfun, status


def selection_async(it, cand, xbest, xbestfun, x, xfun, maxiter, xtol, ftol, fun, i):
    """Asynchronous selection."""
    # Selection
    candfun = fun(cand[i])

    status = None
    if candfun <= xfun[i]:
        x[i] = cand[i].copy()
        xfun[i] = candfun

        # Update best individual
        if candfun <= xbestfun:
            # Stop if best solution changes less than xtol
            cond1 = np.linalg.norm(xbest - cand[i]) <= xtol
            cond2 = candfun <= ftol
            if cond1 and cond2:
                xbest = cand[i].copy()
                xbestfun = candfun
                status = 0

            # Stop if best solution value is less than ftol
            elif candfun <= ftol:
                xbest = cand[i].copy()
                xbestfun = candfun
                status = 1

            # Otherwise, update best solution
            else:
                xbest = cand[i].copy()
                xbestfun = candfun

    return xbest, xbestfun, candfun, status
