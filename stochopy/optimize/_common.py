from joblib import delayed, Parallel

import numpy


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


def run(optimizer, fun, args, sync, workers, backend, optargs=()):
    if workers not in {0, 1} and backend == "joblib":
        with Parallel(n_jobs=workers, prefer="threads") as parallel:
            fun = wrapfun(fun, args, sync, backend, parallel)
            res = optimizer(fun, *optargs)

    else:
        parallel = backend == "mpi"
        fun = wrapfun(fun, args, sync, backend, parallel)
        res = optimizer(fun, *optargs)

    return res


def wrapfun(fun, args, sync, backend, parallel):
    if sync:
        if parallel:
            if backend == "joblib":
                fun = delayed(fun)

                def wrapper(x):
                    f = parallel(fun(xx, *args) for xx in x)
                    return numpy.array(f)

            elif backend == "mpi":
                raise NotImplementedError()

            else:
                raise ValueError(f"unknown backend '{backend}'")

        else:
            def wrapper(x):
                return numpy.array([fun(xx, *args) for xx in x])

    else:
        def wrapper(x):
            return fun(x, *args)

    return wrapper


def lhs(popsize, ndim, bounds=None):
    x = numpy.random.uniform(size=(popsize, ndim)) / popsize
    x += numpy.linspace(-1.0, 1.0, popsize, endpoint=False)[:, None]
    pop = numpy.transpose([
        x[numpy.random.permutation(popsize), i] for i in range(ndim)
    ])

    if bounds is not None:
        lower, upper = numpy.transpose(bounds)
        pop *= 0.5 * (upper - lower)
        pop += 0.5 * (upper + lower)

    return pop


def selection_sync(it, cand, xbest, x, xfun, maxiter, xtol, ftol, fun):
    # Selection
    candfun = fun(cand)
    idx = candfun < xfun
    xfun[idx] = candfun[idx].copy()
    x[idx] = cand[idx].copy()

    # Best solution index
    idx = numpy.argmin(xfun)

    # Stop if best solution changes less than xtol
    cond1 = numpy.linalg.norm(xbest - x[idx]) <= xtol
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
    # Selection
    candfun = fun(cand[i])

    status = None
    if candfun <= xfun[i]:
        x[i] = cand[i].copy()
        xfun[i] = candfun
        
        # Update best individual
        if candfun <= xbestfun:
            # Stop if best solution changes less than xtol
            cond1 = numpy.linalg.norm(xbest - cand[i]) <= xtol
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
