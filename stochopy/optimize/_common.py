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


def parallelize(fun, args, sync, parallel):
    if sync:
        if parallel:
            raise NotImplementedError()

        else:
            def wrapper(x):
                return numpy.array([fun(xx, *args) for xx in x])

    else:
        if parallel:
            raise ValueError("cannot perform asynchronous optimization in parallel")

        else:
            def wrapper(x):
                return fun(x, *args)

    return wrapper


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
