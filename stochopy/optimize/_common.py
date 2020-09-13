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
            raise ValueError("cannot run asynchronous mode in parallel")

        else:
            def wrapper(x):
                return numpy.asarray(fun(x, *args))

    return wrapper
