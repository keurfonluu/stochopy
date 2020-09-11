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


def parallelize(fun, mpi):
    if mpi:
        raise NotImplementedError()

    else:
        def wrapper(x, *args, **kwargs):
            return numpy.array([fun(xx, *args, **kwargs) for xx in x])

    return wrapper
