from .._helpers import register
from .. import cpso

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
    xtol=1.0e-8,
    ftol=1.0e-8,
    constraints=None,
    updating="deferred",
    workers=1,
):
    competitivity = None
    
    return cpso(fun, bounds, x0, args, maxiter, popsize, inertia, cognitivity, sociability, competitivity, xtol, ftol, constraints, updating, workers)


register("pso", minimize)
