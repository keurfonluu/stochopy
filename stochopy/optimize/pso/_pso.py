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
    w=0.7298,
    c1=1.49618,
    c2=1.49618,
    xtol=1.0e-8,
    ftol=1.0e-8,
    constraints=False,
    sync=True,
    parallel=False,
):
    gamma = None
    
    return cpso(fun, bounds, x0, args, maxiter, popsize, w, c1, c2, gamma, xtol, ftol, constraints, sync, parallel)


register("pso", minimize)
