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
    options={},
):
    options["gamma"] = None

    return cpso(fun, bounds, x0, args, options)


register("pso", minimize)
