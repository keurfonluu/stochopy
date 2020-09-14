from .._common import BaseResult

__all__ = [
    "minimize",
]


_optimizer_map = {}


class OptimizeResult(BaseResult):
    pass


def register(name, minimize):
    _optimizer_map[name] = minimize


def minimize(fun, bounds, x0=None, args=(), method="de", options=None):
    options = options if options else {}
    
    return _optimizer_map[method](
        fun=fun,
        bounds=bounds,
        x0=x0,
        args=args,
        **options,
    )
