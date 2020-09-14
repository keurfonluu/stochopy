from .._common import BaseResult

__all__ = [
    "sample",
]


_sampler_map = {}


class SampleResult(BaseResult):
    pass


def register(name, sample):
    _sampler_map[name] = sample


def sample(fun, bounds, x0=None, args=(), method="mcmc", options=None):
    options = options if options else {}
    
    return _sampler_map[method](
        fun=fun,
        bounds=bounds,
        x0=x0,
        args=args,
        **options,
    )
