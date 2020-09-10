from .._common import BaseResult

__all__ = [
    "sample",
]


_sampler_map = {}


class SampleResult(BaseResult):
    pass


def register(name, sample):
    _sampler_map[name] = sample


def sample(fun, bounds, args=(), method="mcmc", options=None):
    options = options if options else {}
    
    return _sampler_map[method](
        fun=fun,
        bounds=bounds,
        args=args,
        options=options,
    )
