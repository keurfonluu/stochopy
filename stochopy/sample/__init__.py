from ._helpers import sample, SampleResult
from .hmc import sample as hmc
from .mcmc import sample as mcmc

__all__ = [
    "SampleResult",
    "sample",
    "hmc",
    "mcmc",
]
