from .._common import BaseResult

__all__ = [
    "sample",
]


_sampler_map = {}


class SampleResult(BaseResult):
    """
    Represent the sampling result.

    Attributes
    ----------
    x : array_like
        The best solution sampled.
    fun : scalar
        The solution function value.
    nit : int
        Number of samples generated.

    Notes
    -----
    There may be additional attributes not listed above depending of the specific solver.

    """

    pass


def register(name, sample):
    """Register a new sampler."""
    _sampler_map[name] = sample


def sample(fun, bounds, x0=None, args=(), method="mcmc", options=None):
    """
    Sample the variable space of an objective function.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form `f(x, *args)`, where `x` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
    bounds : array_like
        Bounds for variables. `(min, max)` pairs for each element in `x`, defining the finite lower and upper bounds for the sampling argument of `fun`. It is required to have `len(bounds) == len(x)`. `len(bounds)` is used to determine the number of parameters in `x`.
    x0 : array_like or None, optional, default None
        Initial sample. Array of real elements of size (`ndim`,), where `ndim` is the number of independent variables.
    args : tuple, optional, default None
        Extra arguments passed to the objective function.
    method : str, optional, default 'mcmc'
        Type of sampler. Should be one of:
         - 'mcmc'
         - 'hmc'
    options : dict or None, optional, default None
        A dictionary of sampler options. All methods accept the following generic options:

        ..

            maxiter : int
                Total number of samples to generate.
            seed : int or None
                Seed for random number generator.
            return_all : bool
                Set to True to return an array of all the solutions at each iteration.


    Returns
    -------
    stochopy.sample.SampleResult
        The sampling result represented as a :class:`stochopy.sample.SampleResult`. Important attributes are:
        - `x`: the best sample array
        - `fun`: the best sample function value
        - `xall`: the samples array
        - 'funall`: the samples' function value array

    """
    options = options if options else {}

    return _sampler_map[method](fun=fun, bounds=bounds, x0=x0, args=args, **options)
