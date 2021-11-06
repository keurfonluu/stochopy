from .._common import BaseResult

__all__ = [
    "minimize",
]


_optimizer_map = {}


class OptimizeResult(BaseResult):
    """
    Represent the optimization result.

    Attributes
    ----------
    x : array_like
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the underlying solver. Refer to message for details.
    message : str
        Description of the cause of the termination.
    fun : scalar
        The solution function value.
    nit : int
        Number of iterations performed by the optimizer.

    Notes
    -----
    There may be additional attributes not listed above depending of the specific solver.

    """

    pass


def register(name, minimize):
    """Register a new optimizer."""
    _optimizer_map[name] = minimize


def minimize(fun, bounds, x0=None, args=(), method="de", options=None):
    """
    Minimize an objective function using a stochastic algorithm.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.
    bounds : array_like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    x0 : array_like or None, optional, default None
        Initial guess. Depending on the solver, array of real elements of size (``ndim``,) or with shape (``popsize``, ``ndim``), where ``ndim`` is the number of independent variables and ``popsize`` is the total population size if the solver is population-based.
    args : tuple, optional, default None
        Extra arguments passed to the objective function.
    method : str, optional, default 'de'
        Type of solver. Should be one of:

         - 'cmaes'
         - 'cpso'
         - 'de'
         - 'na'
         - 'pso'
         - 'vdcma'

    options : dict or None, optional, default None
        A dictionary of solver options. All methods accept the following generic options:

         - maxiter (int): maximum number of iterations to perform
         - seed (int or None): seed for random number generator
         - return_all (bool): set to True to return an array of all the solutions at each iteration.

    Returns
    -------
    :class:`stochopy.optimize.OptimizeResult`
        The optimization result represented as a :class:`stochopy.optimize.OptimizeResult`. Important attributes are:

         - ``x``: the solution array
         - ``fun``: the solution function value
         - ``success``: a Boolean flag indicating if the optimizer exited successfully
         - ``message``: a string which describes the cause of the termination

    """
    options = options if options else {}

    return _optimizer_map[method](fun=fun, bounds=bounds, x0=x0, args=args, **options)
