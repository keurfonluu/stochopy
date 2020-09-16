import numpy

from .._common import in_search_space
from .._helpers import register, SampleResult

__all__ = [
    "sample",
]


def sample(
    fun,
    bounds,
    x0=None,
    args=(),
    maxiter=100,
    nleap=10,
    stepsize=0.01,
    seed=None,
    jac=None,
    finite_diff_abs_step=1.0e-4,
    constraints=None,
    return_all=False,
):
    # Cost function
    if not hasattr(fun, "__call__"):
        raise TypeError()

    fun = count(fun)  # Wrap to count the number of function evaluations

    # Dimensionality and search space
    if numpy.ndim(bounds) != 2:
        raise ValueError()

    ndim = len(bounds)
    lower, upper = numpy.transpose(bounds)

    # Initial guess x0
    if x0 is not None and len(x0) != ndim:
        raise ValueError()

    # Number of leap-frog steps
    if nleap < 1:
        raise ValueError()

    # Step size
    if numpy.ndim(stepsize) == 0:
        stepsize = numpy.full(ndim, stepsize)

    if len(stepsize) != ndim:
        raise ValueError()

    stepsize *= 0.5 * (upper - lower)

    # Jacobian
    if not (jac is None or hasattr(jac, "__call__")):
        raise TypeError()

    if jac is None:
        jac = lambda x: numerical_gradient(x, fun, args, finite_diff_abs_step)
    else:
        jac = lambda x: jac(x, *args)

    # Seed
    if seed is not None:
        numpy.random.seed(seed)

    # Initialize arrays
    xall = numpy.empty((maxiter, ndim))
    funall = numpy.empty(maxiter)
    xall[0] = x0 if x0 is not None else numpy.random.uniform(lower, upper)
    funall[0] = fun(xall[0], *args)

    # Leap-frog algorithm
    n_accepted = 0
    for i in range(1, maxiter):
        q = numpy.copy(xall[i - 1])
        p = numpy.random.randn(ndim)  # Random momentum
        q0 = numpy.copy(q)
        p0 = numpy.copy(p)
        
        p -= 0.5 * stepsize * jac(q)  # First half momentum step
        q += stepsize * p             # First full position step
        for _ in range(nleap):
            p -= stepsize * jac(q)  # Momentum
            q += stepsize * p  # Position
        p -= 0.5 * stepsize * jac(q)  # Last half momentum step

        accept = False
        if in_search_space(q, lower, upper, constraints):
            U0 = fun(q0, *args)
            K0 = 0.5 * numpy.square(p0).sum()
            U = fun(q, *args)
            K = 0.5 * numpy.square(p).sum()

            log_alpha = min(0.0, U0 - U + K0 - K)
            accept = log_alpha > numpy.log(numpy.random.rand())

        if accept:
            n_accepted += 1
            xall[i] = q
            funall[i] = U
        else:
            xall[i] = xall[i - 1]
            funall[i] = funall[i - 1]

    idx = numpy.argmin(funall)
    res = SampleResult(
        x=xall[idx],
        fun=funall[idx],
        nfev=nfev,
        nit=maxiter,
        accept_ratio=n_accepted / maxiter,
    )
    if return_all:
        res["xall"] = xall
        res["funall"] = funall

    return res


def count(fun):
    global nfev
    nfev = 0
    
    def wrapper(*args, **kwargs):
        global nfev
        nfev += 1

        return fun(*args, **kwargs)

    return wrapper


def numerical_gradient(x, fun, args, finite_diff_abs_step):
    ndim = len(x)
    x1 = numpy.copy(x)
    x2 = numpy.copy(x)

    grad = numpy.empty(ndim)
    for i in range(ndim):
        x1[i] -= finite_diff_abs_step
        x2[i] += finite_diff_abs_step

        grad[i] = fun(x2, *args) - fun(x1, *args)

        x1[i] += finite_diff_abs_step
        x2[i] -= finite_diff_abs_step

    return 0.5 * grad / finite_diff_abs_step


register("hmc", sample)
