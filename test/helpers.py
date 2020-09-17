import numpy

import stochopy


def rosenbrock():
    fun = stochopy.factory.rosenbrock
    bounds = [[-5.12, 5.12], [-5.12, 5.12]]

    return fun, bounds


def optimize(method, options, xref):
    options.update({
        "maxiter": 128,
        "popsize": 8,
        "seed": 42,
        "return_all": True,
    })
    fun, bounds = rosenbrock()
    x = stochopy.optimize.minimize(fun, bounds, options=options, method=method)

    assert numpy.allclose(xref, x.x)
    if options["constraints"] is not None:
        assert numpy.all(x.xall + 1.0e-15 >= -5.12)
        assert numpy.all(x.xall - 1.0e-15 <= 5.12)


def optimize_parallel(method, options, xref):
    # Serial
    optimize(method, options, xref)

    # Parallel
    if "updating" in options and options["updating"] == "deferred":
        for backend in ["threading", "mpi"]:
            options.update({
                "workers": 2,
                "backend": backend,
            })
            optimize(method, options, xref)


def sample(method, options, xref):
    options.update({"seed": 42})
    fun, bounds = rosenbrock()
    x = stochopy.sample.sample(fun, bounds, options=options, method=method)

    assert numpy.allclose(xref, x.x)
