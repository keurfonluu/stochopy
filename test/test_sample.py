import helpers
import numpy as np
import pytest

from stochopy.factory import rosenbrock
from stochopy.sample import sample


@pytest.mark.parametrize("options, xref", [({}, [-1.6315912, 2.60020735])])
def test_mcmc(options, xref):
    options.update({"stepsize": 0.1})
    helpers.sample("mcmc", options, xref)


@pytest.mark.parametrize("options, xref", [({}, [-1.28470918, 4.6153145])])
def test_hmc(options, xref):
    options.update({"nleap": 10, "stepsize": 0.1})
    helpers.sample("hmc", options, xref)


@pytest.mark.parametrize("method", ["mcmc", "hmc"])
def test_callback(method):
    global count
    count = 0

    def callback(X, state):
        global count
        count += 1

    maxiter = np.random.randint(2, 10)
    _ = sample(
        rosenbrock,
        [[-5.12, 5.12]] * 2,
        method=method,
        options={"maxiter": maxiter},
        callback=callback,
    )
    assert count == maxiter
