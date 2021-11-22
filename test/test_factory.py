import numpy as np
import pytest

import stochopy


@pytest.mark.parametrize(
    "fun, fref",
    [
        (stochopy.factory.ackley, 3.625384938440362),
        (stochopy.factory.griewank, 0.8067591547236139),
        (stochopy.factory.quartic, 55.0),
        (stochopy.factory.rastrigin, 10.0),
        (stochopy.factory.rosenbrock, 0.0),
        (stochopy.factory.sphere, 10.0),
        (stochopy.factory.styblinski_tang, 341.6599),
    ],
)
def test_factory(fun, fref):
    x = np.ones(10)
    f = fun(x)

    assert np.allclose(fref, f)
