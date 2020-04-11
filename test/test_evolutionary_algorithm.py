import numpy
import pytest

from stochopy import Evolutionary


@pytest.mark.parametrize(
    "solver, solver_kws, xopt_ref",
    [
        ("pso", {"w": 0.42, "c1": 1.409, "c2": 1.991}, [0.70242052, 0.49260076]),
        ("cpso", {"w": 0.42, "c1": 1.409, "c2": 1.991, "gamma": 0.8}, [0.55554141, 0.30918171]),
        ("de", {"CR": 0.42, "F": 1.491}, [1.35183858, 1.81825907]),
        ("cmaes", {"sigma": 0.1, "mu_perc": 0.2, "xstart": [-3.0, -3.0]}, [0.80575841, 0.649243]),
        ("vdcma", {"sigma": 0.1, "mu_perc": 0.2, "xstart": [-3.0, -3.0]}, [1.38032658, 1.89976049]),
    ],
)
def test_evolutionary(solver, solver_kws, xopt_ref):
    ea = Evolutionary(
        func=lambda x: 100.0 * numpy.sum((x[1:] - x[:-1]**2)**2) + numpy.sum((1.0 - x[:-1])**2),
        lower=numpy.full(2, -5.12),
        upper=numpy.full(2, 5.12),
        popsize=int(4 + numpy.floor(3.0 * numpy.log(2))),
        max_iter=50,
        random_state=42,
    )
    xopt, _ = ea.optimize(solver=solver, **solver_kws)

    assert numpy.allclose(xopt_ref, xopt)
