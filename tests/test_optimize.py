import helpers
import numpy as np
import pytest

from stochopy.factory import rosenbrock
from stochopy.optimize import minimize


@pytest.mark.parametrize(
    "options, xref",
    [
        ({"constraints": None}, [0.29967256, 0.0803311]),
        ({"x0": [-5.0, -5.0], "constraints": None}, [0.99998135, 0.99995618]),
        ({"constraints": "Penalize"}, [0.18765786, 0.05858025]),
        ({"x0": [-5.0, -5.0], "constraints": "Penalize"}, [0.99998135, 0.99995618]),
    ],
)
def test_cmaes(options, xref):
    options.update({"sigma": 0.1, "muperc": 0.5})
    helpers.optimize_parallel("cmaes", options, xref)


@pytest.mark.parametrize(
    "options, xref",
    [
        (
            {"inertia": 0.7298, "constraints": None, "updating": "deferred"},
            [0.95990315, 0.92082304],
        ),
        (
            {"inertia": 0.7298, "constraints": None, "updating": "immediate"},
            [0.93258856, 0.86919435],
        ),
        (
            {"inertia": 0.91, "constraints": "Shrink", "updating": "deferred"},
            [0.73752093, 0.54625484],
        ),
        (
            {"inertia": 0.91, "constraints": "Shrink", "updating": "immediate"},
            [0.76668308, 0.58381385],
        ),
    ],
)
def test_cpso(options, xref):
    options.update(
        {"cognitivity": 1.49618, "sociability": 1.49618, "competitivity": 1.0}
    )
    helpers.optimize_parallel("cpso", options, xref)


@pytest.mark.parametrize(
    "options, xref",
    [
        (
            {"strategy": "rand1bin", "constraints": None, "updating": "deferred"},
            [0.83228338, 0.68910339],
        ),
        (
            {"strategy": "rand2bin", "constraints": None, "updating": "deferred"},
            [0.79409325, 0.60743767],
        ),
        (
            {"strategy": "best1bin", "constraints": None, "updating": "deferred"},
            [1.00025932, 1.00051521],
        ),
        (
            {"strategy": "best2bin", "constraints": None, "updating": "deferred"},
            [1.00515037, 1.01055037],
        ),
        (
            {"strategy": "rand1bin", "constraints": None, "updating": "immediate"},
            [0.85658185, 0.726094],
        ),
        (
            {"strategy": "rand1bin", "constraints": "Random", "updating": "deferred"},
            [1.02340815, 1.04590782],
        ),
        (
            {"strategy": "rand1bin", "constraints": "Random", "updating": "immediate"},
            [0.99438151, 0.9944796],
        ),
    ],
)
def test_de(options, xref):
    options.update({"recombination": 0.1, "mutation": 0.5})
    helpers.optimize_parallel("de", options, xref)


@pytest.mark.parametrize("options, xref", [({}, [1.14849912, 1.31885465])])
def test_na(options, xref):
    options.update({"nrperc": 0.5})
    helpers.optimize_parallel("na", options, xref)


@pytest.mark.parametrize(
    "options, xref",
    [
        (
            {"inertia": 0.7298, "constraints": None, "updating": "deferred"},
            [0.95990315, 0.92082304],
        ),
        (
            {"inertia": 0.7298, "constraints": None, "updating": "immediate"},
            [0.95909508, 0.91977272],
        ),
        (
            {"inertia": 0.91, "constraints": "Shrink", "updating": "deferred"},
            [0.73752093, 0.54625484],
        ),
        (
            {"inertia": 0.91, "constraints": "Shrink", "updating": "immediate"},
            [0.76668308, 0.58381385],
        ),
    ],
)
def test_pso(options, xref):
    options.update({"cognitivity": 1.49618, "sociability": 1.49618})
    helpers.optimize_parallel("pso", options, xref)


@pytest.mark.parametrize(
    "options, xref",
    [
        ({"constraints": None}, [0.90013445, 0.85037782]),
        ({"x0": [-5.0, -5.0], "constraints": None}, [0.84059993, 0.69998341]),
        ({"constraints": "Penalize"}, [0.90013445, 0.85037782]),
        ({"x0": [-5.0, -5.0], "constraints": "Penalize"}, [0.82405114, 0.61993136]),
    ],
)
def test_vdcma(options, xref):
    options.update({"sigma": 0.1, "muperc": 0.5})
    helpers.optimize_parallel("vdcma", options, xref)


@pytest.mark.parametrize("method", ["cmaes", "cpso", "de", "na", "pso", "vdcma"])
def test_callback(method):
    global count
    count = 0

    def callback(X, state):
        global count
        count += 1

    maxiter = np.random.randint(2, 10)
    _ = minimize(
        rosenbrock,
        [[-5.12, 5.12]] * 2,
        method=method,
        options={"maxiter": maxiter},
        callback=callback,
    )
    assert count == maxiter
