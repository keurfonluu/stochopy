import numpy
import pytest

from stochopy import MonteCarlo


@pytest.mark.parametrize(
    "sampler, sampler_kws, mean_ref",
    [
        ("pure", {}, [-0.6070602, -0.00363818]),
        ("hastings", {"stepsize": 0.1409}, [-1.61141558, 2.73788443]),
        ("hamiltonian", {"stepsize": 0.0091991, "n_leap": 14}, [0.89343405, 1.18474131]),
    ],
)
def test_montecarlo(sampler, sampler_kws, mean_ref):
    mc = MonteCarlo(
        func=lambda x: 100.0 * numpy.sum((x[1:] - x[:-1]**2)**2) + numpy.sum((1.0 - x[:-1])**2),
        lower=numpy.full(2, -5.12),
        upper=numpy.full(2, 5.12),
        max_iter=50,
        random_state=42,
    )
    mc.sample(sampler=sampler, **sampler_kws)

    assert numpy.allclose(mean_ref, mc.models.mean(axis=0))
