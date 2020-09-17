import pytest

import helpers


@pytest.mark.parametrize("options, xref", [({}, [-1.6315912, 2.60020735])])
def test_mcmc(options, xref):
    options.update({
        "stepsize": 0.1
    })
    helpers.sample("mcmc", options, xref)


@pytest.mark.parametrize("options, xref", [({}, [-1.28470918, 4.6153145])])
def test_hmc(options, xref):
    options.update({
        "nleap": 10,
        "stepsize": 0.1,
    })
    helpers.sample("hmc", options, xref)
