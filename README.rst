stochopy
========

|License| |Stars| |Pyversions| |Version| |Downloads| |Code style: black| |Codacy Badge| |Codecov| |Build| |Travis|

**stochopy** provides functions for sampling or optimizing objective functions with or without constraints. Its API is directly inspired by **scipy**'s own optimization submodule which should make the switch from one module to another straightforward.

Features
--------

Sampling algorithms:

-  Hamiltonian (Hybrid) Monte-Carlo (HMC),
-  Markov-Chain Monte-Carlo (McMC).

Stochastic optimizers:

-  Competitive Particle Swarm Optimization (CPSO),
-  Covariance Matrix Adaptation - Evolution Strategy (CMA-ES),
-  Differential Evolution (DE),
-  Particle Swarm Optimization (PSO),
-  VD-CMA.

Parallel backends:

- `joblib <https://github.com/joblib/joblib>`__ (``threading`` and ``loky``),
- `mpi4py <https://bitbucket.org/mpi4py/mpi4py/src/master/>`__ (``mpi``).

Installation
------------

The recommended way to install **stochopy** and all its dependencies is through the Python Package Index:

.. code::

   pip install stochopy --user

Otherwise, clone and extract the package, then run from the package location:

.. code::

   pip install . --user

To test the integrity of the installed package, check out this repository and run:

.. code::

   pytest

Documentation
-------------

Refer to the online `documentation <https://keurfonluu.github.io/stochopy/>`__ for detailed description of the API and examples.

Alternatively, the documentation can be built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__

.. code:: bash

   pip install -r doc/requirements.txt
   sphinx-build -b html doc/source doc/build

Usage
-----

Given an optimization problem defined by an objective function and a feasible space:

.. code-block:: python

   import numpy

   def rosenbrock(x):
      x = numpy.asarray(x)
      sum1 = ((x[1:] - x[:-1] ** 2) ** 2).sum()
      sum2 = numpy.square(1.0 - x[:-1]).sum()
      return 100.0 * sum1 + sum2

   bounds = [[-5.12, 5.12], [-5.12, 5.12]]  # The number of variables to optimize is len(bounds)

The optimal solution can be found following:

.. code-block:: python

   from stochopy.optimize import minimize

   x = minimize(rosenbrock, bounds, method="cmaes", options={"maxiter": 100, "popsize": 10, "seed": 0})

``minimize`` returns a dictionary that contains the results of the optimization:

.. code-block::

        fun: 3.862267657514075e-09
    message: 'best solution value is lower than ftol'
       nfev: 490
        nit: 49
     status: 1
    success: True
          x: array([0.99997096, 0.99993643])

Contributing
------------

Please refer to the `Contributing
Guidelines <https://github.com/keurfonluu/stochopy/blob/master/CONTRIBUTING.rst>`__ to see how you can help. This project is released with a `Code of Conduct <https://github.com/keurfonluu/stochopy/blob/master/CODE_OF_CONDUCT.rst>`__ which you agree to abide by when contributing.

.. |License| image:: https://img.shields.io/github/license/keurfonluu/stochopy
   :target: https://github.com/keurfonluu/stochopy/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/keurfonluu/stochopy?logo=github
   :target: https://github.com/keurfonluu/stochopy

.. |Pyversions| image:: https://img.shields.io/pypi/pyversions/stochopy.svg?style=flat
   :target: https://pypi.org/pypi/stochopy/

.. |Version| image:: https://img.shields.io/pypi/v/stochopy.svg?style=flat
   :target: https://pypi.org/project/stochopy

.. |Downloads| image:: https://pepy.tech/badge/stochopy
   :target: https://pepy.tech/project/stochopy

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black

.. |Codacy Badge| image:: https://img.shields.io/codacy/grade/29b21d65d07e40219dcc9ad1c978cbeb.svg?style=flat
   :target: https://www.codacy.com/manual/keurfonluu/stochopy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keurfonluu/stochopy&amp;utm_campaign=Badge_Grade

.. |Codecov| image:: https://img.shields.io/codecov/c/github/keurfonluu/stochopy.svg?style=flat
   :target: https://codecov.io/gh/keurfonluu/stochopy

.. |Build| image:: https://img.shields.io/github/workflow/status/keurfonluu/stochopy/Python%20package
   :target: https://github.com/keurfonluu/stochopy

.. |Travis| image:: https://img.shields.io/travis/com/keurfonluu/stochopy/master?label=docs
   :target: https://keurfonluu.github.io/stochopy/