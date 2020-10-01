"""
Minimize a function
===================

This examples shows a basic usage of :func:`stochopy.optimize.minimize`.

"""

########################################################################################
# Let's import an objective function to optimize. :mod:`stochopy.factory` has several sample benchmark functions to test.
# We also have to define the feasible space (or boundaries) for each variable to optimize. The length of the boundary array is used internally to define the dimensionality of the problem. In this example, we will optimize 20 variables within [-5.12, 5.12].

import numpy
from stochopy.factory import rosenbrock

upper = numpy.full(20, 5.12)
bounds = numpy.column_stack((-upper, upper))

########################################################################################

########################################################################################
# The main optimization function :func:`stochopy.optimize.minimize` has an API inspired by :mod:`scipy`.
# In this example, we will use CMA-ES to minimize the Rosenbrock function 

from stochopy.optimize import minimize

x = minimize(rosenbrock, bounds, method="cmaes", options={"maxiter": 2000, "popsize": 20, "seed": 42})

########################################################################################

########################################################################################
# :func:`stochopy.optimize.minimize` returns a :class:`stochopy.optimize.OptimizeResult` dictionary-like that contains the optimization result.

print(x)

########################################################################################
