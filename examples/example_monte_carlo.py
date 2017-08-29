# -*- coding: utf-8 -*-

"""
This example shows how to optimize a benchmark function using Monte-Carlo
algorithms.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from time import time
try:
    from stochopy import MonteCarlo, BenchmarkFunction
except ImportError:
    import sys
    sys.path.append("../")
    from stochopy import MonteCarlo, BenchmarkFunction


if __name__ == "__main__":
    # Parameters
    func = "rosenbrock"
    
    # Initialize function
    bf = BenchmarkFunction(func, n_dim = 2)
    
    # Initialize solver
    mc = MonteCarlo(max_iter = 5000, random_state = 0, **bf.get())
    
    # Solve
    starttime = time()
    mc.sample(sampler = "hastings", stepsize = 0.1)
    
    # Print solution
    print(mc)
    print("Elapsed time: %.2f" % (time() - starttime))