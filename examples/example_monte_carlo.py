# -*- coding: utf-8 -*-

"""
This example shows how to optimize a benchmark function using Monte-Carlo
algorithms.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from time import time
import seaborn as sns
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
    mc = MonteCarlo(max_iter = 20000, random_state = 0, **bf.get())
    
    # Solve
    starttime = time()
    mc.sample(sampler = "hastings", stepsize = 0.02, perc = 1)
    
    # Print solution
    print(mc)
    print("Elapsed time: %.2f" % (time() - starttime))
    
    # Plot distribution
    sns.set_style("ticks")
    sns.jointplot(mc.models[:,0], mc.models[:,1], kind = "scatter")