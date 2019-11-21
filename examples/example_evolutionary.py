# -*- coding: utf-8 -*-

"""
This example shows how to optimize a benchmark function using an evolutionary
algorithm.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from time import time
try:
    from stochopy import Evolutionary, BenchmarkFunction
except ImportError:
    import sys
    sys.path.append("../")
    from stochopy import Evolutionary, BenchmarkFunction


if __name__ == "__main__":
    # Parameters
    func = "sphere"
    
    # Initialize function
    bf = BenchmarkFunction(func, n_dim = 10)
    
    # Initialize solver
    ea = Evolutionary(popsize = 10, max_iter = 200, constrain = True, random_state = -1, **bf.get())
    
    # Solve
    starttime = time()
    ea.optimize(solver = "pso", sync = False)
    
    # Print solution
    print(ea)
    print("Elapsed time: %.2f seconds" % (time() - starttime))