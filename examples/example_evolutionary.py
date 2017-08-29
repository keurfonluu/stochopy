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
    func = "rastrigin"
    
    # Initialize function
    bf = BenchmarkFunction(func, n_dim = 20)
    
    # Initialize solver
    ea = Evolutionary(popsize = 20, max_iter = 1000, random_state = 2, **bf.get())
    
    # Solve
    starttime = time()
    ea.optimize(solver = "de", CR = 0.1, F = 0.5, sync = False)
    
    # Print solution
    print(ea)
    print("Elapsed time: %.2f seconds" % (time() - starttime))