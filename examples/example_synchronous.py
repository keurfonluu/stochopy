# -*- coding: utf-8 -*-

"""
This example evaluates the performance of an evolutionary algorithm executed
serially and concurrently.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
try:
    from stochopy import Evolutionary, BenchmarkFunction
except ImportError:
    import sys
    sys.path.append("../")
    from stochopy import Evolutionary, BenchmarkFunction


if __name__ == "__main__":
    # Parameters
    func = "rastrigin"
    n_run = 100
    
    # Initialize function
    bf = BenchmarkFunction(func, n_dim = 20)
    
    # Initialize solver
    ea = Evolutionary(popsize = 20, max_iter = 1000, eps2 = 1., **bf.get())
    
    # Solve
    sync, async = [], []
    for i in range(n_run):
        ea.optimize(solver = "de", sync = True)
        sync.append(ea._n_eval)
        ea.optimize(solver = "de", sync = False)
        async.append(ea._n_eval)
        
    # Print results
    print("Mean number of fitness evaluations:")
    print("Synchronous: %d" % np.mean(sync))
    print("Asynchronous: %d" % np.mean(async))