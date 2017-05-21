# -*- coding: utf-8 -*-

"""
This example shows how to plot benchmark functions.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
try:
    from stochopy import MonteCarlo, Evolutionary, BenchmarkFunction
except:
    import sys
    sys.path.append("../stochopy")
    from monte_carlo import MonteCarlo
    from evolutionary_algorithm import Evolutionary
    from benchmark_functions import BenchmarkFunction
    

if __name__ == "__main__":
    # Parameters
    func = "rosenbrock"
    
    # Initialize function
    bf = BenchmarkFunction(func, n_dim = 2)
    
    # Initialize solver
    ea = Evolutionary(**bf.get(), popsize = 5, max_iter = 200)
    
    # Solve
    xopt, gfit = ea.optimize(solver = "cpso", snap = True)
    
    # Animate
    bf.animate(ea.models, ea.energy, interval = 50, yscale = "log")