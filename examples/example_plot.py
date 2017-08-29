# -*- coding: utf-8 -*-

"""
This example shows how to optimize and plot a benchmark function.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import matplotlib.pyplot as plt
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
    bf = BenchmarkFunction(func, n_dim = 2)
    
    # Initialize solver
    ea = Evolutionary(popsize = 5, max_iter = 200, snap = True, **bf.get())
    
    # Solve
    ea.optimize(solver = "cpso")
    
    # Plot in 3D
    ax1 = bf.plot(figsize = (8, 6), projection = "3d")
    ax1.view_init(azim = -45, elev = 74)
    plt.show()
    
    # Save figure
    plt.savefig("%s.png" % func)