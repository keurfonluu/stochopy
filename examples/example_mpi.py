# -*- coding: utf-8 -*-

"""
This example shows how to optimize a benchmark function in parallel using MPI.
In a terminal, type: mpiexec -n 4 python example_mpi.py
Computation can be a little bit faster or slower compared with 1 process due to
communication cost. Usage of MPI is relevant only if the computation time of
the fitness function is high (speed-up almost ideal).

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from mpi4py import MPI
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
    
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    
    # Initialize function
    bf = BenchmarkFunction(func, n_dim = 20)
    
    # Initialize solver
    ea = Evolutionary(popsize = 20, max_iter = 10000, random_state = 42, mpi = True,
                      **bf.get())
    
    # Solve
    starttime = time()
    ea.optimize(solver = "cpso")
    
    # Print solution
    if mpi_rank == 0:
        print(ea)
        print("Elapsed time: %.2f seconds" % (time() - starttime))