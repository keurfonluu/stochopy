# -*- coding: utf-8 -*-

"""
Evolutionary Algorithms are population based stochastic global optimization
methods.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from warnings import warn
try:
    from mpi4py import MPI
except ImportError:
    mpi_exist = False
else:
    mpi_exist = True

__all__ = [ "Evolutionary" ]


class Evolutionary:
    """
    Evolutionary Algorithm optimizer.
    
    This optimizer minimizes an objective function using Differential
    Evolution (DE), Particle Swarm Optimization (PSO), Competitive Particle
    Swarm Optimization (CPSO), Covariance Matrix Adaptation - Evolution
    Strategy (CMA-ES), or VD-CMA.
    
    Parameters
    ----------
    func : callable
        Objective function. If necessary, the variables required for its
        computation should be passed in 'args' and/or 'kwargs'.
    lower : ndarray, optional, default None
        Search space lower boundary.
    upper : ndarray, optional, default None
        Search space upper boundary.
    n_dim : int, optional, default 1
        Search space dimension. Only used if 'lower' and 'upper' are not
        provided.
    popsize : int, optional, default 10
        Population size.
    max_iter : int, optional, default 100
        Maximum number of iterations.
    eps1 : scalar, optional, default 1e-8
        Minimum change in best individual.
    eps2 : scalar, optional, default 1e-8
        Minimum objective function precision.
    constrain : bool, optional, default True
        Constrain to search space if an individual leave the search space.
    snap : bool, optional, default False
        Save the positions and energy of all individuals at each iteration
        in a 3-D array with shape (n_dim, popsize, max_iter) and 2-D array
        with shape (popsize, max_iter) in attributes 'models' and 'energy'.
    random_state : int, optional, default None
        Seed for random number generator.
    mpi : bool, default False
        Enable MPI parallelization.
    args : tuple, default ()
        Arguments to pass to objective function.
    kwargs : dict, default {}
        Keyworded arguments to pass to objective function.
    """
    
    _ATTRIBUTES = [ "solution", "fitness", "n_iter", "n_eval", "flag" ]
    
    def __init__(self, func, lower = None, upper = None, n_dim = 1,
                 popsize = 10, max_iter = 100, eps1 = 1e-8, eps2 = 1e-8,
                 constrain = True, snap = False, random_state = None, mpi = False,
                 args = (), kwargs = {}):
        # Check inputs
        if not hasattr(func, "__call__"):
            raise ValueError("func is not callable")
        else:
            self._func = lambda x: func(x, *args, **kwargs)
        if lower is None and upper is not None:
            raise ValueError("lower is not defined")
        elif upper is None and lower is not None:
            raise ValueError("upper is not defined")
        elif lower is not None and upper is not None:
            if len(lower) != len(upper):
                raise ValueError("lower and upper must have the same length")
            if np.any(upper < lower):
                raise ValueError("upper must be greater than lower")
            self._lower = np.array(lower)
            self._upper = np.array(upper)
            self._n_dim = len(lower)
        else:
            self._lower = np.full(n_dim, -100.)
            self._upper = np.full(n_dim, 100.)
            self._n_dim = n_dim
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer, got %s" % max_iter)
        else:
            self._max_iter = max_iter
        if not isinstance(popsize, float) and not isinstance(popsize, int) or popsize < 2:
            raise ValueError("popsize must be an integer > 1, got %s" % popsize)
        else:
            self._popsize = int(popsize)
        if not isinstance(eps1, float) and not isinstance(eps1, int) or eps1 < 0.:
            raise ValueError("eps1 must be positive, got %s" % eps1)
        else:
            self._eps1 = eps1
        if not isinstance(eps2, float) and not isinstance(eps2, int):
            raise ValueError("eps2 must be an integer or float, got %s" % eps2)
        else:
            self._eps2 = eps2
        if not isinstance(constrain, bool):
            raise ValueError("constrain must be either True or False, got %s" % constrain)
        else:
            self._constrain = constrain
        if not isinstance(snap, bool):
            raise ValueError("snap must be either True or False, got %s")
        else:
            self._snap = snap
        if random_state is not None and random_state >= 0:
            np.random.seed(random_state)
        if not isinstance(mpi, bool):
            raise ValueError("mpi must be either True or False, got %s" % mpi)
        else:
            self._mpi = mpi
            if mpi and not mpi_exist:
                raise ValueError("mpi4py is not installed or not properly installed")
        if not isinstance(args, (list, tuple)):
            raise ValueError("args must be a list or a tuple")
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a dictionary")
        return
    
    def __repr__(self):
        attributes = [ "%s: %s" % (attr.rjust(13), self._print_attr(attr))
                        for attr in self._ATTRIBUTES ]
        if self._solver == "cpso":
            attributes.append("%s: %s" % ("n_restart".rjust(13), self._print_attr("n_restart")))
        if self._mpi:
            attributes.append("%s: %s seconds" % ("t_serial".rjust(13), self._print_attr("t_serial")))
            attributes.append("%s: %s seconds" % ("t_parallel".rjust(13), self._print_attr("t_parallel")))
        return "\n".join(attributes) + "\n"
    
    def _print_attr(self, attr):
        ATTRIBUTES = self._ATTRIBUTES + [ "n_restart" ]
        if self._mpi:
            ATTRIBUTES += [ "t_serial", "t_parallel" ]
        if attr not in ATTRIBUTES:
            raise ValueError("attr should be in %s" % ATTRIBUTES)
        else:
            if attr == "solution":
                param = "\n"
                for i in range(self._n_dim):
                    tmp = "%.8g" % self._xopt[i]
                    if self._xopt[i] >= 0.:
                        tmp = " " + tmp
                    param += "\t\t%s\n" % tmp
                return param[:-1]
            elif attr == "fitness":
                return "%.8g" % self._gfit
            elif attr == "n_iter":
                return "%d" % self._n_iter
            elif attr == "n_eval":
                return "%d" % self._n_eval
            elif attr == "n_restart":
                return "%d" % self._n_restart
            elif attr == "flag":
                return "%s" % self.flag
            elif attr == "t_serial":
                return "%.8g" % (np.sum(self._time_serial))
            elif attr == "t_parallel":
                return "%.8g" % (np.sum(self._time_parallel))
    
    def optimize(self, solver = "cpso", xstart = None, sync = True,
                 w = 0.7298, c1 = 1.49618, c2 = 1.49618, gamma = 1.,
                 F = 0.5, CR = 0.1, strategy = "best2",
                 sigma = 0.5, mu_perc = 0.5):
        """
        Minimize an objective function using Differential Evolution (DE),
        Particle Swarm Optimization (PSO), Competitive Particle Swarm
        Optimization (CPSO), Covariance Matrix Adaptation - Evolution
        Strategy (CMA-ES), or VD-CMA.
        
        Parameters
        ----------
        solver : {'de', 'pso', 'cpso', 'cmaes', 'vdcma'}, default 'cpso'
            Optimization method.
            - 'de', Differential Evolution.
            - 'pso', Particle Swarm Optimization.
            - 'cpso', Competitive Particle Swarm Optimization.
            - 'cmaes', Covariance Matrix Adaptation - Evolution Strategy.
            - 'vdcma', VD-CMA.
        xstart : None or ndarray, optional, default None
            Initial positions of the population or mean (if solver = 'cmaes').
        sync : bool, optional, default True
            Synchronize population, the best individual is updated after each
            iteration which allows the parallelization. Only used if 'solver'
            is 'pso', 'cpso', or 'de'.
        w : scalar, optional, default 0.7298
            Inertial weight. Only used when solver = {'pso', 'cpso'}.
        c1 : scalar, optional, default 1.49618
            Cognition parameter. Only used when solver = {'pso', 'cpso'}.
        c2 : scalar, optional, default 1.49618
            Sociability parameter. Only used when solver = {'pso', 'cpso'}.
        gamma : scalar, optional, default 1.
            Competitivity parameter. Only used when solver = 'cpso'.
        F : scalar, optional, default 0.5
            Differential weight. Only used when solver = 'de'.
        CR : scalar, optional, default 0.1
            Crossover probability. Only used when solver = 'de'.
        strategy : {'rand1', 'rand2', 'best1', 'best2'}, optional, default 'best2'
            Mutation strategy.
            - 'rand1', mutate a random vector by adding one scaled difference vector.
            - 'rand2', mutate a random vector by adding two scaled difference vectors.
            - 'best1', mutate the best vector by adding one scaled difference vector.
            - 'best2', mutate the best vector by adding two scaled difference vectors.
        sigma : scalar, optional, default 0.5
            Step size. Only used when solver = {'cmaes', 'vdcma'}.
        mu_perc : scalar, optional, default 0.5
            Number of parents as a percentage of population size. Only used
            when solver = {'cmaes', 'vdcma'}.
            
        Returns
        -------
        xopt : ndarray
            Optimal solution found by the optimizer.
        gfit : scalar
            Objective function value of the optimal solution.
        
        Examples
        --------
        Import the module and define the objective function (Rosenbrock):
        
        >>> import numpy as np
        >>> from stochopy import Evolutionary
        >>> f = lambda x: 100*np.sum((x[1:]-x[:-1]**2)**2)+np.sum((1-x[:-1])**2)
        
        Define the search space boundaries in 5-D:
        
        >>> n_dim = 5
        >>> lower = np.full(n_dim, -5.12)
        >>> upper = np.full(n_dim, 5.12)
        
        Initialize the Evolutionary Algorithm optimizer:
        
        >>> popsize = 30
        >>> max_iter = 1000
        >>> ea = Evolutionary(f, lower = lower, upper = upper,
                              popsize = popsize, max_iter = max_iter)
        
        Differential Evolution:
        
        >>> xopt, gfit = ea.optimize(solver = "de", F = 0.5, CR = 0.1)
        
        Particle Swarm Optimization:
        
        >>> xopt, gfit = ea.optimize(solver = "pso")
        
        Covariance Matrix Adaptation - Evolution Strategy:
        
        >>> xopt, gfit = ea.optimize(solver = "cmaes")
        """
        # Check input
        if not isinstance(solver, str) or solver not in [ "cpso", "pso", "de", "cmaes", "vdcma" ]:
            raise ValueError("solver must either be 'cpso', 'pso', 'de', 'cmaes' or 'vdcma', got %s" % solver)
        if not isinstance(sync, bool):
            raise ValueError("sync must either be True or False")
        if self._mpi and solver in [ "cpso", "pso", "de" ] and not sync:
            raise ValueError("cannot use MPI with asynchrone population")
        
        # Initialize
        self._solver = solver
        self._n_eval = 0
        self._n_restart = 0
        self._init_models()
        self._mu_scale = 0.5 * (self._upper + self._lower)
        self._std_scale = 0.5 * (self._upper - self._lower)
        if self._mpi:
            self._mpi_comm = MPI.COMM_WORLD
            self._mpi_rank = self._mpi_comm.Get_rank()
            self._mpi_size = self._mpi_comm.Get_size()
            self._time_serial = np.zeros(self._max_iter)
            self._time_parallel = np.zeros(self._max_iter)
        else:
            self._mpi_rank = 0
            self._mpi_size = 1
        
        # Solve
        if solver == "pso":
            xopt, gfit = self._cpso(w = w, c1 = c1, c2 = c2, gamma = 0.,
                                    xstart = xstart, sync = sync)
        elif solver == "cpso":
            xopt, gfit = self._cpso(w = w, c1 = c1, c2 = c2, gamma = gamma,
                                    xstart = xstart, sync = sync)
        elif solver == "de":
            xopt, gfit = self._de(F = F, CR = CR, strategy = strategy,
                                  xstart = xstart, sync = sync)
        elif solver == "cmaes":
            xopt, gfit = self._cmaes(sigma = sigma, mu_perc = mu_perc,
                                     xstart = xstart)
        elif solver == "vdcma":
            xopt, gfit = self._vdcma(sigma = sigma, mu_perc = mu_perc,
                                     xstart = xstart)
        return xopt, gfit
    
    def _standardize(self, models):
        return (models - self._mu_scale) / self._std_scale
    
    def _unstandardize(self, models):
        return models * self._std_scale + self._mu_scale
    
    def _init_models(self):
        self._models = np.zeros((self._popsize, self._n_dim, self._max_iter))
        self._energy = np.zeros((self._popsize, self._max_iter))
        return
    
    def _eval_models(self, models, it):
        n = models.shape[0]
        if self._mpi:
            starttime_parallel = MPI.Wtime()
            fit = np.zeros(n)
            fit_mpi = np.zeros_like(fit)
            self._mpi_comm.Barrier()
            self._mpi_comm.Bcast([ models, MPI.DOUBLE ], root = 0)
            for i in np.arange(self._mpi_rank, n, self._mpi_size):
                fit_mpi[i] = self._func(self._unstandardize(models[i]))
            self._mpi_comm.Barrier()
            self._mpi_comm.Allreduce([ fit_mpi, MPI.DOUBLE ], [ fit, MPI.DOUBLE ],
                                     op = MPI.SUM)
            self._time_parallel[it-1] = MPI.Wtime() - starttime_parallel
        else:
            fit = np.array([ self._func(self._unstandardize(models[i])) for i in range(n) ])
        self._n_eval += n
        return fit
    
    def _constrain_de(self, models):
        """
        Random constraint for Differential Evolution. Parameters of models that
        are in the infeasible space are regenerated uniformly.
        """
        models = np.where(np.logical_or(models < -1., models > 1.),
                          np.random.uniform(-1., 1., models.shape), models)
        return models
    
    def _constrain_cpso(self, models, models_old):
        """
        Shrinking approach for Particle Swarm Optimization and Competitive PSO.
        Velocity vector amplitude is shrinked for models that are in the
        infeasible space. This approach preserves the trajectory of the
        particles.
        """
        maskl = models < -1.
        masku = models > 1.
        if np.any(maskl) and np.any(masku):
            beta_l = np.min((models_old[maskl] + 1.) / (models_old[maskl] - models[maskl]))
            beta_u = np.min((models_old[masku] - 1.) / (models_old[masku] - models[masku]))
            beta = min(beta_l, beta_u)
            models = models_old + beta * (models - models_old)
        elif np.any(maskl) and not np.any(masku):
            beta = np.min((models_old[maskl] + 1.) / (models_old[maskl] - models[maskl]))
            models = models_old + beta * (models - models_old)
        elif not np.any(maskl) and np.any(masku):
            beta = np.min((models_old[masku] - 1.) / (models_old[masku] - models[masku]))
            models = models_old + beta * (models - models_old)
        return models
    
    def _constrain_cma(self, arxvalid, arx, xmean, xold, sigma, diagC, mueff, it,
                       bnd_weights, dfithist, validfitval, iniphase):
        """
        Box constraint handling by adding a penalty term that quantifies the
        distance of the parameters from the feasible parameter space.
        """
        # Clip to boundaries
        arxvalid = np.where(arxvalid < -1., -np.ones_like(arxvalid), arxvalid)
        arxvalid = np.where(arxvalid > 1., np.ones_like(arxvalid), arxvalid)
        arfitness = self._eval_models(arxvalid, it)
        
        # Get delta fitness values
        perc = np.percentile(arfitness, [ 25, 75 ])
        delta = ( perc[1] - perc[0] ) / self._n_dim / np.mean(diagC) / sigma**2
        
        # Catch non-sensible values
        if delta == 0:
            delta = np.min(dfithist[dfithist > 0.])
        elif not validfitval:
            dfithist = np.array([])
            validfitval = True
            
        # Store delta fitness values
        if len(dfithist) < 20 + (3.*self._n_dim) / self._popsize:
            dfithist = np.append(dfithist, delta)
        else:
            dfithist = np.append(dfithist[1:len(dfithist)+1], delta)
            
        # Corrected mean
        ti = np.logical_or(xmean < -1., xmean > 1.)
        tx = np.where(xmean < -1., -np.ones_like(xmean), xmean)
        tx = np.where(xmean > 1., np.ones_like(xmean), xmean)
        
        # Set initial weights
        if iniphase:
            if np.any(ti):
                bnd_weights.fill(2.0002 * np.median(dfithist))
                if validfitval and it > 2:
                    iniphase = False
                    
        if np.any(ti):
            tx = xmean - tx
            idx = np.logical_and(ti, np.abs(tx) > 3. * max( 1., np.sqrt(self._n_dim/mueff) ) \
                                 * sigma * np.sqrt(diagC))
            idx = np.logical_and(idx, np.sign(tx) == np.sign(xmean - xold))
            bnd_weights = np.array([ w*1.2**min(1., mueff/10./self._n_dim) if i else w
                                        for i, w in zip(idx, bnd_weights) ])
                    
        # Calculate scaling biased to unity, product is one
        bnd_scale = np.exp( 0.9 * ( np.log(diagC) - np.mean(np.log(diagC)) ) )
        
        # Assigned penalized fitness
        arfitness += np.dot((arxvalid - arx)**2, bnd_weights / bnd_scale)
        return arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase
    
    def _de_mutation(self, X, F, gbest, strategy, sync, i = None):
        if sync:
            idx = [ list(range(self._popsize)) for i in range(self._popsize) ]
            for i, l in enumerate(idx):
                l.remove(i)
                l = np.random.shuffle(l)
            idx = np.transpose(idx)
        else:
            idx = list(range(self._popsize))
            idx.remove(i)
            np.random.shuffle(idx)
            idx = np.array(idx)
            
        if strategy == "rand1":
            X1 = np.array(X[idx[0],:])
            X2 = np.array(X[idx[1],:])
            X3 = np.array(X[idx[2],:])
            V = X1 + F * (X2 - X3)
        elif strategy == "rand2":
            X1 = np.array(X[idx[0],:])
            X2 = np.array(X[idx[1],:])
            X3 = np.array(X[idx[2],:])
            X4 = np.array(X[idx[3],:])
            X5 = np.array(X[idx[4],:])
            V = X1 + F * (X2 + X3 - X4 - X5)
        elif strategy == "best1":
            X1 = np.array(X[idx[0],:])
            X2 = np.array(X[idx[1],:])
            V = gbest + F * (X1 - X2)
        elif strategy == "best2":
            X1 = np.array(X[idx[0],:])
            X2 = np.array(X[idx[1],:])
            X3 = np.array(X[idx[2],:])
            X4 = np.array(X[idx[3],:])
            V = gbest + F * (X1 + X2 - X3 - X4)
        return V
    
    def _de(self, F = 0.5, CR = 0.1, strategy = "best2", xstart = None, sync = True):
        """
        Minimize an objective function using Differential Evolution (DE).
        
        Parameters
        ----------
        F : scalar, optional, default 0.5
            Differential weight.
        CR : scalar, optional, default 0.1
            Crossover probability.
        strategy : {'rand1', 'rand2', 'best1', 'best2'}, optional, default 'best2'
            Mutation strategy.
            - 'rand1', mutate a random vector by adding one scaled difference vector.
            - 'rand2', mutate a random vector by adding two scaled difference vectors.
            - 'best1', mutate the best vector by adding one scaled difference vector.
            - 'best2', mutate the best vector by adding two scaled difference vectors.
        xstart : None or ndarray, optional, default None
            Initial positions of the population.
        sync : bool, optional, default True
            Synchronize population, the best individual is updated after each
            iteration which allows the parallelization.
            
        Returns
        -------
        xopt : ndarray
            Optimal solution found by the optimizer.
        gfit : scalar
            Objective function value of the optimal solution.
        
        References
        ----------
        .. [1] R. Storn and K. Price, *Differential Evolution - A Simple and
               Efficient Heuristic for global Optimization over Continuous
               Spaces*, Journal of Global Optimization, 1997, 11(4): 341-359
        """
        # Check inputs
        self._check_inputs(F, CR, strategy, xstart)
        
        # Start timer
        if self._mpi:
            starttime_serial = MPI.Wtime()
        
        # Population initial positions
        if xstart is None:
            X = np.random.uniform(-1., 1., (self._popsize, self._n_dim))
        else:
            X = self._standardize(xstart)
        
        # Compute fitness
        pfit = self._eval_models(X, 1)
        pbestfit = np.array(pfit)
        self._n_eval = self._popsize
        if self._snap:
            self._init_models()
            self._models[:,:,0] = self._unstandardize(X)
            self._energy[:,0] = np.array(pbestfit)
        
        # Initialize best individual
        gbidx = np.argmin(pbestfit)
        gfit = pbestfit[gbidx]
        gbest = np.array(X[gbidx,:])
        
        if self._mpi:
            self._time_serial[0] = MPI.Wtime() - starttime_serial
        
        # Iterate until one of the termination criterion is satisfied
        it = 1
        converge = False
        while not converge:
            if self._mpi:
                starttime_serial = MPI.Wtime()
            
            it += 1
            r1 = np.random.rand(self._popsize, self._n_dim)
            
            # Synchronous population
            if sync:
                # Mutation
                V = self._de_mutation(X, F, gbest, strategy, sync)
                
                # Recombination
                mask = np.zeros_like(r1, dtype = bool)
                irand = np.random.randint(self._n_dim, size = self._popsize)
                for i in range(self._popsize):
                    mask[i,irand[i]] = True
                mask = np.logical_or(mask, r1 <= CR)
                U = np.where(mask, V, X)
                if self._constrain:
                    U = self._constrain_de(U)
                
                # Selection
                pfit = self._eval_models(U, it)
                idx = pfit < pbestfit
                pbestfit[idx] = pfit[idx]
                X[idx] = U[idx]
                
                # Update best individual
                gbidx = np.argmin(pbestfit)
                
                # Stop if best individual position changes less than eps1
                if np.linalg.norm(gbest - X[gbidx]) <= self._eps1 \
                    and pbestfit[gbidx] <= self._eps2:
                    converge = True
                    xopt = self._unstandardize(X[gbidx])
                    gfit = pbestfit[gbidx]
                    self._flag = 0
                    
                # Stop if fitness is less than eps2
                elif pbestfit[gbidx] <= self._eps2:
                    converge = True
                    xopt = self._unstandardize(X[gbidx])
                    gfit = pbestfit[gbidx]
                    self._flag = 1
                
                # Stop if maximum iteration is reached
                elif it >= self._max_iter:
                    converge = True
                    xopt = self._unstandardize(X[gbidx])
                    gfit = pbestfit[gbidx]
                    self._flag = -1
                
                # Otherwise, update best individual
                else:
                    gbest = np.array(X[gbidx])
                    gfit = pbestfit[gbidx]
                    
            # Asynchronous population
            else:
                for i in range(self._popsize):
                    # Mutation
                    V = self._de_mutation(X, F, gbest, strategy, sync, i)
                    
                    # Recombination
                    mask = np.zeros(self._n_dim, dtype = bool)
                    irand = np.random.randint(self._n_dim)
                    mask[irand] = True
                    mask = np.logical_or(mask, r1[i] <= CR)
                    U = np.where(mask, V, X[i])
                    if self._constrain:
                        U = self._constrain_de(U)
                        
                    # Selection
                    pfit[i] = self._func(self._unstandardize(U))
                    self._n_eval += 1
                    if pfit[i] <= pbestfit[i]:
                        X[i] = np.array(U)
                        pbestfit[i] = pfit[i]
                        
                        # Update best individual
                        if pfit[i] <= gfit:
                            # Stop if best individual position changes less than eps1
                            if np.linalg.norm(gbest - X[i]) <= self._eps1 \
                                and pfit[i] <= self._eps2:
                                converge = True
                                xopt = self._unstandardize(X[i])
                                gfit = pbestfit[i]
                                self._flag = 0
                                
                            # Stop if fitness is less than eps2
                            elif pfit[i] <= self._eps2:
                                converge = True
                                xopt = self._unstandardize(X[i])
                                gfit = pbestfit[i]
                                self._flag = 1
                
                            # Otherwise, update best individual
                            else:
                                gbest = np.array(X[i])
                                gfit = pfit[i]
                                
                # Stop if maximum iteration is reached
                if not converge and it >= self._max_iter:
                    converge = True
                    xopt = self._unstandardize(gbest)
                    self._flag = -1
                    
            # Save models and energy
            if self._snap:
                self._models[:,:,it-1] = self._unstandardize(X)
                self._energy[:,it-1] = np.array(pbestfit)
                
            if self._mpi:
                self._time_serial[it-1] = MPI.Wtime() - starttime_serial
                    
        self._xopt = xopt
        self._gfit = gfit
        self._n_iter = it
        if self._mpi:
            self._time_serial = self._time_serial[:it] - self._time_parallel[:it]
            self._time_parallel = self._time_parallel[:it]
        if self._snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
        return xopt, gfit
        
    def _cpso(self, w = 0.7298, c1 = 1.49618, c2 = 1.49618, gamma = 1.,
              xstart = None, sync = True):
        """
        Minimize an objective function using Competitive Particle Swarm
        Optimization (CPSO). Set gamma = 0. for classical PSO.
        
        Parameters
        ----------
        w : scalar, optional, default 0.7298
            Inertial weight.
        c1 : scalar, optional, default 1.49618
            Cognition parameter.
        c2 : scalar, optional, default 1.49618
            Sociability parameter.
        gamma : scalar, optional, default 1.
            Competitivity parameter.
        xstart : None or ndarray, optional, default None
            Initial positions of the population.
        sync : bool, optional, default True
            Synchronize population, the best individual is updated after each
            iteration which allows the parallelization.
            
        Returns
        -------
        xopt : ndarray
            Optimal solution found by the optimizer.
        gfit : scalar
            Objective function value of the optimal solution.
        
        References
        ----------
        .. [1] J. Kennedy and R. Eberhart, *Particle swarm optimization*,
               Proceedings of ICNN'95 - International Conference on Neural
               Networks, 1995, 4: 1942-1948
        .. [2] F. Van Den Bergh, *An analysis of particle swarm optimizers*,
               University of Pretoria, 2001
        .. [3] K. Luu, M. Noble, A. Gesret, N. Belayouni and P.-F. Roux,
               *A parallel competitive Particle Swarm Optimization for
               non-linear first arrival traveltime tomography and uncertainty
               quantification*, Computers & Geosciences, 2018, 113: 81-93
        """
        # Check inputs
        self._check_inputs(w, c1, c2, gamma, xstart)
        
        # Start timer
        if self._mpi:
            starttime_serial = MPI.Wtime()
        
        # Particles initial positions
        if xstart is None:
            X = np.random.uniform(-1., 1., (self._popsize, self._n_dim))
        else:
            X = self._standardize(xstart)
        pbest = np.array(X)
        
        # Initialize particle velocity
        V = np.zeros((self._popsize, self._n_dim))
        
        # Compute fitness
        pfit = self._eval_models(X, 1)
        pbestfit = np.array(pfit)
        self._n_eval = self._popsize
        if self._snap:
            self._init_models()
            self._models[:,:,0] = self._unstandardize(X)
            self._energy[:,0] = np.array(pbestfit)
        
        # Initialize best individual
        gbidx = np.argmin(pbestfit)
        gfit = pbestfit[gbidx]
        gbest = np.array(X[gbidx])
        
        # Swarm maximum radius
        delta = np.log(1. + 0.003 * self._popsize) / np.max((0.2, np.log(0.01*self._max_iter)))
        
        if self._mpi:
            self._time_serial[0] = MPI.Wtime() - starttime_serial
        
        # Iterate until one of the termination criterion is satisfied
        it = 1
        converge = False
        while not converge:
            if self._mpi:
                starttime_serial = MPI.Wtime()
            
            it += 1
            r1 = np.random.rand(self._popsize, self._n_dim)
            r2 = np.random.rand(self._popsize, self._n_dim)
            
            # Synchronous population
            if sync:
                # Mutation
                V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
                if self._constrain:
                    X = np.array([ self._constrain_cpso(X[i,:] + V[i,:], X[i,:])
                                    for i in range(self._popsize) ])
                else:
                    X += V
                
                # Selection
                pfit = self._eval_models(X, it)
                idx = pfit < pbestfit
                pbestfit[idx] = np.array(pfit[idx])
                pbest[idx] = np.array(X[idx])
                
                # Update best individual
                gbidx = np.argmin(pbestfit)
                
                # Stop if best individual position changes less than eps1
                if np.linalg.norm(gbest - pbest[gbidx]) <= self._eps1 \
                    and pbestfit[gbidx] <= self._eps2:
                    converge = True
                    xopt = self._unstandardize(pbest[gbidx])
                    gfit = pbestfit[gbidx]
                    self._flag = 0
                    
                # Stop if fitness is less than eps2
                elif pbestfit[gbidx] <= self._eps2:
                    converge = True
                    xopt = self._unstandardize(pbest[gbidx])
                    gfit = pbestfit[gbidx]
                    self._flag = 1
                
                # Stop if maximum iteration is reached
                elif it >= self._max_iter:
                    converge = True
                    xopt = self._unstandardize(pbest[gbidx])
                    gfit = pbestfit[gbidx]
                    self._flag = -1
                
                # Otherwise, update best individual
                else:
                    gbest = np.array(pbest[gbidx])
                    gfit = pbestfit[gbidx]
                    
            # Asynchronous population
            else:
                for i in range(self._popsize):
                    # Mutation
                    V[i] = w * V[i] + c1 * r1[i] * (pbest[i] - X[i]) + c2 * r2[i] * (gbest - X[i])
                    if self._constrain:
                        X[i] = self._constrain_cpso(X[i] + V[i], X[i])
                    else:
                        X[i] += V[i]
                        
                    # Selection
                    pfit[i] = self._func(self._unstandardize(X[i]))
                    self._n_eval += 1
                    if pfit[i] <= pbestfit[i]:
                        pbest[i] = np.array(X[i])
                        pbestfit[i] = pfit[i]
                        
                        # Update best individual
                        if pfit[i] <= gfit:
                            # Stop if best individual position changes less than eps1
                            if np.linalg.norm(gbest - X[i]) <= self._eps1 \
                                and pfit[i] <= self._eps2:
                                converge = True
                                xopt = self._unstandardize(X[i])
                                gfit = pbestfit[i]
                                self._flag = 0
                                
                            # Stop if fitness is less than eps2
                            elif pfit[i] <= self._eps2:
                                converge = True
                                xopt = self._unstandardize(X[i])
                                gfit = pbestfit[i]
                                self._flag = 1
                
                            # Otherwise, update best individual
                            else:
                                gbest = np.array(X[i])
                                gfit = pfit[i]
                                
                # Stop if maximum iteration is reached
                if not converge and it >= self._max_iter:
                    converge = True
                    xopt = self._unstandardize(gbest)
                    self._flag = -1
                    
            # Save models and energy
            if self._snap:
                self._models[:,:,it-1] = self._unstandardize(X)
                self._energy[:,it-1] = np.array(pfit)
                
            # Competitive PSO algorithm
            if not converge and gamma > 0.:
                # Evaluate swarm size
                swarm_radius = np.max([ np.linalg.norm(X[i] - gbest) for i in range(self._popsize) ])
                swarm_radius /= np.sqrt(4.*self._n_dim)
                
                # Restart particles if swarm size is lower than threshold
                if swarm_radius < delta:
                    inorm = it / self._max_iter
                    nw = int((self._popsize-1.) / (1.+np.exp(1./0.09*(inorm-gamma+0.5))))
                    
                    # Reset positions, velocities and personal bests
                    if nw > 0:
                        self._n_restart += 1
                        idx = pbestfit.argsort()[:-nw-1:-1]
                        V[idx] = np.zeros((nw, self._n_dim))
                        X[idx] = np.random.uniform(-1., 1., (nw, self._n_dim))
                        pbest[idx] = np.array(X[idx])
                        pbestfit[idx] = np.full(nw, 1e30)
                        
            if self._mpi:
                self._time_serial[it-1] = MPI.Wtime() - starttime_serial
                
        self._xopt = np.array(xopt)
        self._gfit = gfit
        self._n_iter = it
        if self._mpi:
            self._time_serial = self._time_serial[:it] - self._time_parallel[:it]
            self._time_parallel = self._time_parallel[:it]
        if self._snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
        return xopt, gfit
        
    def _cmaes(self, sigma = 0.5, mu_perc = 0.5, xstart = None):
        """
        Minimize an objective function using Covariance Matrix Adaptation
        - Evolution Strategy (CMA-ES).
        
        Parameters
        ----------
        sigma : scalar, optional, default 0.5
            Step size.
        mu_perc : scalar, optional, default 0.5
            Number of parents as a percentage of population size.
        xstart : None or ndarray, optional, default None
            Initial position of the mean.
            
        Returns
        -------
        xopt : ndarray
            Optimal solution found by the optimizer.
        gfit : scalar
            Objective function value of the optimal solution.
        
        References
        ----------
        .. [1] N. Hansen, *The CMA evolution strategy: A tutorial*, Inria,
               Université Paris-Saclay, LRI, 2011, 102: 1-34
        """
        # Check inputs
        self._check_inputs(sigma, mu_perc, xstart)
        
        # Initialize saved outputs
        if self._snap:
            self._init_models()
            self._means = np.zeros((self._max_iter, self._n_dim))
        
        # Population initial positions
        if xstart is None:
            xmean = np.random.uniform(-1., 1., self._n_dim)
        else:
            if np.asarray(xstart).ndim == 1:
                xmean = self._standardize(xstart)
            else:
                arfitness = self._eval_models(self._standardize(xstart), 1)
                xmean = self._standardize(xstart[np.argmin(arfitness)])
        xold = np.empty_like(xmean)
        
        # Number of parents
        mu = int(mu_perc * self._popsize)
            
        # Strategy parameter setting: Selection
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        # Strategy parameter setting: Adaptation
        cc = ( 4. + mueff / self._n_dim ) / ( self._n_dim + 4. + 2. * mueff / self._n_dim )
        cs = ( mueff + 2. ) / ( self._n_dim + mueff + 5. )
        c1 = 2. / ( ( self._n_dim + 1.3 )**2 + mueff )
        cmu = min(1. - c1, 2. * ( mueff - 2. + 1. / mueff ) / ( ( self._n_dim + 2. )**2 + mueff ) )
        damps = 1. + 2. * max(0., np.sqrt( ( mueff - 1. ) / ( self._n_dim + 1. ) ) - 1.) + cs
        
        # Initialize dynamic (internal) strategy parameters and constants
        pc = np.zeros(self._n_dim)
        ps = np.zeros(self._n_dim)
        B = np.eye(self._n_dim)
        D = np.ones(self._n_dim)
        C = np.eye(self._n_dim)
        invsqrtC = np.eye(self._n_dim)
        chind = np.sqrt(self._n_dim) * ( 1. - 1. / ( 4. * self._n_dim ) + 1. / ( 21. * self._n_dim**2 ) )
        
        # Initialize boundaries weights
        bnd_weights = np.zeros(self._n_dim)
        dfithist = np.array([ 1. ])
        
        # (mu, lambda)-CMA-ES
        self._n_eval = 0
        it = 0
        eigeneval = 0
        arbestfitness = np.zeros(self._max_iter)
        ilim = int(10 + 30 * self._n_dim / self._popsize)
        insigma = sigma
        validfitval = False
        iniphase = True
        converge = False
        
        while not converge:
            if self._mpi:
                starttime_serial = MPI.Wtime()
            
            it += 1
            
            # Generate lambda offsprings
            arx = np.array([ xmean + sigma * np.dot(B, D*np.random.randn(self._n_dim))
                            for i in range(self._popsize) ])
            arxvalid = np.array(arx)
                
            # Evaluate fitness
            if self._constrain:
                arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase = self._constrain_cma(
                        arxvalid, arx, xmean, xold, sigma, np.diag(C), mueff, it,
                        bnd_weights, dfithist, validfitval, iniphase)
            else:
                arfitness = self._eval_models(arxvalid, it)
            if self._snap:
                self._models[:,:,it-1] = self._unstandardize(arxvalid)
                self._energy[:,it-1] = np.array(arfitness)
                self._means[it-1,:] = self._unstandardize(xmean)
            
            # Sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(arfitness)
            xold = np.array(xmean)
            xmean = np.dot(weights, arx[arindex[:mu],:])
            
            # Save best fitness
            arbestfitness[it-1] = arfitness[arindex[0]]
            
            # Cumulation
            ps = ( 1. - cs ) * ps \
                 + np.sqrt( cs * ( 2. - cs ) * mueff ) * np.dot(invsqrtC, xmean - xold) / sigma
            if np.linalg.norm(ps) / np.sqrt( 1. - ( 1. - cs )**(2.*self._n_eval/self._popsize) ) / chind < 1.4 + 2. / ( self._n_dim + 1. ):
                hsig = 1.
                pc = ( 1. - cc ) * pc \
                     + np.sqrt( cc * ( 2. - cc ) * mueff ) * (xmean - xold) / sigma
            else:
                hsig = 0.
                pc = ( 1. - cc ) * pc
                 
            # Adapt covariance matrix C
            artmp = ( arx[arindex[:mu],:] - np.tile(xold, (mu, 1)) ) / sigma
            if hsig:
                C = ( 1. - c1 - cmu ) * C \
                    + c1 * np.outer(pc, pc) \
                    + cmu * np.dot(np.dot(artmp.transpose(), np.diag(weights)), artmp)
            else:
                C = ( 1. - c1 - cmu ) * C \
                    + c1 * ( np.outer(pc, pc) + cc * ( 2. - cc ) * C ) \
                    + cmu * np.dot(np.dot(artmp.transpose(), np.diag(weights)), artmp)
                
            # Adapt step size sigma
            sigma *= np.exp( ( cs / damps ) * ( np.linalg.norm(ps) / chind - 1. ) )
            
            # Diagonalization of C
            if self._n_eval - eigeneval > self._popsize / ( c1 + cmu ) / self._n_dim / 10.:
                eigeneval = self._n_eval
                C = np.triu(C) + np.triu(C, 1).transpose()
                D, B = np.linalg.eigh(C)
                idx = np.argsort(D)
                D = D[idx]
                B = B[:,idx]
                D = np.sqrt(D)
                invsqrtC = np.dot(np.dot(B, np.diag(1./D)), B.transpose())
            
            # Stop if maximum iteration is reached
            if it >= self._max_iter:
                converge = True
                self._flag = -1
            
            # Stop if mean position changes less than eps1
            if not converge and np.linalg.norm(xold - xmean) <= self._eps1 \
                and arfitness[arindex[0]] < self._eps2:
                converge = True
                self._flag = 0
                
            # Stop if fitness is less than eps2
            if not converge and arfitness[arindex[0]] <= self._eps2:
                converge = True
                self._flag = 1
                
            # NoEffectAxis: stop if numerical precision problem
            i = int(np.floor(np.mod(it, self._n_dim)))
            if not converge and np.all( np.abs(0.1 * sigma * B[:,i] * D[i]) < 1e-10 ):
                converge = True
                self._flag = 2
                
            # NoEffectCoord: stop if too low coordinate axis deviations
            if not converge and np.any( 0.2 * sigma * np.sqrt(np.diag(C)) < 1e-10 ):
                converge = True
                self._flag = 3
            
            # ConditionCov: stop if the condition number exceeds 1e14
            if not converge and np.max(D) > 1e7 * np.min(D):
                converge = True
                self._flag = 4
            
            # EqualFunValues: stop if the range of fitness values is zero
            if not converge and it >= ilim:
                if np.max(arbestfitness[it-ilim:it+1]) - np.min(arbestfitness[it-ilim:it+1]) < 1e-10:
                    converge = True
                    self._flag = 5
                    
            # TolXUp: stop if x-changes larger than 1e3 times initial sigma
            if not converge and np.any( sigma * np.sqrt(np.diag(C)) > 1e3 * insigma ):
                converge = True
                self._flag = 6
                
            # TolFun: stop if fun-changes smaller than 1e-12
            if not converge and it > 2 and np.max(np.append(arfitness, arbestfitness)) - np.min(np.append(arfitness, arbestfitness)) < 1e-12:
                converge = True
                self._flag = 7
                
            # TolX: stop if x-changes smaller than 1e-11 times initial sigma
            if not converge and np.all( sigma * np.max(np.append(np.abs(pc), np.sqrt(np.diag(C)))) < 1e-11 * insigma ):
                converge = True
                self._flag = 8
                
            if self._mpi:
                self._time_serial[it-1] = MPI.Wtime() - starttime_serial
                
        xopt = self._unstandardize(arxvalid[arindex[0]])
        gfit = arfitness[arindex[0]]
        self._xopt = np.array(xopt)
        self._gfit = gfit
        self._n_iter = it
        if self._mpi:
            self._time_serial = self._time_serial[:it] - self._time_parallel[:it]
            self._time_parallel = self._time_parallel[:it]
        if self._snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
            self._means = self._means[:it,:]
        return xopt, gfit
    
    def _vdcma(self, sigma = 0.5, mu_perc = 0.5, xstart = None):
        """
        Minimize an objective function using VD-CMA.
        
        Parameters
        ----------
        sigma : scalar, optional, default 0.5
            Step size.
        mu_perc : scalar, optional, default 0.5
            Number of parents as a percentage of population size.
        xstart : None or ndarray, optional, default None
            Initial position of the mean.
            
        Returns
        -------
        xopt : ndarray
            Optimal solution found by the optimizer.
        gfit : scalar
            Objective function value of the optimal solution.
        
        References
        ----------
        .. [1] Y. Akimoto, A. Auger and N. Hansen, *Comparison-Based Natural
               Gradient Optimization in High Dimension*, Proceedings of the
               2014 conference on Genetic and evolutionary computation, 2014,
               373-380
        """
        # Check inputs
        self._check_inputs(sigma, mu_perc, xstart)
        
        # Initialize saved outputs
        if self._snap:
            self._init_models()
            self._means = np.zeros((self._max_iter, self._n_dim))
        
        # Population initial positions
        if xstart is None:
            xmean = np.random.uniform(-1., 1., self._n_dim)
        else:
            if np.asarray(xstart).ndim == 1:
                xmean = self._standardize(xstart)
            else:
                arfitness = self._eval_models(self._standardize(xstart), 1)
                xmean = self._standardize(xstart[np.argmin(arfitness)])
        xold = np.empty_like(xmean)
        
        # Number of parents
        mu = int(mu_perc * self._popsize)
        
        # Strategy parameter setting: Selection
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        # Strategy parameter setting: Adaptation
        cc = ( 4. + mueff / self._n_dim ) / ( self._n_dim + 4. + 2. * mueff / self._n_dim )
        cfactor = ( self._n_dim - 5. ) / 6.
        c1 = cfactor * 2. / ( ( self._n_dim + 1.3 )**2 + mueff )
        cmu = min(1. - c1, cfactor * 2. * ( mueff - 2. + 1. / mueff ) / ( ( self._n_dim + 2. )**2 + mueff) )
        
        # Initialize dynamic (internal) strategy parameters and constants
        flg_injection = False
        cs = 0.3
        ds = np.sqrt(self._n_dim)
        dx = np.zeros(self._n_dim)
        ps = 0.
        dvec = np.ones(self._n_dim)
        vvec = np.random.normal(0., 1., self._n_dim) / np.sqrt(self._n_dim)
        norm_v2 = np.dot(vvec, vvec)
        norm_v = np.sqrt(norm_v2)
        vn = vvec / norm_v
        vnn = vn**2
        pc = np.zeros(self._n_dim)
        
        # Initialize boundaries weights
        bnd_weights = np.zeros(self._n_dim)
        dfithist = np.array([ 1. ])
        
        # VD-CMA
        self._n_eval = 0
        it = 0
        arbestfitness = np.zeros(self._max_iter)
        ilim = int(10 + 30 * self._n_dim / self._popsize)
        insigma = sigma
        validfitval = False
        iniphase = True
        converge = False
        
        while not converge:
            if self._mpi:
                starttime_serial = MPI.Wtime()
                
            it += 1
            
            # Generate lambda offsprings
            arz = np.random.randn(self._popsize, self._n_dim)
            ary = dvec * ( arz + ( np.sqrt( 1. + norm_v2 ) - 1. ) * np.outer(np.dot(arz, vn), vn) )
            if flg_injection:
                ddx = dx / dvec
                mnorm = (ddx**2).sum() - np.dot(ddx, vvec)**2 / ( 1. + norm_v2 )
                dy = np.linalg.norm(np.random.randn(self._n_dim)) / np.sqrt(mnorm) * dx
                ary[0] = dy
                ary[1] = -dy
            arx = xmean + sigma * ary
            arxvalid = np.array(arx)
            diagC = np.diag(np.dot(np.dot(np.diag(dvec), np.eye(self._n_dim) + np.outer(vvec, vvec)), np.diag(dvec)))
                
            # Evaluate fitness
            if self._constrain:
                arfitness, arxvalid, bnd_weights, dfithist, validfitval, iniphase = self._constrain_cma(
                        arxvalid, arx, xmean, xold, sigma, diagC, mueff, it,
                        bnd_weights, dfithist, validfitval, iniphase)
            else:
                arfitness = self._eval_models(arxvalid, it)
            if self._snap:
                self._models[:,:,it-1] = self._unstandardize(arxvalid)
                self._energy[:,it-1] = np.array(arfitness)
                self._means[it-1,:] = self._unstandardize(xmean)
            
            # Sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(arfitness)
            dx = np.dot(weights, arx[arindex[:mu]]) - np.sum(weights) * xmean
            xold = np.array(xmean)
            xmean += dx
            
            # Save best fitness
            arbestfitness[it-1] = arfitness[arindex[0]]
    
            # Update sigma
            if flg_injection:
                alpha_act = np.where(arindex == 1)[0][0] - np.where(arindex == 0)[0][0]
                alpha_act /= self._popsize - 1.
                ps += cs * ( alpha_act - ps )
                sigma *= np.exp( ps / ds )
                hsig = ps < 0.5
            else:
                flg_injection = True
                hsig = True
    
            # Cumulation
            pc = ( 1. - cc ) * pc + hsig * np.sqrt( cc * ( 2. - cc ) * mueff ) * np.dot(weights, ary[arindex[:mu]])
    
            # Alpha and related variables
            gamma = 1. / np.sqrt( 1. + norm_v2 )
            alpha = np.sqrt( norm_v2**2 + ( 1. + norm_v2 ) / max(vnn) * ( 2. - gamma ) ) / ( 2. + norm_v2 )
            if alpha < 1.:
                beta = ( 4. - ( 2. - gamma ) / max(vnn) ) / ( 1. + 2. / norm_v2 )**2
            else:
                alpha = 1.
                beta = 0.
            bsca = 2. * alpha**2 - beta
            avec = 2. - ( bsca + 2. * alpha**2 ) * vnn
            invavnn = vnn / avec
            
            # Rank-mu
            if cmu == 0.:
                pvec_mu = np.zeros(self._n_dim)
                qvec_mu = np.zeros(self._n_dim)
            else:
                pvec_mu, qvec_mu = self._pvec_and_qvec(vn, norm_v2, ary[arindex[:mu]] / dvec, weights)
                
            # Rank-one
            if c1 == 0.:
                pvec_one = np.zeros(self._n_dim)
                qvec_one = np.zeros(self._n_dim)
            else:
                pvec_one, qvec_one = self._pvec_and_qvec(vn, norm_v2, pc / dvec)
                
            # Add rank-one and rank-mu before computing the natural gradient
            pvec = cmu * pvec_mu + hsig * c1 * pvec_one
            qvec = cmu * qvec_mu + hsig * c1 * qvec_one
            # Natural gradient
            if cmu + c1 > 0.:
                ngv, ngd = self._ngv_ngd(dvec, vn, vnn, norm_v, norm_v2, alpha, avec, bsca, invavnn,
                                         pvec, qvec)
                # Truncation factor to guarantee at most 70 percent change
                upfactor = 1.
                upfactor = min( upfactor, 0.7 * norm_v / np.sqrt( np.dot(ngv, ngv) ) )
                upfactor = min( upfactor, 0.7 * ( dvec / np.abs(ngd) ).min() )
            else:
                ngv = np.zeros(self._n_dim)
                ngd = np.zeros(self._n_dim)
                upfactor = 1.
            # Update parameters
            vvec += upfactor * ngv
            dvec += upfactor * ngd
    
            # Update the constants
            norm_v2 = np.dot(vvec, vvec)
            norm_v = np.sqrt(norm_v2)
            vn = vvec / norm_v
            vnn = vn**2
            
            # Stop if maximum iteration is reached
            if it >= self._max_iter:
                converge = True
                self._flag = -1
            
            # Stop if mean position changes less than eps1
            if not converge and np.linalg.norm(xold - xmean) <= self._eps1 \
                and arfitness[arindex[0]] < self._eps2:
                converge = True
                self._flag = 0
                
            # Stop if fitness is less than eps2
            if not converge and arfitness[arindex[0]] <= self._eps2:
                converge = True
                self._flag = 1
                
            # NoEffectCoord: stop if too low coordinate axis deviations
            if not converge and np.any( 0.2 * sigma * np.sqrt(diagC) < 1e-10 ):
                converge = True
                self._flag = 3
            
            # EqualFunValues: stop if the range of fitness values is zero
            if not converge and it >= ilim:
                if np.max(arbestfitness[it-ilim:it+1]) - np.min(arbestfitness[it-ilim:it+1]) < 1e-10:
                    converge = True
                    self._flag = 5
                    
            # TolXUp: stop if x-changes larger than 1e3 times initial sigma
            if not converge and np.any( sigma * np.sqrt(diagC) > 1e3 * insigma ):
                converge = True
                self._flag = 6
                
            # TolFun: stop if fun-changes smaller than 1e-12
            if not converge and it > 2 and np.max(np.append(arfitness, arbestfitness)) - np.min(np.append(arfitness, arbestfitness)) < 1e-12:
                converge = True
                self._flag = 7
                
            # TolX: stop if x-changes smaller than 1e-11 times initial sigma
            if not converge and np.all( sigma * np.max(np.append(np.abs(pc), np.sqrt(diagC))) < 1e-11 * insigma ):
                converge = True
                self._flag = 8
                
            if self._mpi:
                self._time_serial[it-1] = MPI.Wtime() - starttime_serial
        
        arindex = np.argsort(arfitness)
        xopt = self._unstandardize(arxvalid[arindex[0]])
        gfit = arfitness[arindex[0]]
        self._xopt = np.array(xopt)
        self._gfit = gfit
        self._n_iter = it
        if self._mpi:
            self._time_serial = self._time_serial[:it] - self._time_parallel[:it]
            self._time_parallel = self._time_parallel[:it]
        if self._snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
            self._means = self._means[:it,:]
        return xopt, gfit
    
    @staticmethod
    def _pvec_and_qvec(vn, norm_v2, y, weights = None):
        y_vn = np.dot(y, vn)
        if weights is None:
            pvec = y**2 - norm_v2 / ( 1. + norm_v2 ) * ( y_vn * ( y * vn ) ) - 1.
            qvec = y_vn * y - ( 0.5 * ( y_vn**2 + 1. + norm_v2 ) ) * vn
        else:
            pvec = np.dot(weights, y**2 - norm_v2 / ( 1. + norm_v2 ) * ( y_vn * ( y * vn ).T ).T - 1. )
            qvec = np.dot(weights, ( y_vn * y.T ).T - np.outer(0.5 * ( y_vn**2 + 1.0 + norm_v2 ), vn) )
        return pvec, qvec
    
    @staticmethod
    def _ngv_ngd(dvec, vn, vnn, norm_v, norm_v2, alpha, avec, bsca, invavnn, pvec, qvec):
        rvec = pvec - alpha / ( 1. + norm_v2 ) * ( ( 2. + norm_v2 ) * ( qvec * vn ) - norm_v2 * np.dot(vn, qvec) * vnn )
        svec = rvec / avec - bsca * np.dot(rvec, invavnn) / ( 1. + bsca * np.dot(vnn, invavnn) ) * invavnn
        ngv = qvec / norm_v - alpha / norm_v * ( ( 2. + norm_v2 ) * ( vn * svec ) - np.dot(svec, vnn) * vn )
        ngd = dvec * svec
        return ngv, ngd
    
    def _check_inputs(self, *args):
        if self._solver == "de":
            F, CR, strategy, xstart = args
            if self._popsize <= 3 and strategy not in [ "rand2", "best2" ]:
                self._popsize = 4
                warn("\npopsize cannot be lower than 4 for DE, popsize set to 4", UserWarning)
            elif self._popsize <= 4 and strategy == "best2":
                self._popsize = 5
                warn("\npopsize cannot be lower than 5 for DE, popsize set to 5", UserWarning)
            elif self._popsize <= 5 and strategy == "rand2":
                self._popsize = 6
                warn("\npopsize cannot be lower than 6 for DE, popsize set to 6", UserWarning)
            if not isinstance(F, float) and not isinstance(F, int) or not 0. <= F <= 2.:
                raise ValueError("F must be an integer or float in [ 0, 2 ], got %s" % F)
            if not isinstance(CR, float) and not isinstance(CR, int) or not 0. <= CR <= 1.:
                raise ValueError("CR must be an integer or float in [ 0, 1 ], got %s" % CR)
            if strategy not in [ "rand1", "rand2", "best1", "best2" ]:
                raise ValueError("strategy should either be 'rand1', 'rand2', 'best1' or 'best2'")
            if xstart is not None and isinstance(xstart, np.ndarray) \
                and xstart.shape != (self._popsize, self._n_dim):
                raise ValueError("xstart must be a ndarray of shape [ %d, %d ], got [ %d, %d ]" \
                                 % (self._popsize, self._n_dim, xstart.shape[0], xstart.shape[1]))
        elif self._solver in [ "pso", "cpso" ]:
            w, c1, c2, gamma, xstart = args
            if not isinstance(w, float) and not isinstance(w, int) or not 0. <= w <= 1.:
                raise ValueError("w must be an integer or float in [ 0, 1 ], got %s" % w)
            if not isinstance(c1, float) and not isinstance(c1, int) or not 0. <= c1 <= 4.:
                raise ValueError("c1 must be an integer or float in [ 0, 4 ], got %s" % c1)
            if not isinstance(c2, float) and not isinstance(c2, int) or not 0. <= c2 <= 4.:
                raise ValueError("c2 must be an integer or float in [ 0, 4 ], got %s" % c2)
            if not isinstance(gamma, float) and not isinstance(gamma, int) or not 0. <= gamma <= 2.:
                raise ValueError("gamma must be an integer or float in [ 0, 2 ], got %s" % gamma)
            if xstart is not None and isinstance(xstart, np.ndarray) \
                and xstart.shape != (self._popsize, self._n_dim):
                raise ValueError("xstart must be a ndarray of shape [ %d, %d ], got [ %d, %d ]" \
                                 % (self._popsize, self._n_dim, xstart.shape[0], xstart.shape[1]))
        elif self._solver in [ "cmaes", "vdcma" ]:
            sigma, mu_perc, xstart = args
            if self._popsize <= 3:
                self._popsize = 4
                warn("\npopsize cannot be lower than 4 for %s, popsize set to 4" % self._solver.upper(), UserWarning)
            if not isinstance(sigma, float) and not isinstance(sigma, int) or sigma <= 0.:
                raise ValueError("sigma must be positive, got %s" % sigma)
            if not isinstance(mu_perc, float) and not isinstance(mu_perc, int) or not 0. < mu_perc <= 1.:
                raise ValueError("mu_perc must be an integer or float in ] 0, 1 ], got %s" % mu_perc)
            if xstart is not None and isinstance(xstart, (list, np.ndarray)) and np.asarray(xstart).ndim not in [ 1, 2 ]:
                raise ValueError("xstart must be a 1-D or 2-D ndarray")
            if np.asarray(xstart).ndim == 1 and len(xstart) != self._n_dim:
                raise ValueError("xstart must be a list or ndarray of length %d, got %d" % (self._n_dim, len(xstart)))
            elif np.asarray(xstart).ndim == 2 and xstart.shape != (self._popsize, self._n_dim):
                raise ValueError("xstart must be a ndarray of shape [ %d, %d ], got [ %d, %d ]" \
                                 % (self._popsize, self._n_dim, xstart.shape[0], xstart.shape[1]))
    
    @property
    def xopt(self):
        """
        ndarray of shape (n_dim)
        Optimal solution found by the optimizer.
        """
        return self._xopt
    
    @property
    def gfit(self):
        """
        scalar
        Objective function value of the optimal solution.
        """
        return self._gfit
    
    @property
    def flag(self):
        """
        int
        Stopping criterion.
        """
        if self._flag == -1:
            return "maximum number of iterations is reached"
        elif self._flag == 0:
            return "best individual position changes less than eps1 (%g)" % self._eps1
        elif self._flag == 1:
            return "fitness is lower than threshold eps2 (%g)" % self._eps2
        elif self._flag == 2:
            return "NoEffectAxis"
        elif self._flag == 3:
            return "NoEffectCoord"
        elif self._flag == 4:
            return "ConditionCov"
        elif self._flag == 5:
            return "EqualFunValues"
        elif self._flag == 6:
            return "TolXUp"
        elif self._flag == 7:
            return "TolFun"
        elif self._flag == 8:
            return "TolX"
    
    @property
    def n_iter(self):
        """
        int
        Number of iterations required to reach stopping criterion.
        """
        return self._n_iter
    
    @property
    def n_eval(self):
        """
        int
        Number of function evaluations performed.
        """
        return self._n_eval
    
    @property
    def models(self):
        """
        ndarray of shape (popsize, n_dim, max_iter)
        Models explored by every individuals at each iteration. Available only
        when snap = True.
        """
        return self._models
    
    @property
    def energy(self):
        """
        ndarray of shape (popsize, max_iter)
        Energy of models explored by every individuals at each iteration.
        Available only when snap = True.
        """
        return self._energy
    
    @property
    def means(self):
        """
        ndarray of shape (max_iter, n_dim)
        Mean models at every iterations. Available only when
        solver = {'cmaes', 'vdcma'} and snap = True.
        """
        return self._means