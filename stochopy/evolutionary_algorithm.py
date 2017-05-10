# -*- coding: utf-8 -*-

"""
Evolutionary Algorithms are population based stochastic global optimization
methods.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from warnings import warn

__all__ = [ "Evolutionary" ]


class Evolutionary:
    """
    Evolutionary Algorithm optimizer.
    
    This optimizer minimizes an objective function using Differential
    Evolution (DE), Particle Swarm Optimization (PSO), Competitive Particle
    Swarm Optimization (CPSO), or Covariance Matrix Adaptation - Evolution
    Strategy (CMA-ES).
    
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
    clip : bool, optional, default False
        Clip to search space if an individual leave the search space.
    random_state : int, optional, default None
        Seed for random number generator.
    """
    
    def __init__(self, func, lower = None, upper = None, n_dim = 1,
                 popsize = 10, max_iter = 100, eps1 = 1e-8, eps2 = 1e-8,
                 clip = False, random_state = None, args = (), kwargs = {}):
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
        if not isinstance(clip, bool):
            raise ValueError("clip must be either True or False, got %s" % clip)
        else:
            self._clip = clip
        if random_state is not None:
            np.random.seed(random_state)
        return
    
    def optimize(self, solver = "cpso", xstart = None, w = 0.72, c1 = 1.49,
                 c2 = 1.49, l = 0.1, alpha = 1.25, delta = None, F = 1., CR = 0.5,
                 sigma = 1., mu_perc = 0.5, snap = False):
        """
        Minimize an objective function using Differential Evolution (DE),
        Particle Swarm Optimization (PSO), Competitive Particle Swarm
        Optimization (CPSO), or Covariance Matrix Adaptation - Evolution
        Strategy (CMA-ES).
        
        Parameters
        ----------
        solver : {'de', 'pso', 'cpso', 'cmaes'}, default 'cpso'
            Optimization method.
            - 'de', Differential Evolution.
            - 'pso', Particle Swarm Optimization.
            - 'cpso', Competitive Particle Swarm Optimization.
            - 'cmaes', Covariance Matrix Adaptation - Evolution Strategy.
        xstart : None or ndarray, optional, default None
            Initial positions of the population or mean (if solver = 'cmaes').
        w : scalar, optional, default 0.72
            Inertial weight. Only used when solver = 'pso'.
        c1 : scalar, optional, default 1.49
            Cognition parameter. Only used when solver = 'pso'.
        c2 : scalar, optional, default 1.49
            Sociability parameter. Only used when solver = 'pso'.
        l : scalar, optional, default 0.1
            Velocity clamping percentage. Only used when solver = 'pso'.
        alpha : scalar, optional, default 1.25
            Competitivity parameter. Only used when solver = 'cpso'.
        delta : None or scalar, optional, default None
            Swarm maximum radius. Only used when solver = 'cpso'.
        F : scalar, optional, default 1.
            Differential weight. Only used when solver = 'de'.
        CR : scalar, optional, default 0.5
            Crossover probability. Only used when solver = 'de'.
        sigma : scalar, optional, default 1.
            Step size. Only used when solver = 'cmaes'.
        mu_perc : scalar, optional, default 0.5
            Number of parents as a percentage of population size. Only used
            when solver = 'cmaes'.
        snap : bool, optional, default False
            Save the positions and energy of all individuals at each iteration
            in a 3-D array with shape (n_dim, popsize, max_iter) and 2-D array
            with shape (popsize, max_iter) in attributes 'models' and 'energy'.
            If solver = 'cmaes', also save the means, the covariance matrix and
            the stepsize in attributes 'means', 'covar' and 'sigma'.
            
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
        if not isinstance(solver, str) or solver not in [ "cpso", "pso", "de", "cmaes" ]:
            raise ValueError("solver must either be 'cpso', 'pso', 'de' or 'cmaes', got %s" % solver)
        
        # Initialize
        self._solver = solver
        self._init_models()
        
        # Solve
        if solver is "pso":
            xopt, gfit = self._cpso(w = w, c1 = c1, c2 = c2, l = l, alpha = 0.,
                                   xstart = xstart, snap = snap)
        elif solver is "cpso":
            xopt, gfit = self._cpso(w = w, c1 = c1, c2 = c2, l = l, alpha = alpha,
                                   delta = delta, xstart = xstart, snap = snap)
        elif solver is "de":
            xopt, gfit = self._de(F = F, CR = CR, xstart = xstart, snap = snap)
        elif solver is "cmaes":
            xopt, gfit = self._cmaes(sigma = sigma, mu_perc = mu_perc,
                                     snap = snap)
        return xopt, gfit
    
    def _init_models(self):
        self._models = np.zeros((self._n_dim, self._popsize, self._max_iter))
        self._energy = np.zeros((self._popsize, self._max_iter))
        return
    
    def _random_model(self):
        return self._lower + np.random.rand(self._n_dim) * (self._upper - self._lower)
    
    def _de(self, F = 1., CR = 0.5, xstart = None, snap = False):
        """
        Minimize an objective function using Differential Evolution (DE).
        
        Parameters
        ----------
        F : scalar, optional, default 1.
            Differential weight.
        CR : scalar, optional, default 0.5
            Crossover probability.
        xstart : None or ndarray, optional, default None
            Initial positions of the population.
        snap : bool, optional, default False
            Save the positions and energy of all individuals at each iteration
            in a 3-D array with shape (n_dim, popsize, max_iter) and 2-D array
            with shape (popsize, max_iter) in attributes 'models' and 'energy'.
            
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
        if self._popsize <= 2:
            self._popsize = 3
            warn("\npopsize cannot be lower than 3 for DE, popsize set to 3", UserWarning)
        if not isinstance(F, float) and not isinstance(F, int) or not 0. <= F <= 2.:
            raise ValueError("F must be an integer or float in [ 0, 2 ], got %s" % F)
        if not isinstance(CR, float) and not isinstance(CR, int) or not 0. <= CR <= 1.:
            raise ValueError("CR must be an integer or float in [ 0, 1 ], got %s" % CR)
        if xstart is not None and isinstance(xstart, np.ndarray) \
            and xstart.shape != (self._n_dim, self._popsize):
            raise ValueError("xstart must be a ndarray of shape (n_dim, popsize)")
        
        # Population initial positions
        X = np.zeros((self._n_dim, self._popsize))
        if xstart is None:
            for i in range(self._popsize):
                X[:,i] = self._random_model()
        else:
            X = np.array(xstart)
        
        # Initialize population
        V = np.zeros((self._n_dim, self._popsize))
        U = np.zeros((self._n_dim, self._popsize))
        pfit = np.zeros(self._popsize)
        self._n_eval = 0
        
        # Compute fitness
        for i in range(self._popsize):
            pfit[i] = self._func(X[:,i])
            self._n_eval += 1
        pbestfit = np.array(pfit)
        if snap:
            self._init_models()
            self._models[:,:,0] = np.array(X)
            self._energy[:,0] = np.array(pbestfit)
        
        # Initialize best individual
        gbidx = np.argmin(pbestfit)
        gfit = pbestfit[gbidx]
        gbest = np.array(X[:,gbidx])
        
        # Iterate until one of the termination criterion is satisfied
        it = 1
        converge = False
        while not converge:
            it += 1
            r1 = np.random.rand(self._n_dim, self._popsize)
            
            # Mutation
            for i in range(self._popsize):
                idx = np.random.permutation(self._popsize)[:4]
                idx = idx[idx != i]
                x1 = np.array(X[:,idx[0]])
                x2 = np.array(X[:,idx[1]])
                x3 = np.array(X[:,idx[2]])
                V[:,i] = x1 + F * (x2 - x3)
                if self._clip:
                    maskl = V[:,i] < self._lower
                    masku = V[:,i] > self._upper
                    V[maskl,i] = self._lower[maskl]
                    V[masku,i] = self._upper[masku]
            
            # Recombination
            irand = np.random.randint(self._n_dim)
            for i in range(self._popsize):
                for j in range(self._n_dim):
                    if r1[j,i] <= CR or j == irand:
                        U[j,i] = V[j,i]
                    else:
                        U[j,i] = X[j,i]
            
            # Compute fitness
            for i in range(self._popsize):
                pfit[i] = self._func(U[:,i])
                self._n_eval += 1
            
            # Selection
            idx = pfit < pbestfit
            pbestfit[idx] = pfit[idx]
            X[:,idx] = U[:,idx]
            if snap:
                self._models[:,:,it-1] = np.array(X)
                self._energy[:,it-1] = np.array(pbestfit)
            
            # Update best individual
            gbidx = np.argmin(pbestfit)
            
            # Stop if best individual position changes less than eps1
            if np.linalg.norm(gbest - X[:,gbidx]) <= self._eps1 \
                and pbestfit[gbidx] <= self._eps2:
                converge = True
                xopt = np.array(X[:,gbidx])
                gfit = pbestfit[gbidx]
                self._flag = 0
            
            # Stop if maximum iteration is reached
            elif it >= self._max_iter:
                converge = True
                xopt = np.array(X[:,gbidx])
                gfit = pbestfit[gbidx]
                self._flag = 1
            
            # Otherwise, update best individual
            else:
                gbest = np.array(X[:,gbidx])
                gfit = pbestfit[gbidx]
        
        self._xopt = xopt
        self._gfit = gfit
        self._n_iter = it
        if snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
        return xopt, gfit
        
    def _cpso(self, w = 0.72, c1 = 1.49, c2 = 1.49, l = 0.1, alpha = 1.25,
             delta = None, xstart = None, snap = False):
        """
        Minimize an objective function using Competitive Particle Swarm
        Optimization (CPSO). Set alpha = 0. for classical PSO.
        
        Parameters
        ----------
        w : scalar, optional, default 0.72
            Inertial weight.
        c1 : scalar, optional, default 1.49
            Cognition parameter.
        c2 : scalar, optional, default 1.49
            Sociability parameter.
        l : scalar, optional, default 0.1
            Velocity clamping percentage.
        alpha : scalar, optional, default 1.25
            Competitivity parameter.
        delta : None or scalar, optional, default None
            Swarm maximum radius.
        xstart : None or ndarray, optional, default None
            Initial positions of the population.
        snap : bool, optional, default False
            Save the positions and energy of all individuals at each iteration
            in a 3-D array with shape (n_dim, popsize, max_iter) and 2-D array
            with shape (popsize, max_iter) in attributes 'models' and 'energy'.
            
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
        .. [3] K. Luu, M. Noble and A. Gesret, *A competitive particle swarm
               optimization for nonlinear first arrival traveltime tomography*,
               In SEG Technical Program Expanded Abstracts 2016 (pp. 2740-2744).
               Society of Exploration Geophysicists.
        """
        # Check inputs
        if not isinstance(w, float) and not isinstance(w, int) or not 0. <= w <= 1.:
            raise ValueError("w must be an integer or float in [ 0, 1 ], got %s" % w)
        if not isinstance(c1, float) and not isinstance(c1, int) or not 0. <= c1 <= 4.:
            raise ValueError("c1 must be an integer or float in [ 0, 4 ], got %s" % c1)
        if not isinstance(c2, float) and not isinstance(c2, int) or not 0. <= c2 <= 4.:
            raise ValueError("c2 must be an integer or float in [ 0, 4 ], got %s" % c2)
        if not isinstance(l, float) and not isinstance(l, int) or not 0. < l <= 1.:
            raise ValueError("l must be an integer or float in ] 0, 1 ], got %s" % l)
        if not isinstance(alpha, float) and not isinstance(alpha, int) or not 0. <= alpha <= 2.:
            raise ValueError("alpha must be an integer or float in [ 0, 2 ], got %s" % alpha)
        if not isinstance(delta, type(None)) and not isinstance(delta, float) and not isinstance(delta, int) and not delta > 0.:
            raise ValueError("delta must be None, a positive integer or float, got %s" % delta)
        if xstart is not None and isinstance(xstart, np.ndarray) \
            and xstart.shape != (self._n_dim, self._popsize):
            raise ValueError("xstart must be a ndarray of shape (n_dim, popsize)")
        
        # Particles initial positions
        X = np.zeros((self._n_dim, self._popsize))
        if xstart is None:
            for i in range(self._popsize):
                X[:,i] = self._random_model()
        else:
            X = np.array(xstart)
        
        # Initialize swarm
        V = np.zeros((self._n_dim, self._popsize))
        pbest = np.array(X)
        pfit = np.zeros(self._popsize)
        pbestfit = np.zeros(self._popsize)
        self._n_eval = 0
        
        # Initialize particle velocity
        vmax = l * (self._upper - self._lower)
        for i in range(self._popsize):
            V[:,i] = vmax * (2. * np.random.rand(self._n_dim) - 1.)
        
        # Compute fitness
        pfit = np.array([ self._func(X[:,i]) for i in range(self._popsize) ])
        self._n_eval += self._popsize
        pbestfit = np.array(pfit)
        if snap:
            self._init_models()
            self._models[:,:,0] = np.array(X)
            self._energy[:,0] = np.array(pbestfit)
        
        # Initialize best individual
        gbidx = np.argmin(pbestfit)
        gfit = pbestfit[gbidx]
        gbest = np.array(X[:,gbidx])
        
        # Swarm maximum radius
        if delta is None:
            delta = 0.08686 * np.log(1. + 0.004 * self._popsize)
        
        # Iterate until one of the termination criterion is satisfied
        it = 1
        converge = False
        while not converge:
            it += 1
            r1 = np.random.rand(self._n_dim, self._popsize)
            r2 = np.random.rand(self._n_dim, self._popsize)
            
            # Update swarm
            for i in range(self._popsize):
                V[:,i] = w*V[:,i] + c1*r1[:,i]*(pbest[:,i]-X[:,i]) \
                                  + c2*r2[:,i]*(gbest-X[:,i])       # Update particle velocity
                X[:,i] = X[:,i] + V[:,i]                            # Update particle position
                if self._clip:
                    maskl = X[:,i] < self._lower
                    masku = X[:,i] > self._upper
                    X[maskl,i] = self._lower[maskl]
                    X[masku,i] = self._upper[masku]
            
            # Compute fitness
            pfit = np.array([ self._func(X[:,i]) for i in range(self._popsize) ])
            self._n_eval += self._popsize
            
            # Update particle best position
            idx = pfit < pbestfit
            pbestfit[idx] = np.array(pfit[idx])
            pbest[:,idx] = np.array(X[:,idx])
            if snap:
                self._models[:,:,it-1] = np.array(X)
                self._energy[:,it-1] = np.array(pfit)
            
            # Update best individual
            gbidx = np.argmin(pbestfit)
            
            # Stop if best individual position changes less than eps1
            if np.linalg.norm(gbest - pbest[:,gbidx]) <= self._eps1 \
                and pbestfit[gbidx] <= self._eps2:
                converge = True
                xopt = np.array(pbest[:,gbidx])
                gfit = pbestfit[gbidx]
                self._flag = 0
            
            # Stop if maximum iteration is reached
            elif it >= self._max_iter:
                converge = True
                xopt = np.array(pbest[:,gbidx])
                gfit = pbestfit[gbidx]
                self._flag = 1
            
            # Otherwise, update best individual
            else:
                gbest = np.array(pbest[:,gbidx])
                gfit = pbestfit[gbidx]
                
            # Competitive PSO algorithm
            if alpha > 0.:
                # Evaluate swarm size
                swarm_radius = np.max([ np.linalg.norm(X[:,i] - gbest)
                                        for i in range(self._popsize) ])
                swarm_radius /= np.linalg.norm(self._upper-self._lower)
                
                # Restart particles if swarm size is lower than threshold
                if swarm_radius < delta:
                    # Rank particles
                    inorm = it / self._max_iter
                    ls = -1. / 0.09
                    nw = int((self._popsize-1.) / (1.+np.exp(-ls*(inorm-alpha+0.5))))
                    idx = pbestfit.argsort()[:-nw-1:-1]
                    
                    # Reset positions, velocities and personal bests
                    V[:,idx] = np.array([ vmax*(2.*np.random.rand(self._n_dim)-1.)
                                            for i in range(nw) ]).transpose()
                    X[:,idx] = np.array([ self._random_model()
                                            for i in range(nw) ]).transpose()
                    pbest[:,idx] = np.array(X[:,idx])
                    
                    # Reset personal best fits
                    pbestfit[idx] = np.array([ self._func(pbest[:,i]) for i in idx ])
                    self._n_eval += nw
                
        self._xopt = np.array(xopt)
        self._gfit = gfit
        self._n_iter = it
        if snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
        return xopt, gfit
        
    def _cmaes(self, sigma = 1., mu_perc = 0.5, xstart = None, snap = False):
        """
        Minimize an objective function using Covariance Matrix Adaptation
        - Evolution Strategy (CMA-ES).
        
        Parameters
        ----------
        sigma : scalar, optional, default 1.
            Step size.
        mu_perc : scalar, optional, default 0.5
            Number of parents as a percentage of population size.
        xstart : None or ndarray, optional, default None
            Initial position of the mean.
        snap : bool, optional, default False
            Save the positions and energy of all individuals at each iteration
            in a 3-D array with shape (n_dim, popsize, max_iter) and 2-D array
            with shape (popsize, max_iter) in attributes 'models' and 'energy'.
            Also save the means, the covariance matrix and the stepsize in
            attributes 'means', 'covar' and 'sigma'.
            
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
        if self._popsize <= 3:
            self._popsize = 4
            warn("\npopsize cannot be lower than 4 for CMA-ES, popsize set to 4", UserWarning)
        if not isinstance(sigma, float) and not isinstance(sigma, int) or sigma <= 0.:
            raise ValueError("sigma must be positive, got %s" % sigma)
        if not isinstance(mu_perc, float) and not isinstance(mu_perc, int) or not 0. < mu_perc <= 1.:
            raise ValueError("mu_perc must be an integer or float in ] 0, 1 ], got %s" % mu_perc)
        if xstart is not None and (isinstance(xstart, list) or isinstance(xstart, np.ndarray)) \
            and len(xstart) != self._n_dim:
            raise ValueError("xstart must be a list or ndarray of length n_dim")
        
        # Initialize saved outputs
        if snap:
            self._init_models()
            self._means = np.zeros((self._n_dim, self._max_iter))
            self._covar = np.zeros((self._n_dim, self._n_dim, self._max_iter))
            self._sigma = np.zeros(self._max_iter)
        
        # Population initial positions
        if xstart is None:
            xmean = self._random_model()
        else:
            xmean = np.array(xstart)
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
        bnd_scale = np.zeros(self._n_dim)
        dfithist = np.array([ 1. ])
        
        # (mu, lambda)-CMA-ES
        it = 0
        self._n_eval = 0
        eigeneval = 0
        arx = np.zeros((self._n_dim, self._popsize))
        arxvalid = np.zeros((self._n_dim, self._popsize))
        arfitness = np.zeros(self._popsize)
        arbestfitness = np.zeros(self._max_iter)
        ilim = int(10 + 30 * self._n_dim / self._popsize)
        insigma = sigma
        validfitval = False
        iniphase = True
        converge = False
        
        while not converge:
            it += 1
            
            # Generate lambda offsprings
            for k in range(self._popsize):
                arx[:,k] = xmean + sigma * np.dot(B, D*np.random.randn(self._n_dim))
                arxvalid[:,k] = np.array(arx[:,k])
                if self._clip:
                    maskl = arxvalid[:,k] < self._lower
                    masku = arxvalid[:,k] > self._upper
                    arxvalid[maskl,k] = self._lower[maskl]
                    arxvalid[masku,k] = self._upper[masku]
                
            # Evaluate fitness
            for k in range(self._popsize):
                arfitness[k] = self._func(arxvalid[:,k])
                self._n_eval += 1
            if snap:
                self._models[:,:,it-1] = np.array(arx)
                self._energy[:,it-1] = np.array(arfitness)
                self._means[:,it-1] = np.array(xmean)
                
            # Handle boundaries by penalizing fitness
            # Get delta fitness values
            perc = np.percentile(arfitness, [ 25, 75 ], interpolation = "midpoint")
            delta = ( perc[1] - perc[0] ) / self._n_dim / np.mean(np.diag(C)) / sigma**2
            
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
            ti = np.logical_or(xmean < self._lower, xmean > self._upper)
            tx = np.array(xmean)
            maskl = tx < self._lower
            masku = tx > self._upper
            tx[maskl] = self._lower[maskl]
            tx[masku] = self._upper[masku]
            
            # Set initial weights
            if iniphase:
                if np.any(ti):
                    bnd_weights.fill(2.0002 * np.median(dfithist))
                    if validfitval and it > 2:
                        iniphase = False
                        
            if np.any(ti):
                tx = xmean - tx
                idx = np.logical_and(ti, np.abs(tx) > 3. * max( 1., np.sqrt(self._n_dim/mueff) ) \
                                     * sigma * np.sqrt(np.diag(C)))
                idx = np.logical_and(idx, np.sign(tx) == np.sign(xmean - xold))
                for i in range(self._n_dim):
                    if idx[i]:
                        bnd_weights[i] *= 1.2**min(1., mueff/10./self._n_dim)
                        
            # Calculate scaling biased to unity, product is one
            bnd_scale = np.exp( 0.9 * ( np.log(np.diag(C)) - np.mean(np.log(np.diag(C))) ) )
            
            # Assigned penalized fitness
            arfitness += np.dot(bnd_weights / bnd_scale, (arxvalid - arx)**2)
            
            # Sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(arfitness)
            xold = np.array(xmean)
            xmean = np.dot(arx[:,arindex[:mu]], weights)
            
            # Save best fitness
            arbestfitness[it-1] = arfitness[arindex[0]]
            
            # Cumulation
            ps = ( 1. - cs ) * ps \
                 + np.sqrt( cs * ( 2. - cs ) * mueff ) * np.dot(invsqrtC, xmean - xold) / sigma
            if np.linalg.norm(ps) / np.sqrt( 1. - ( 1. - cs )**(2.*self._n_eval/self._popsize) ) / chind < 1.4 + 2. / ( self._n_dim + 1. ):
                hsig = 1.
            else:
                hsig = 0.
            pc = ( 1. - cc ) * pc \
                 + hsig * np.sqrt( cc * ( 2. - cc ) * mueff ) * (xmean - xold) / sigma
                 
            # Adapt covariance matrix C
            artmp = ( arx[:,arindex[:mu]] - np.tile(xold[:,None], mu) ) / sigma
            C = ( 1. - c1 - cmu ) * C \
                + c1 * ( np.dot(pc[:,None], pc[None,:]) \
                         + ( 1. - hsig ) * cc * ( 2. - cc ) * C ) \
                + cmu * np.dot(np.dot(artmp, np.diag(weights)), np.transpose(artmp))
                
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
                
            # Save outputs
            if snap:
                self._sigma[it-1] = sigma
                self._covar[:,:,it-1] = np.array(C)
                
            # Stop if mean position changes less than eps1
            if np.linalg.norm(xold - xmean) <= self._eps1 \
                and arfitness[arindex[0]] < self._eps2:
                converge = True
                self._flag = 0
                
            # Stop if maximum iteration is reached
            if it >= self._max_iter:
                converge = True
                self._flag = 1
                
            # NoEffectAxis: stop if numerical precision problem
            i = int(np.floor(np.mod(it, self._n_dim)))
            if np.all( np.abs(0.1 * sigma * B[:,i] * D[i]) < 1e-10 ):
                converge = True
                self._flag = 2
                
            # NoEffectCoord: stop if too low coordinate axis deviations
            if np.any( 0.2 * sigma * np.sqrt(np.diag(C)) < 1e-10 ):
                converge = True
                self._flag = 3
            
            # ConditionCov: stop if the condition number exceeds 1e14
            if np.max(D) > 1e7 * np.min(D):
                converge = True
                self._flag = 4
            
            # EqualFunValues: stop if the range of fitness values is zero
            if it >= ilim:
                if np.max(arbestfitness[it-ilim:it+1]) - np.min(arbestfitness[it-ilim:it+1]) < 1e-10:
                    converge = True
                    self._flag = 5
                    
            # TolXUp: stop if x-changes larger than 1e3 times initial sigma
            if np.any( sigma * np.sqrt(np.diag(C)) > 1e3 * insigma ):
                converge = True
                self._flag = 6
                
            # TolFun: stop if fun-changes smaller than 1e-12
            if it > 2 and np.max(np.append(arfitness, arbestfitness)) - np.min(np.append(arfitness, arbestfitness)) < 1e-12:
                converge = True
                self._flag = 7
                
            # TolX: stop if x-changes smaller than 1e-11 times inigial sigma
            if np.all( sigma * np.max(np.append(np.abs(pc), np.sqrt(np.diag(C)))) < 1e-11 * insigma ):
                converge = True
                self._flag = 8
                
        xopt = arx[:,arindex[0]]
        gfit = arfitness[arindex[0]]
        self._xopt = np.array(xopt)
        self._fit = gfit
        self._n_iter = it
        if snap:
            self._models = self._models[:,:,:it]
            self._energy = self._energy[:,:it]
            self._means = self._means[:,:it]
            self._covar = self._covar[:,:,:it]
            self._sigma = self._sigma[:it]
        return xopt, gfit
    
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
        Stopping criterion:
            - 0, best individual position changes less than eps1.
            - 1, maximum number of iterations is reached.
            - 2, NoEffectAxis (only when solver = 'cmaes').
            - 3, NoEffectCoord (only when solver = 'cmaes').
            - 4, ConditionCov (only when solver = 'cmaes').
            - 5, EqualFunValues (only when solver = 'cmaes').
            - 6, TolXUp (only when solver = 'cmaes').
            - 7, TolFun (only when solver = 'cmaes').
            - 8, TolX (only when solver = 'cmaes').
        """
        return self._flag
    
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
        ndarray of shape (n_dim, popsize, max_iter)
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
        ndarray of shape (n_dim, max_iter)
        Mean models at every iterations. Available only when solver = 'cmaes'
        and snap = True.
        """
        return self._means
        
    @property
    def covar(self):
        """
        ndarray of shape (n_dim, n_dim, max_iter)
        Adapted covariance matrix at every iterations. Available only when
        solver = 'cmaes' and snap = True.
        """
        return self._covar
    
    @property
    def sigma(self):
        """
        ndarray of shape (max_iter)
        Step size at every iterations. Available only when solver = 'cmaes' and
        snap = True.
        """
        return self._sigma