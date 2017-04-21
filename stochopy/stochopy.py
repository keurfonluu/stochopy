# -*- coding: utf-8 -*-

"""
StochOPy (STOCHastic OPtimization for PYthon) provides user-friendly routines
to sample or optimize objective functions with the most popular algorithms.
"""

import numpy as np


__all__ = [ "MonteCarlo", "Evolutionary" ]


# Object MonteCarlo
#===================
class MonteCarlo:
    """
    Monte-Carlo sampler.
    
    This sampler explores the parameter space using pure Monte-Carlo,
    Metropolis-Hastings algorithm or Hamiltonian (Hybrid) Monte-Carlo.
    
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
    max_iter : int, optional, default 1000
        Number of models to sample.
    random_state : int, optional, default None
        Seed for random number generator.
    """
    
    def __init__(self, func, lower = None, upper = None, n_dim = 1,
                 max_iter = 1000, random_state = None, args = (), kwargs = {}):
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
                raise ValueError("lower and upper have different size")
            if np.any(upper < lower):
                raise ValueError("lower greater than upper")
            self._lower = np.array(lower)
            self._upper = np.array(upper)
            self._n_dim = len(lower)
        else:
            self._lower = np.full(n_dim, -1.)
            self._upper = np.full(n_dim, 1.)
            self._n_dim = n_dim
        if max_iter <= 0:
            raise ValueError("max_iter cannot be zero or negative")
        else:
            self._max_iter = max_iter
        if random_state is not None:
            np.random.seed(random_state)
        return
    
    def sample(self, sampler = "hastings", stepsize = 1., xstart = None,
               n_leap = 10, fprime = None, delta = 1e-3, snap_leap = False,
               args = (), kwargs = {}):
        """
        Sample the parameter space using pure Monte-Carlo,
        Metropolis-Hastings algorithm or Hamiltonian (Hybrid) Monte-Carlo.
        
        Parameters
        ----------
        sampler : {'pure', 'hastings', 'hamiltonian'}, default 'hastings'
            Sampling method.
            - 'pure', uniform sampling in the search space [ lower, upper ].
            - 'hastings', random-walk with a gaussian perturbation.
            - 'hamiltonian', propose a new sample simulated with hamiltonian
              dynamics.
        stepsize : scalar, optional, default 1.
            If sampler = 'pure', 'xstart' is not used.
            If sampler = 'hastings', standard deviation of gaussian
            perturbation.
            If sampler = 'hamiltonian', leap-frog step size.
        xstart : None or ndarray, optional, default None
            First model of the Markov chain. If sampler = 'pure', 'xstart'
            is not used.
        n_leap : int, optional, default 10
            Number of leap-frog steps. Only used when sampler = 'hamiltonian'.
        fprime : callable, optional, default None
            Gradient of the objective function. If necessary, the variables
            required for its computation should be passed in 'args' and/or
            'kwargs'. If 'fprime' is None, the gradient is computed numerically
            with a centred finite-difference scheme. Only used when
            sampler = 'hamiltonian'.
        delta : scalar, optional, default 1e-3
            Discretization size of the numerical gradient. Only used when
            'fprime' is None. Only used when sampler = 'hamiltonian'.
        snap_leap : bool, optional, default False
            Save the leap-frog positions in a 3-D array with shape
            (n_dim, n_leap+1, max_iter-1) in an attribute 'leap_frog'. For
            visualization purpose only. Only used when sampler = 'hamiltonian'.
            
        Returns
        -------
        xopt : ndarray
            Maximum a posteriori (MAP) model.
        gfit : scalar
            Energy of the MAP model.
        
        Examples
        --------
        Import the module and define the objective function (Sphere):
        
        >>> import numpy as np
        >>> from stochopy import MonteCarlo
        >>> f = lambda x: np.sum(x**2)
        
        Define the search space boundaries in 2-D:
        
        >>> n_dim = 2
        >>> lower = np.full(n_dim, -5.12)
        >>> upper = np.full(n_dim, 5.12)
        
        Initialize the Monte-Carlo sampler:
        
        >>> max_iter = 1000
        >>> mc = MonteCarlo(f, lower = lower, upper = upper,
                            max_iter = max_iter)
        
        Pure Monte-Carlo:
        
        >>> xopt, gfit = mc.sample(sampler = "pure")
        
        Monte-Carlo Markov-Chain (Metropolis-Hastings):
        
        >>> xopt, gfit = mc.sample(sampler = "hastings", stepsize = 0.8)
        
        Hamiltonian (Hybrid) Monte-Carlo:
        >>> xopt, gfit = mc.sample(sampler = "hamiltonian", stepsize = 0.1,
                                   n_leap = 20)
        
        HMC with custom gradient and xstart:
        
        >>> grad = lambda x: 2.*x
        >>> x0 = np.array([ 2., 2. ])
        >>> xopt, gfit = mc.sample(sampler = "hamiltonian", stepsize = 0.1,
                                   n_leap = 20, fprime = grad, xstart = x0)
        """
        # Check inputs
        if not isinstance(sampler, str) and  sampler not in [ "pure", "hastings", "hamiltonian" ]:
            raise ValueError("unknown sampler '" + str(sampler) + "'")
        if xstart is not None and len(xstart) != self._n_dim:
            raise ValueError("xstart dimension mismatches n_dim")
        if stepsize <= 0:
            raise ValueError("stepsize cannot be zero or negative")
        
        # Initialize
        self._solver = sampler
        self._init_models()
        
        # Sample
        if sampler is "pure":
            xopt, gfit = self._pure()
        elif sampler is "hastings":
            xopt, gfit = self._hastings(stepsize = stepsize, xstart = xstart)
        elif sampler is "hamiltonian":
            xopt, gfit = self._hamiltonian(fprime = fprime,
                                           stepsize = stepsize,
                                           n_leap = n_leap,
                                           xstart = xstart,
                                           delta = delta,
                                           snap_leap = snap_leap,
                                           args = (), kwargs = {})
        return xopt, gfit
    
    def _init_models(self):
        self._models = np.zeros((self._n_dim, self._max_iter))
        self._energy = np.zeros(self._max_iter)
        return
    
    def _random_model(self):
        return self._lower + np.random.rand(self._n_dim) * (self._upper - self._lower)
    
    def _best_model(self):
        idx = np.argmin(self._energy)
        return self._models[:,idx], self._energy[idx]
        
    def _pure(self):
        """
        Sample the parameter space using the a pure Monte-Carlo algorithm.
            
        Returns
        -------
        xopt : ndarray
            Maximum a posteriori (MAP) model.
        gfit : scalar
            Energy of the MAP model.
        """
        for i in range(self._max_iter):
            self._models[:,i] = self._random_model()
            self._energy[i] = self._func(self._models[:,i])
        return self._best_model()
        
    def _hastings(self, stepsize = 1., xstart = None):
        """
        Sample the parameter space using the Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        stepsize : scalar, optional, default 1.
            Standard deviation of gaussian perturbation.
        xstart : None or ndarray, optional, default None
            First model of the Markov chain.
            
        Returns
        -------
        xopt : ndarray
            Maximum a posteriori (MAP) model.
        gfit : scalar
            Energy of the MAP model.
        
        Notes
        -----
        A rule-of-thumb for proper sampling is:
         -  if n_dim <= 2 : acceptance ratio of 50%
         -  otherwise : acceptance ratio of 25%
        The acceptance ratio is given by the attribute 'acceptance_ratio'.
        """
        # Initialize models
        if xstart is None:
            self._models[:,0] = self._random_model()
        else:
            self._models[:,0] = np.array(xstart)
        self._energy[0] = self._func(self._models[:,0])
        
        # Metropolis-Hastings algorithm
        rejected = 0
        for i in range(1, self._max_iter):
            r1 = np.random.randn(self._n_dim)
            self._models[:,i] = self._models[:,i-1] + r1 * stepsize
            self._energy[i] = self._func(self._models[:,i])
            
            log_alpha = min(0., self._energy[i-1] - self._energy[i])
            if log_alpha < np.log(np.random.rand()):
                rejected += 1
                self._models[:,i] = self._models[:,i-1]
                self._energy[i] = self._energy[i-1]
        self._acceptance_ratio = 1. - rejected / self._max_iter
                
        # Return best model
        return self._best_model()
        
    def _hamiltonian(self, fprime = None, stepsize = 1., n_leap = 10, xstart = None,
                    delta = 1e-3, snap_leap = False, args = (), kwargs = {}):
        """
        Sample the parameter space using the Hamiltonian (Hybrid) Monte-Carlo
        algorithm.
        
        Parameters
        ----------
        fprime : callable, optional, default None
            Gradient of the objective function. If necessary, the variables
            required for its computation should be passed in 'args' and/or
            'kwargs'. If 'fprime' is None, the gradient is computed numerically
            with a centred finite-difference scheme.
        stepsize : scalar, optional, default 1.
            Leap-frog step size.
        n_leap : int, optional, default 10
            Number of leap-frog steps.
        xstart : None or ndarray, optional, default None
            First model of the Markov chain.
        delta : scalar, optional, default 1e-3
            Discretization size of the numerical gradient. Only used when
            'fprime' is None.
        snap_leap : bool, optional, default False
            Save the leap-frog positions in a 3-D array with shape
            (n_dim, n_leap+1, max_iter-1) in an attribute 'leap_frog'. For
            visualization purpose only.
            
        Returns
        -------
        xopt : ndarray
            Maximum a posteriori (MAP) model.
        gfit : scalar
            Energy of the MAP model.
            
        References
        ----------
        .. [1] S. Duane, A. D. Kennedy, B. J. Pendleton and D. Roweth, *Hybrid
               Monte Carlo*, Physics Letters B., 1987, 195(2): 216-222
        .. [2] N. Radford, *MCMC Using Hamiltonian Dynamics*, Handbook of
               Markov Chain Monte Carlo, Chapman and Hall/CRC, 2011
        """
        # Check inputs
        if fprime is None:
            grad = lambda x: self._approx_grad(x, delta)
        else:
            if not hasattr(fprime, "__call__"):
                raise ValueError("fprime is not callable")
            else:
                grad = lambda x: fprime(x, *args, **kwargs)
        if n_leap <= 0:
            raise ValueError("n_leap cannot be zero or negative")
        
        # Initialize models
        if xstart is None:
            self._models[:,0] = self._random_model()
        else:
            self._models[:,0] = np.array(xstart)
        self._energy[0] = self._func(self._models[:,0])
        
        # Save leap frog trajectory
        if snap_leap:
            self._leap_frog = np.zeros((self._n_dim, n_leap+1, self._max_iter-1))
        
        # Leap-frog algorithm
        rejected = 0
        for i in range(1, self._max_iter):
            q = np.array(self._models[:,i-1])
            p = np.random.randn(self._n_dim)            # Random momentum
            q0, p0 = np.array(q), np.array(p)
            if snap_leap:
                self._leap_frog[:,0,i-1] = np.array(q)
            
            p -= 0.5 * stepsize * grad(q)               # First half momentum step
            q += stepsize * p                           # First full position step
            for l in range(n_leap):
                p -= stepsize * grad(q)                 # Momentum
                q += stepsize * p                       # Position
                if snap_leap:
                    self._leap_frog[:,l+1,i-1] = np.array(q)
            p -= 0.5 * stepsize * grad(q)               # Last half momentum step
            
            U0 = self._func(q0)
            K0 = 0.5 * np.sum(p0**2)
            U = self._func(q)
            K = 0.5 * np.sum(p**2)
            log_alpha = min(0., U0 - U + K0 - K)
            if log_alpha < np.log(np.random.rand()):
                rejected += 1
                self._models[:,i] = self._models[:,i-1]
                self._energy[i] = self._energy[i-1]
            else:
                self._models[:,i] = q
                self._energy[i] = U
        self._acceptance_ratio = 1. - rejected / self._max_iter
        
        # Return best model
        return self._best_model()
    
    def _approx_grad(self, x, delta = 1e-3):
        grad = np.zeros(self._n_dim)
        for i in range(self._n_dim):
            x1, x2 = np.array(x), np.array(x)
            x1[i] -= delta
            x2[i] += delta
            grad[i] = 0.5 * ( self._func(x2) - self._func(x1) ) / delta
        return grad
    
    @property
    def models(self):
        """
        ndarray of shape (n_dim, max_iter)
        Sampled models.
        """
        return self._models
    
    @property
    def energy(self):
        """
        ndarray of shape (max_iter)
        Energy of sampled models.
        """
        return self._energy
    
    @property
    def acceptance_ratio(self):
        """
        scalar between 0 and 1
        Acceptance ratio of sampler. Not available when sampler = 'pure'.
        """
        return self._acceptance_ratio
    
    @property
    def leap_frog(self):
        """
        ndarray of shape (n_leap, n_leap+1, max_iter-1)
        Leap frog positions. Available only when sampler = 'hamiltonian' and
        snap_leap = True.
        """
        return self._leap_frog


# Object Evolutionary
#=====================
class Evolutionary:
    """
    Evolutionary Algorithm optimizer.
    
    This optimizer minimizes an objective function using Differential
    Evolution (DE), Particle Swarm Optimization (PSO) or Covariance Matrix
    Adaptation - Evolution Strategy (CMA-ES).
    
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
    popsize : int, optional, default 4
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
                 popsize = 4, max_iter = 100, eps1 = 1e-8, eps2 = 1e-8,
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
                raise ValueError("lower and upper have different size")
            if np.any(upper < lower):
                raise ValueError("lower greater than upper")
            self._lower = np.array(lower)
            self._upper = np.array(upper)
            self._n_dim = len(lower)
        else:
            self._lower = np.full(n_dim, -1.)
            self._upper = np.full(n_dim, 1.)
            self._n_dim = n_dim
        if max_iter <= 0:
            raise ValueError("max_iter cannot be zero or negative")
        else:
            self._max_iter = max_iter
        if popsize < 2:
            raise ValueError("popsize cannot be lower than 2")
        else:
            self._popsize = int(popsize)
        self._eps1 = eps1
        self._eps2 = eps2
        self._clip = clip
        if random_state is not None:
            np.random.seed(random_state)
        return
    
    def optimize(self, solver = "pso", xstart = None, w = 0.72, c1 = 1.49,
                 c2 = 1.49, l = 0.1, F = 1., CR = 0.5, sigma = 1.,
                 mu_perc = 0.5, snap = False):
        """
        Minimize an objective function using Differential Evolution (DE),
        Particle Swarm Optimization (PSO) or Covariance Matrix Adaptation
        - Evolution Strategy (CMA-ES).
        
        Parameters
        ----------
        solver : {'de', 'pso', 'cmaes'}, default 'pso'
            Optimization method.
            - 'de', Differential Evolution.
            - 'pso', Particle Swarm Optimization.
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
        if not isinstance(solver, str) and solver not in [ "pso", "de", "cmaes" ]:
            raise ValueError("unknown solver '" + str(solver) + "'")
        
        # Initialize
        self._solver = solver
        self._init_models()
        
        # Solve
        if solver is "pso":
            xopt, gfit = self._pso(w = w, c1 = c1, c2 = c2, xstart = xstart,
                                   snap = snap)
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
            raise Warning("popsize cannot be lower than 3 for DE, popsize set to 3")
        if not 0. <= F <= 2.:
            raise ValueError("F is not in [ 0, 2 ]")
        if not 0. <= CR <= 1.:
            raise ValueError("CR is not in [ 0, 1 ]")
        if xstart is not None and xstart.shape != (self._n_dim, self._popsize):
            raise ValueError("xstart dimension mismatches n_dim and popsize")
        
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
        
    def _pso(self, w = 0.72, c1 = 1.49, c2 = 1.49, l = 0.1, xstart = None,
             snap = False):
        """
        Minimize an objective function using Particle Swarm Optimization (PSO).
        
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
        """
        # Check inputs
        if not 0. <= w <= 1.:
            raise ValueError("w is not in [ 0, 1 ]")
        if not 0. <= c1 <= 4.:
            raise ValueError("c1 is not in [ 0, 4 ]")
        if not 0. <= c2 <= 4.:
            raise ValueError("c2 is not in [ 0, 4 ]")
        if xstart is not None and xstart.shape != (self._n_dim, self._popsize):
            raise ValueError("xstart dimension mismatches n_dim and popsize")
        
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
            for i in range(self._popsize):
                pfit[i] = self._func(X[:,i])
                self._n_eval += 1
            
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
            raise Warning("popsize cannot be lower than 4 for CMA-ES, popsize set to 4")
        if sigma <= 0:
            raise ValueError("sigma cannot be zero or negative")
        if not 0. < mu_perc <= 1.:
            raise ValueError("mu_perc is not in ] 0, 1 ]")
        if xstart is not None and len(xstart) != self._n_dim:
            raise ValueError("xstart dimension mismatches n_dim")
        
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
                    bnd_weights = 2.0002 * np.median(dfithist)
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
            - 2, NoEffectAxis (only if solver = 'cmaes').
            - 3, NoEffectCoord (only if solver = 'cmaes').
            - 4, ConditionCov (only if solver = 'cmaes').
            - 5, EqualFunValues (only if solver = 'cmaes').
            - 6, TolXUp (only if solver = 'cmaes').
            - 7, TolFun (only if solver = 'cmaes').
            - 8, TolX (only if solver = 'cmaes').
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