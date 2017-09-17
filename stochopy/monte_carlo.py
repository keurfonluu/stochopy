# -*- coding: utf-8 -*-

"""
Monte-Carlo algorithms are methods that randomly sample the model parameter
space.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np

__all__ = [ "MonteCarlo" ]


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
    constrain : bool, optional, default True
        Accept sample only within search space.
    random_state : int, optional, default None
        Seed for random number generator.
    """
    
    _ATTRIBUTES = [ "solution", "fitness", "acceptance_ratio" ]
    
    def __init__(self, func, lower = None, upper = None, n_dim = 1,
                 max_iter = 1000, constrain = True, random_state = None,
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
            self._lower = np.full(n_dim, -1.)
            self._upper = np.full(n_dim, 1.)
            self._n_dim = n_dim
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer, got %s" % max_iter)
        else:
            self._max_iter = max_iter
        if not isinstance(constrain, bool):
            raise ValueError("constrain must be either True or False, got %s" % constrain)
        else:
            self._constrain = constrain
        if random_state is not None:
            np.random.seed(random_state)
        return
    
    def __repr__(self):
        attributes = [ "%s: %s" % (attr.rjust(13), self._print_attr(attr))
                        for attr in self._ATTRIBUTES ]
        return "\n".join(attributes) + "\n"
    
    def _print_attr(self, attr):
        if attr not in self._ATTRIBUTES:
            raise ValueError("attr should be either 'solution', 'fitness' or 'acceptance_ratio'")
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
            elif attr == "acceptance_ratio":
                return "%.2f" % self._acceptance_ratio
    
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
        stepsize : scalar or ndarray, optional, default 1.
            If sampler = 'pure', 'stepsize' is not used.
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
        if not isinstance(sampler, str) or sampler not in [ "pure", "hastings", "hamiltonian" ]:
            raise ValueError("sampler must either be 'pure', 'hastings' or 'hamiltonian', got %s" % sampler)
        if xstart is not None and (isinstance(xstart, list) or isinstance(xstart, np.ndarray)) \
            and len(xstart) != self._n_dim:
            raise ValueError("xstart must be a list or ndarray of length n_dim")
        if sampler == "hastings":
            if not isinstance(stepsize, (float, int, list, np.ndarray)):
                raise ValueError("stepsize must be a float, integer, list or ndarray")
            else:
                if isinstance(stepsize, (float, int)) and stepsize <= 0.:
                    raise ValueError("stepsize must be positive, got %s" % stepsize)
                elif isinstance(stepsize, (list, np.ndarray)) and np.any([ s <= 0 for s in stepsize ]):
                    raise ValueError("elements in stepsize must be positive")
                elif isinstance(stepsize, (list, np.ndarray)) and len(stepsize) != self._n_dim:
                    raise ValueError("stepsize must be a list or ndarray of length n_dim")
        elif sampler == "hamiltonian":
            if not isinstance(stepsize, (float, int)) or stepsize <= 0.:
                raise ValueError("stepsize must be positive, got %s" % stepsize)
        
        # Initialize
        self._solver = sampler
        self._init_models()
        
        # Sample
        if sampler == "pure":
            xopt, gfit = self._pure()
        elif sampler == "hastings":
            xopt, gfit = self._hastings(stepsize = stepsize, xstart = xstart)
        elif sampler == "hamiltonian":
            xopt, gfit = self._hamiltonian(fprime = fprime,
                                           stepsize = stepsize,
                                           n_leap = n_leap,
                                           xstart = xstart,
                                           delta = delta,
                                           snap_leap = snap_leap,
                                           args = args, kwargs = kwargs)
        return xopt, gfit
    
    def _init_models(self):
        self._models = np.zeros((self._max_iter, self._n_dim))
        self._energy = np.zeros(self._max_iter)
        return
    
    def _random_model(self):
        return self._lower + np.random.rand(self._n_dim) * (self._upper - self._lower)
    
    def _best_model(self):
        idx = np.argmin(self._energy)
        return self._models[idx,:], self._energy[idx]
        
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
        self._models = np.array([ self._random_model() for i in range(self._max_iter) ])
        self._energy = np.array([ self._func(self._models[i,:]) for i in range(self._max_iter) ])
        xopt, gfit = self._best_model()
        self._xopt = xopt
        self._gfit = gfit
        return xopt, gfit
        
    def _hastings(self, stepsize = 1., xstart = None):
        """
        Sample the parameter space using the Metropolis-Hastings algorithm.
        
        Parameters
        ----------
        stepsize : scalar or ndarray, optional, default 1.
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
            self._models[0,:] = self._random_model()
        else:
            self._models[0,:] = np.array(xstart)
        self._energy[0] = self._func(self._models[0,:])
        
        # Metropolis-Hastings algorithm
        rejected = 0
        for i in range(1, self._max_iter):
            r1 = np.random.randn(self._n_dim)
            self._models[i,:] = self._models[i-1,:] + r1 * stepsize
            self._energy[i] = self._func(self._models[i,:])
            
            log_alpha = min(0., self._energy[i-1] - self._energy[i])
            if log_alpha < np.log(np.random.rand()) \
                or not self._in_search_space(self._models[i,:]):
                rejected += 1
                self._models[i,:] = self._models[i-1,:]
                self._energy[i] = self._energy[i-1]
        self._acceptance_ratio = 1. - rejected / self._max_iter
                
        # Return best model
        xopt, gfit = self._best_model()
        self._xopt = xopt
        self._gfit = gfit
        return xopt, gfit
        
    def _hamiltonian(self, fprime = None, stepsize = 0.1, n_leap = 10, xstart = None,
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
        stepsize : scalar, optional, default 0.1
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
        if not isinstance(n_leap, int) or n_leap <= 0:
            raise ValueError("n_leap must be a positive integer, got %s" % n_leap)
        
        # Initialize models
        if xstart is None:
            self._models[0,:] = self._random_model()
        else:
            self._models[0,:] = np.array(xstart)
        self._energy[0] = self._func(self._models[0,:])
        
        # Save leap frog trajectory
        if snap_leap:
            self._leap_frog = np.zeros((self._max_iter-1, self._n_dim, n_leap+1))
        
        # Leap-frog algorithm
        rejected = 0
        for i in range(1, self._max_iter):
            q = np.array(self._models[i-1,:])
            p = np.random.randn(self._n_dim)            # Random momentum
            q0, p0 = np.array(q), np.array(p)
            if snap_leap:
                self._leap_frog[i-1,:,0] = np.array(q)
            
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
            if log_alpha < np.log(np.random.rand()) \
                or not self._in_search_space(self._models[i,:]):
                rejected += 1
                self._models[i,:] = self._models[i-1,:]
                self._energy[i] = self._energy[i-1]
            else:
                self._models[i,:] = q
                self._energy[i] = U
        self._acceptance_ratio = 1. - rejected / self._max_iter
        
        # Return best model
        xopt, gfit = self._best_model()
        self._xopt = xopt
        self._gfit = gfit
        return xopt, gfit
    
    def _approx_grad(self, x, delta = 1e-3):
        grad = np.zeros(self._n_dim)
        for i in range(self._n_dim):
            x1, x2 = np.array(x), np.array(x)
            x1[i] -= delta
            x2[i] += delta
            grad[i] = 0.5 * ( self._func(x2) - self._func(x1) ) / delta
        return grad
    
    def _in_search_space(self, x):
        if self._constrain:
            return np.logical_and(np.all(x <= self._upper), np.all(x >= self._lower))
        else:
            return True
        
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
    def models(self):
        """
        ndarray of shape (max_iter, n_dim)
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
        ndarray of shape (max_iter-1, n_dim, n_leap+1)
        Leap frog positions. Available only when sampler = 'hamiltonian' and
        snap_leap = True.
        """
        return self._leap_frog