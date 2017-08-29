# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import unittest
if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from evolutionary_algorithm import Evolutionary
else:
    from stochopy import Evolutionary


class EvolutionaryTest(unittest.TestCase):
    """
    Evolutionary algorithms unit tests.
    """
    
    def setUp(self):
        """
        Initialize evolutionary algorithm tests.
        """
        func = lambda x: 100*np.sum((x[1:]-x[:-1]**2)**2)+np.sum((1-x[:-1])**2)
        n_dim = 2
        lower = np.full(n_dim, -5.12)
        upper = np.full(n_dim, 5.12)
        popsize = int(4 + np.floor(3.*np.log(n_dim)))
        max_iter = 50
        random_state = 42
        self.optimizer = Evolutionary(func, lower = lower, upper = upper,
                                      popsize = popsize, max_iter = max_iter,
                                      snap = True, random_state = random_state)
        self.n_dim = n_dim
        self.popsize = popsize
        
    def tearDown(self):
        """
        Cleaning after each test.
        """
        del self.optimizer
        
    def test_pso(self):
        """
        Particle Swarm Optimization test.
        """
        w = 0.42
        c1 = 1.409
        c2 = 1.991
        l = 0.26
        xopt, gfit = self.optimizer.optimize(solver = "pso", w = w, c1 = c1,
                                             c2 = c2, l = l)
        xopt_true = np.array([ 1.46014373, 2.13358457 ])
        for i, val in enumerate(xopt):
            self.assertAlmostEqual(val, xopt_true[i])
            
    def test_pso_param_zero(self):
        """
        Checking PSO behaviour when all the parameters are set to 0.
        """
        xstart = np.random.rand(self.popsize, self.n_dim)
        w = 0.
        c1 = 0.
        c2 = 0.
        self.optimizer.optimize(solver = "pso", w = w, c1 = c1, c2 = c2,
                                xstart = xstart)
        for i, val in enumerate(self.optimizer.models[:,:,-1].ravel()):
            self.assertAlmostEqual(val, xstart.ravel()[i])
            
    def test_cpso(self):
        """
        Competitive Particle Swarm Optimization test.
        """
        w = 0.42
        c1 = 1.409
        c2 = 1.991
        l = 0.26
        gamma = 0.8
        delta = 0.1
        xopt, gfit = self.optimizer.optimize(solver = "cpso", w = w, c1 = c1,
                                             c2 = c2, l = l, gamma = gamma,
                                             delta = delta)
        xopt_true = np.array([ 1.46014373, 2.13358457 ])
        for i, val in enumerate(xopt):
            self.assertAlmostEqual(val, xopt_true[i])
            
    def test_cpso_param_zero(self):
        """
        Checking CPSO behaviour when all the parameters are set to 0.
        """
        xstart = np.random.rand(self.popsize, self.n_dim)
        w = 0.
        c1 = 0.
        c2 = 0.
        gamma = 0.
        self.optimizer.optimize(solver = "pso", w = w, c1 = c1, c2 = c2, 
                                xstart = xstart)
        pso_models = self.optimizer.models[:,:,-1]
        self.optimizer.optimize(solver = "cpso", w = w, c1 = c1, c2 = c2,
                                xstart = xstart, gamma = gamma)
        cpso_models = self.optimizer.models[:,:,-1]
        for i, val in enumerate(cpso_models.ravel()):
            self.assertAlmostEqual(val, pso_models.ravel()[i])
            
    def test_de(self):
        """
        Differential Evolution test.
        """
        CR = 0.42
        F = 1.491
        xopt, gfit = self.optimizer.optimize(solver = "de", CR = CR, F = F)
        xopt_true = np.array([ 1.35183858, 1.81825907 ])
        for i, val in enumerate(xopt):
            self.assertAlmostEqual(val, xopt_true[i])
        
    def test_cmaes(self):
        """
        Covariance Matrix Adaptation - Evolution Strategy test.
        """
        sigma = 0.1
        mu_perc = 0.2
        xstart = np.array([ -3., -3. ])
        xopt, gfit = self.optimizer.optimize(solver = "cmaes", sigma = sigma,
                                             mu_perc = mu_perc, xstart = xstart)
        xopt_true = np.array([ -0.44198412, 0.24806027 ])
        for i, val in enumerate(xopt):
            self.assertAlmostEqual(val, xopt_true[i])
  
      
if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvolutionaryTest)
    unittest.TextTestRunner().run(suite)