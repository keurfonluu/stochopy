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
    from monte_carlo import MonteCarlo
else:
    from stochopy import MonteCarlo


class MonteCarloTest(unittest.TestCase):
    """
    Monte-Carlo algorithms unit tests.
    """
    
    def setUp(self):
        """
        Initialize Monte-Carlo algorithm tests.
        """
        func = lambda x: 100*np.sum((x[1:]-x[:-1]**2)**2)+np.sum((1-x[:-1])**2)
        n_dim = 2
        lower = np.full(n_dim, -5.12)
        upper = np.full(n_dim, 5.12)
        max_iter = 50
        random_state = 42
        self.sampler = MonteCarlo(func, lower = lower, upper = upper,
                                  max_iter = max_iter, random_state = random_state)
        self.n_dim = n_dim
        
    def tearDown(self):
        """
        Cleaning after each test.
        """
        del self.sampler
        
    def test_pure(self):
        """
        Pure Monte-Carlo test.
        """
        self.sampler.sample(sampler = "pure")
        mean = np.mean(self.sampler.models, axis = 0)
        mean_true = np.array([ -0.6070602, -0.00363818 ])
        for i, val in enumerate(mean):
            self.assertAlmostEqual(val, mean_true[i])
            
    def test_hastings(self):
        """
        Metropolis-Hastings algorithm test.
        """
        stepsize = 0.1409
        self.sampler.sample(sampler = "hastings", stepsize = stepsize)
        mean = np.mean(self.sampler.models, axis = 0)
        mean_true = np.array([ -1.96808276, 4.26872887 ])
        for i, val in enumerate(mean):
            self.assertAlmostEqual(val, mean_true[i])
            
    def test_hamiltonian(self):
        """
        Hamiltonian Monte-Carlo algorithm test.
        """
        stepsize = 0.0091991
        n_leap = 14
        self.sampler.sample(sampler = "hamiltonian", stepsize = stepsize,
                            n_leap = n_leap)
        mean = np.mean(self.sampler.models, axis = 0)
        mean_true = np.array([ 0.89343405, 1.18474131 ])
        for i, val in enumerate(mean):
            self.assertAlmostEqual(val, mean_true[i])
  
      
if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MonteCarloTest)
    unittest.TextTestRunner().run(suite)