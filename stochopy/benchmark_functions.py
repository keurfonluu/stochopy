# -*- coding: utf-8 -*-

"""
This package contains several classical benchmark functions used to test
global optimization algorithms performance.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [ "BenchmarkFunction" ]


class BenchmarkFunction:
    """
    Benchmark functions.
    
    This class provides several benchmark functions to test global optimization
    algorithms.
    
    Parameters
    ----------
    func: {'ackley', 'griewank', 'quartic', 'quartic_noise', 'rastrigin',
           'rosenbrock', 'sphere', 'styblinski-tang'}
        Benchmark function name.
    n_dim: int, default 2
        Number of dimensions.
    """
    
    def __init__(self, func, n_dim = 2):
        self._n_dim = n_dim
        if func.lower() == "ackley":
            self._func = self._ackley
            self._lower = np.full(n_dim, -32.768)
            self._upper = np.full(n_dim, 32.768)
            self._min = 0.
        elif func.lower() == "griewank":
            self._func = self._griewank
            self._lower = np.full(n_dim, -600.)
            self._upper = np.full(n_dim, 600.)
            self._min = 0.
        elif func.lower() == "quartic":
            self._func = self._quartic
            self._lower = np.full(n_dim, -1.28)
            self._upper = np.full(n_dim, 1.28)
            self._min = 0.
        elif func.lower() == "quartic_noise":
            self._func = self._quartic
            self._lower = np.full(n_dim, -1.28)
            self._upper = np.full(n_dim, 1.28)
            self._min = 0.
        elif func.lower() == "rastrigin":
            self._func = self._rastrigin
            self._lower = np.full(n_dim, -5.12)
            self._upper = np.full(n_dim, 5.12)
            self._min = 0.
        elif func.lower() == "rosenbrock":
            self._func = self._rosenbrock
            self._lower = np.full(n_dim, -5.12)
            self._upper = np.full(n_dim, 5.12)
            self._min = 0.
        elif func.lower() == "sphere":
            self._func = self._sphere
            self._lower = np.full(n_dim, -5.12)
            self._upper = np.full(n_dim, 5.12)
            self._min = 0.
        elif func.lower() == "styblinski-tang":
            self._func = self._styblinski_tang
            self._lower = np.full(n_dim, -5.12)
            self._upper = np.full(n_dim, 5.12)
            self._min = 0.
        else:
            raise ValueError("unknown benchmark function '%s'" % func)
            
    def get(self):
        """
        Get the benchmark function.
        
        Returns
        -------
        dict: dictionary
            Dictionary containing the function, the lower and upper boundaries.
        """
        return dict(func = self._func, lower = self._lower, upper = self._upper)

    def _ackley(self, x):
        n_dim = len(x)
        e = 2.7182818284590451
        sum1 = np.sqrt( 1.0 / n_dim * np.sum( np.array(x)**2 ) )
        sum2 = 1.0 / n_dim * np.sum( np.cos( 2.0 * np.pi * np.array(x) ) )
        return 20.0 + e - 20.0 * np.exp( -0.2 * sum1 ) - np.exp(sum2)
        
    def _griewank(self, x):
        n_dim = len(x)
        sum1 = np.sum( np.array(x)**2 ) / 4000.0
        prod1 = np.prod( np.cos( np.array(x) / np.sqrt( np.arange(1, n_dim+1) ) ) )
        return 1.0 + sum1 - prod1
        
    def _quartic(self, x):
        n_dim = len(x)
        return np.sum( np.arange(1, n_dim+1) * np.array(x)**4 )
        
    def _quartic_noise(self, x):
        return self._quartic(x) + np.random.rand()
        
    def _rastrigin(self, x):
        n_dim = len(x)
        sum1 = np.sum( np.array(x)**2 - 10.0 * np.cos( 2.0 * np.pi * np.array(x) ) )
        return 10.0 * n_dim + sum1
        
    def _rosenbrock(self, x):
        sum1 = np.sum( ( np.array(x[1:]) - np.array(x[:-1])**2 )**2 )
        sum2 = np.sum( ( 1.0 - np.array(x[:-1]) )**2 )
        return 100.0 * sum1 + sum2
        
    def _sphere(self, x):
        return np.sum( np.array(x)**2 )
        
    def _styblinski_tang(self, x):
        sum1 = np.sum( np.array(x)**4 - 16.0 * np.array(x)**2 + 5.0 * np.array(x) )
        return sum1 / 2.0 + 39.16599 * len(x)
    
    def plot(self, nx = 101, ny = 101, n_levels = 10, axes = None,
             figsize = (8, 8), kwargs = {}):
        """
        Plot the benchmark function in 2-D.
        
        Parameters
        ----------
        nx: int, default 101
            Number of samples in the first dimension.
        ny: int, default 101
            Number of samples in the second dimension.
        n_levels: int, default 10
            Number of levels for contour.
        axes: matplotlib axes or None, default None
            Axes used for plot.
        figsize: tuple, default (8, 8)
            Figure width and height if axes is None.
            
        Returns
        -------
        ax1: matplotlib axes
            Axes used for plot.
        """
        if axes is None:
            fig = plt.figure(figsize = figsize, facecolor = "white")
            fig.patch.set_alpha(0.)
            ax1 = fig.add_subplot(1, 1, 1)
        else:
            ax1 = axes
        ax = np.linspace(self._lower[0], self._upper[0], nx)
        ay = np.linspace(self._lower[1], self._upper[1], ny)
        X, Y = np.meshgrid(ax, ay)
        funcgrid = np.array([ self._func([x, y]) for x, y
                                in zip(X.ravel(), Y.ravel()) ]).reshape((nx, ny))
        ax1.contourf(ax, ay, funcgrid, 100, **kwargs)
        ax1.contour(ax, ay, funcgrid, n_levels, colors = "black", alpha = 0.3)
        ax1.grid(True)
        plt.show()
        return ax1