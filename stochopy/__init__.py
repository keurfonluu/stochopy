# -*- coding: utf-8 -*-

"""
StochOPy (STOCHastic OPtimization for PYthon) provides user-friendly routines
to sample or optimize objective functions with the most popular algorithms.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from .monte_carlo import MonteCarlo
from .evolutionary_algorithm import Evolutionary
from .benchmark_functions import BenchmarkFunction
from .gui import StochOGUI

__all__ = [ "MonteCarlo", "Evolutionary", "BenchmarkFunction", "StochOGUI" ]
__version__ = "1.5.2"