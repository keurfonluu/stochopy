# -*- coding: utf-8 -*-

"""
StochOPy (STOCHastic OPtimization for PYthon) provides user-friendly routines
to sample or optimize objective functions with the most popular algorithms.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from .stochopy import MonteCarlo, Evolutionary

__all__ = [ "MonteCarlo", "Evolutionary" ]
__version__ = "1.1.2"