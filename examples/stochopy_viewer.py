# -*- coding: utf-8 -*-

"""
Run StochOPy Viewer to see how popular stochastic algorithms work, and play
with the tuning parameters on several benchmark functions.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

try:
    from stochopy.gui import main
except ImportError:
    import sys
    sys.path.append("../")
    from stochopy.gui import main


if __name__ == "__main__":
    main()