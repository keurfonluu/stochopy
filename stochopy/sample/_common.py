import numpy as np


def in_search_space(x, lower, upper, constraints):
    """Determine feasibility of a sample."""
    if constraints == "Reject":
        np.logical_and(np.all(x >= lower), np.all(x <= upper))
    else:
        return True
