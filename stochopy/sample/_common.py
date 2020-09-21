import numpy


def in_search_space(x, lower, upper, constraints):
    """Determine feasibility of a sample."""
    if constraints == "Reject":
        numpy.logical_and(numpy.all(x >= lower), numpy.all(x <= upper))
    else:
        return True
