import numpy


def in_search_space(x, lower, upper, constrain):
    return (
        numpy.logical_and(numpy.all(x >= lower), numpy.all(x <= upper))
        if constrain
        else True
    )
