import numpy


def NoConstraint(lower, upper):
    """No constraint."""

    def cons(U):
        return U

    return cons


def Random(lower, upper):
    """
    Random constraint.
    
    Solutions that are in the infeasible space are resampled.

    """
    
    def cons(U):
        return numpy.where(
            numpy.logical_or(U < lower, U > upper),
            numpy.random.uniform(lower, upper, U.shape),
            U,
        )

    return cons


_constraints_map = {
    None: NoConstraint,
    "Random": Random,
}
