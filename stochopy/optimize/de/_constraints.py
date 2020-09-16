import numpy


def NoConstraint(lower, upper):
    def cons(U):
        return U

    return cons


def Random(lower, upper):
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
