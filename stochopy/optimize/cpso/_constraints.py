import numpy


def NoConstraint(lower, upper, sync):
    """No constraint."""
    
    def cons(X, V):
        return X + V, V

    return cons


def Shrink(lower, upper, sync):
    """
    Shrinking approach.

    Velocity vector amplitude is shrinked for solutions that are in the infeasible space.
    This approach preserves the trajectory of the particles.

    """

    def shrink(X, V, maskl, masku):
        condl = maskl.any()
        condu = masku.any()

        if condl and condu:
            bl = (lower[maskl] - X[maskl]) / V[maskl]
            bu = (upper[masku] - X[masku]) / V[masku]
            return min(bl.min(), bu.min())

        elif condl and not condu:
            bl = (lower[maskl] - X[maskl]) / V[maskl]
            return bl.min()

        elif not condl and condu:
            bu = (upper[masku] - X[masku]) / V[masku]
            return bu.min()

        else:
            return 1.0

    if sync:

        def cons(X, V):
            Xcand = X + V
            maskl = Xcand < lower
            masku = Xcand > upper
            beta = numpy.array(
                [shrink(x, v, ml, mu) for x, v, ml, mu in zip(X, V, maskl, masku)]
            )
            V *= beta[:, None]

            return X + V, V

    else:

        def cons(X, V):
            Xcand = X + V
            maskl = Xcand < lower
            masku = Xcand > upper
            beta = shrink(X, V, maskl, masku)
            V *= beta

            return X + V, V

    return cons


_constraints_map = {
    None: NoConstraint,
    "Shrink": Shrink,
}
