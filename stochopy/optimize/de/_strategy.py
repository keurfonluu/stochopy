def rand1bin(i, F, X, gbest=None):
    """
    Strategy rand1bin.

    Mutate a random vector by adding one scaled difference vector.

    """
    return X[i[0]] + F * (X[i[1]] - X[i[2]])


def rand2bin(i, F, X, gbest=None):
    """
    Strategy rand2bin.

    Mutate a random vector by adding two scaled difference vectors.

    """
    return X[i[0]] + F * (X[i[1]] + X[i[2]] - X[i[3]] - X[i[4]])


def best1bin(i, F, X, gbest):
    """
    Strategy best1bin.

    Mutate the best vector by adding one scaled difference vector.

    """
    return gbest + F * (X[i[0]] - X[i[1]])


def best2bin(i, F, X, gbest):
    """
    Strategy best2bin.

    Mutate the best vector by adding two scaled difference vectors.

    """
    return gbest + F * (X[i[0]] + X[i[1]] - X[i[2]] - X[i[3]])


_strategy_map = {
    "rand1bin": rand1bin,
    "rand2bin": rand2bin,
    "best1bin": best1bin,
    "best2bin": best2bin,
}
