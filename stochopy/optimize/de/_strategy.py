def rand1bin(i, F, X, gbest=None):
    return X[i[0]] + F * (X[i[1]] - X[i[2]])


def rand2bin(i, F, X, gbest=None):
    return X[i[0]] + F * (X[i[1]] + X[i[2]] - X[i[3]] - X[i[4]])


def best1bin(i, F, X, gbest):
    return gbest + F * (X[i[0]] - X[i[1]])


def best2bin(i, F, X, gbest):
    return gbest + F * (X[i[0]] + X[i[1]] - X[i[2]] - X[i[3]])


_strategy_map = {
    "rand1bin": rand1bin,
    "rand2bin": rand2bin,
    "best1bin": best1bin,
    "best2bin": best2bin,
}
