import numpy


def delete_shuffle_sync(popsize):
    return numpy.transpose([
        delete_shuffle_async(i, popsize) for i in range(popsize)
    ])


def delete_shuffle_async(i, popsize):
    return numpy.random.permutation(numpy.delete(numpy.arange(popsize), i))


def rand1(i, F, X, gbest=None):
    return X[i[0]] + F * (X[i[1]] - X[i[2]])


def rand2(i, F, X, gbest=None):
    return X[i[0]] + F * (X[i[1]] + X[i[2]] - X[i[3]] - X[i[4]])


def best1(i, F, X, gbest):
    return gbest + F * (X[i[0]] - X[i[1]])


def best2(i, F, X, gbest):
    return gbest + F * (X[i[0]] + X[i[1]] - X[i[2]] - X[i[3]])


strategies = {
    "rand1": rand1,
    "rand2": rand2,
    "best1": best1,
    "best2": best2,
}
