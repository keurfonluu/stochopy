from .._common import BaseResult


optimizer_map = {}


class OptimizeResult(BaseResult):
    pass


def register(name, minimize):
    optimizer_map[name] = minimize
