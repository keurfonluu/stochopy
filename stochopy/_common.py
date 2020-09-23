class BaseResult(dict):
    """
    Represent the optimization or sampling result.

    Base class. Do not use.

    """

    def __getattr__(self, name):
        """Define dict.attr as an alias of dict[attr]."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Pretty result."""
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [
                    k.rjust(m) + ": " + repr(v)
                    for k, v in sorted(self.items())
                    if k not in {"xall", "funall"}
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        """Return a list of attributes."""
        return list(self.keys())
