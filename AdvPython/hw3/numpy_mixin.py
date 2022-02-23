import pickle
from numbers import Number

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class SaveMixin:
    def save(self, path: str) -> None:
        with open(path, "wb") as store_file:
            pickle.dump(self, store_file)


class PrettyPrintMixin:
    def __str__(self):
        """
        supporting only D2 arrays
        """
        return "\n".join(["\t".join([str(cell) for cell in row]) for row in self])


class ValueMixin:
    def __init__(self, value):
        self._value = np.asarray(value)

    @property
    def data(self):
        return self._value

    @data.setter
    def data(self, value):
        self._value = value

    @property
    def shape(self):
        return self._value.shape

    def __getitem__(self, index):
        return self._value[index]


class MatrixMixed(NDArrayOperatorsMixin, ValueMixin, PrettyPrintMixin, SaveMixin):
    _HANDLED_TYPES = (np.ndarray, Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (MatrixMixed,)):
                return NotImplemented

        inputs = tuple(x.data if isinstance(x, MatrixMixed) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.data if isinstance(x, MatrixMixed) else x for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            return tuple(type(self)(x) for x in result)

        elif method == "at":
            return None

        else:
            return type(self)(result)
