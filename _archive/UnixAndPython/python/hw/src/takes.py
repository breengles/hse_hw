#!/usr/bin/env python3
import functools


def with_arguments(deco):
    @functools.wraps(deco)
    def wrapper(*dargs, **dkwargs):
        def decorator(func):
            result = deco(func, *dargs, **dkwargs)
            functools.update_wrapper(result, func)
            return result

        return decorator

    return wrapper


@with_arguments
def takes(func, *types):
    @functools.wraps(func)
    def inner(*args):
        for i in range(min(len(types), len(args))):
            if not isinstance(args[i], types[i]):
                raise TypeError(
                    f"{args[i]} is not of type {types[i].__name__}"
                )
        return func(*args)

    return inner


@takes(int, str)
def f(a, b):
    pass


f(1, "ab")
f(1, 2)  # ошибка
