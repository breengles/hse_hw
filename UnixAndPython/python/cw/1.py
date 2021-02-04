#!/usr/bin/env python3

import functools
import time


def spy(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        args_str = str(args)
        kwargs_str = str(kwargs)
        inner.info.append((time.strftime("%H:%M:%S"), f"{args_str} {kwargs_str}"))
        return func(*args, **kwargs)

    inner.info = []
    inner.decorated_by_spy = True
    return inner


def bond(func):
    if hasattr(func, "decorated_by_spy"):
        return func.info
    else:
        raise TypeError(f"Function {func.__name__} is not decorated by spy")


@spy
def foo(n):
    return n


foo(30)
foo("hello")
foo(5)


for (time, parameters) in bond(foo):
    print(time)
    print(parameters)
