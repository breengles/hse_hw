#!/usr/bin/env python3
import functools
import time


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
def logger(func, file=None):
    log = open(file, "a")
    @functools.wraps(func)
    def inner(*args, **kwargs):
        # with open(file, "a") as log:
        nonlocal log
        args_str = " ".join(["%d"] * len(args)) % tuple(args)
        kwargs_str = (
            str(kwargs).replace(":", "=").replace("{", "").replace("}", "")
        )
        res = func(*args, **kwargs)
        datetime = time.strftime("%b %d %Y %H:%M:%S")
        log.write(
            f"{datetime} {func.__name__} ({args_str}) "
            + f"({kwargs_str}) {res}\n"
        )
        return res

    return inner


@logger("my_log.txt")
def f(n, k=None, x=None, y=None):
    if n >= 0:
        ans = f(n - 1, k=k, x=x, y=y) + n
        time.sleep(1)  # some heavy computation
        return ans
    else:
        time.sleep(1)
        return 0


f(1)
f(5, k=1, x=1, y=1)
f(5, k=2, x=2, y=2)
