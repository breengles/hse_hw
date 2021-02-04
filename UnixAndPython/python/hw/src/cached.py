#!/usr/bin/env python3
import functools
import time
from collections import OrderedDict


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
def cached(func, cache_size=None):
    if cache_size > 0:
        cache = OrderedDict()
    else:
        cache = None

    @functools.wraps(func)
    def inner(*args):
        nonlocal cache
        targs = tuple(args)
        if targs in cache.keys():
            return cache[targs]
        else:
            res = func(*args)
            if cache is not None:
                if len(cache) >= cache_size:
                    cache.popitem()
                cache[targs] = res
            return res

    return inner


@cached(cache_size=1)
def foo(a):
    time.sleep(2)
    return a + 1


# t1 = time.time()
foo(1)  # считает значение
# print(time.time() - t1)

# t1 = time.time()
foo(1)  # использует закешированное значение
# print(time.time() - t1)

# t1 = time.time()
foo(2)
# print(time.time() - t1)

# t1 = time.time()
foo(1)  # считает значение
# print(time.time() - t1)
