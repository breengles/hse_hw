#!/usr/bin/env python3

import functools
from time import time, sleep


def profiler(func):
    initial = True

    @functools.wraps(func)
    def inner(*args, **kwargs):
        nonlocal initial
        if initial:
            initial = False
            inner.calls = 0
            inner.last_time_taken = 0

            start = time()
            res = func(*args, **kwargs)
            end = time()
            inner.last_time_taken = end - start
            return res
        else:
            inner.calls += 1
            res = func(*args, **kwargs)
            initial = True  # crucial place
            return res

    initial = True
    return inner


@profiler
def test_func(n):
    if n != 0:
        sleep(0.1)
        return test_func(n - 1)
    else:
        return 0


test_func(4)
test_func(6)
test_func(8)
print(test_func.calls)
print(test_func.last_time_taken)
