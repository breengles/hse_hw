from math import sqrt


def fib(n):
    PHI = (1 + sqrt(5)) * 0.5
    return int((PHI ** n - (1 - PHI) ** n) / (2 * PHI - 1))
