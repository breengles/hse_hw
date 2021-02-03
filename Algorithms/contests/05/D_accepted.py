import sys


def get_inverse(a, m):
    m0 = m
    y = 0
    x = 1
    if m == 1:
        return 0

    while a > 1:
        try:
            q = a // m
        except ZeroDivisionError:
            sys.stdout.write("-1")
            exit()
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
    if x < 0:
        x = x + m0

    return x


def main():
    n, m = map(int, sys.stdin.readline().split())
    sys.stdout.write(str(get_inverse(n, m)))


if __name__ == '__main__':
    main()
