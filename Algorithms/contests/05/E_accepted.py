import sys


def ex_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x, y = ex_gcd(b % a, a)
    return gcd, y - (b // a) * x, x


def hack_rsa(n, e, c):
    q = 2
    while q * q <= n:
        if n % q == 0:
            p = n // q
            break
        q += 1
    b = (q - 1) * (p - 1)
    g, d, k = ex_gcd(e, b)
    d = (d % b + b) % b

    return pow(c, d, n)


def main():
    n = int(sys.stdin.readline())
    e = int(sys.stdin.readline())
    c = int(sys.stdin.readline())
    m = hack_rsa(n, e, c)
    sys.stdout.write(str(m))


if __name__ == '__main__':
    main()
