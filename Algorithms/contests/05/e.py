import sys, math, threading

input = sys.stdin.readline
print2 = lambda x: sys.stdout.write(str(x) + "\n")
# sys.setrecursionlimit(1 << 30)
# threading.stack_size(1 << 27)
# main_threading = threading.Thread(target=main),
# main_threading.start()
# main_threading.join()


def egcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = egcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def rsa(n, e, c):
    q = 2
    while q ** 2 <= n:
        if n % q == 0:
            p = n // q
            break
        q += 1
    b = (p - 1) * (q - 1)
    _, x, _ = egcd(e, b)
    x = (x % b + b) % b
    print2(pow(c, x, n))


def main():
    n = int(input())
    e = int(input())
    c = int(input())
    rsa(n, e, c)


main()
