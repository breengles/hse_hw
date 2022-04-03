import sys, math, threading

input = sys.stdin.readline
print2 = lambda x: sys.stdout.write(str(x) + "\n")

# sys.setrecursionlimit(1 << 30)
# threading.stack_size(1 << 27)

def egcd(a, b):
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = egcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def bezout(n, m):
    g, x, _ = egcd(n, m)
    if g != 1:
        print2(-1)
        exit()
    print2((x % m + m) % m)


def test(n, m):
    k = 0
    while (1 + k * m) % n != 0:
        if (1 + k * m) / n > m:
            print2(-1)
            exit()
        k += 1
    print2((1 + k * m) // n)


def main():
    n, m = map(int, input().split())
    # test(n, m)
    bezout(n, m)
    
main()
# main_threading = threading.Thread(target=main)
# main_threading.start()
# main_threading.join()