import sys, math, threading

input = sys.stdin.readline
print2 = lambda x: sys.stdout.write(str(x) + " ")
# sys.setrecursionlimit(1 << 30)
# threading.stack_size(1 << 27)
# main_threading = threading.Thread(target=main),
# main_threading.start()
# main_threading.join()


def z_func(s):
    n = len(s)
    l = r = 0
    z = [0] * n
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1
    return z


def main():
    n, m = map(int, input().split())
    kubiki = [int(i) for i in input().split()]
    n1 = 2 * n + 1
    s = [-1] * n1
    for i in range(n):
        s[i] = s[2 * n - i] = kubiki[i]

    z = z_func(s)

    for i in range(n + 1, n1):
        if z[i] == n1 - i and z[i] % 2 == 0:
            print2(n - z[i] // 2)
            
    print2(n)
            
main()
