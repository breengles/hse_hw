import sys, math, threading

input = sys.stdin.readline
print = lambda x: sys.stdout.write(str(x) + '\n')
# sys.setrecursionlimit(1 << 30)
# threading.stack_size(1 << 27)
# main_threading = threading.Thread(target=main)
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
    s = input().strip() * 2
    n = len(s)
    z = z_func(s)
    num = 1
    for i in range(n // 2):
        if i + z[i] < n and s[z[i]] > s[i + z[i]]:
            num += 1
    print(num)
    
    
main()

