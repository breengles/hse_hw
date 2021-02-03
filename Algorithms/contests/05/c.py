import sys, math, threading

input = sys.stdin.readline
print2 = lambda x: sys.stdout.write(f"{x} ")
# sys.setrecursionlimit(1 << 30)
# threading.stack_size(1 << 27)
# main_threading = threading.Thread(target=main),
# main_threading.start()
# main_threading.join()

def prefix_func(s):
    m = len(s)
    pi = [0] * m
    k = 0
    for q in range(2, m + 1):
        while k > 0 and s[k] != s[q - 1]:
            k = pi[k - 1]
        if s[k] == s[q - 1]:
            k += 1
        pi[q - 1] = k
    return pi



def main():
    s = input().strip()
    print2(len(s) - prefix_func(s)[-1])


main()