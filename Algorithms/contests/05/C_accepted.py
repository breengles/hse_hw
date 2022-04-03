import sys


def pref_func(s):
    p = [0] * (len(s))
    for i in range(1, len(s)):
        k = p[i - 1]
        while k > 0 and s[i] != s[k]:
            k = p[k - 1]
        if s[i] == s[k]:
            k += 1
        p[i] = k
    return p


def main():
    string = sys.stdin.readline().strip()
    pref = pref_func(string)
    sys.stdout.write(f"{len(string) - pref[-1]}")


if __name__ == '__main__':
    main()
