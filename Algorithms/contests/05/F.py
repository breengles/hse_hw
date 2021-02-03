import sys

VN = int(1e5) + 2
t = [0 for _ in range(int(1e6) + 1)]
s = [0 for _ in range(30 + 1)]
root = 1
vn = 2
End = [0 for _ in range(VN)]
Next = [[0 for i in range(26)] for j in range(VN)]
n = 0
isi = [0 for _ in range(VN)]


def add(i, s):
    global vn
    v = root
    for ch in s:
        # r = Next[v][ord(ch) - ord("a")]
        if not Next[v][ord(ch) - ord("a")]:
            Next[v][ord(ch) - ord("a")] = vn
            vn += 1
        v = Next[v][ord(ch) - ord("a")]
    End[v] = i


def main():
    t = sys.stdin.readline().strip()
    n = int(sys.stdin.readline().strip())
    for i in range(len(End)):
        End[i] = n
    for i in range(n):
        s = sys.stdin.readline().strip()
        add(i, s)
    for i in range(len(t)):
        v = root
        j = i
        while v and j < len(t):
            v = Next[v][ord(t[j]) - ord("a")]
            isi[End[v]] = 1
            j += 1
    for i in range(n):
        if isi[i]:
            print("Yes")
        else:
            print("No")


if __name__ == '__main__':
    main()
