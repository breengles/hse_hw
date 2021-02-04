import sys


def z_function(nums):
    num_size = len(nums)
    left = right = 0
    z_func = [0] * num_size
    z_func[0] = num_size
    for i in range(1, num_size):
        z_func[i] = max(0, min(right - i, z_func[i - left]))
        while i + z_func[i] < num_size and nums[z_func[i]] == nums[i + z_func[i]]:
            z_func[i] += 1
        if i + z_func[i] > right:
            left = i
            right = i + z_func[i]
    return z_func


def main():
    string = sys.stdin.readline().strip()
    s = string * 2
    n = len(s)
    z = z_function(s)
    num = 1
    for i in range(int(n / 2)):
        if i + z[i] < n and s[z[i]] > s[i + z[i]]:
            num += 1
    sys.stdout.write(str(num))


if __name__ == '__main__':
    main()
