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
    size, color = map(int, sys.stdin.readline().split())
    nums = [-1] * (2 * size + 1)
    cubes = [int(i) for i in sys.stdin.readline().split()]
    for i in range(size):
        nums[i] = cubes[i]
        nums[2 * size - i] = nums[i]

    z_func = z_function(nums)
    for i in range(size + 1, len(nums)):
        if z_func[i] == len(nums) - i and z_func[i] % 2 == 0:
            sys.stdout.write(f"{size - int(z_func[i] / 2)} ")
    sys.stdout.write(f"{size}")


if __name__ == '__main__':
    main()
