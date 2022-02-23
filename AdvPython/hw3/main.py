import numpy as np
from matrix import Matrix
from numpy_mixin import MatrixMixed


def check_correctness(first, second):
    assert first.shape == second.shape, f"{first.shape}, {second.shape}"

    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            assert first[i][j] == second[i][j]

    return True


def task(a, b, A, B, part="easy"):
    add_res = A + B
    mul_res = A * B
    matmul_res = A @ B

    if check_correctness(a + b, add_res):
        print("+ passed!")
    if check_correctness(a * b, mul_res):
        print("* passed!")
    if check_correctness(a @ b, matmul_res):
        print("@ passed!")

    with open(f"artifacts/{part}/matrix+.txt", "w+") as art:
        art.write(str(add_res))

    with open(f"artifacts/{part}/matrix*.txt", "w+") as art:
        art.write(str(mul_res))

    with open(f"artifacts/{part}/matrix@.txt", "w+") as art:
        art.write(str(matmul_res))


if __name__ == "__main__":
    np.random.seed(0)

    a_numpy = np.random.randint(0, 10, (10, 10))
    b_numpy = np.random.randint(0, 10, (10, 10))

    a_matrix = Matrix(a_numpy.tolist())
    b_matrix = Matrix(b_numpy.tolist())

    a_mixin = MatrixMixed(a_numpy.tolist())
    b_mixin = MatrixMixed(b_numpy.tolist())

    task(a_numpy, b_numpy, a_matrix, b_matrix, part="easy")
    task(a_numpy, b_numpy, a_mixin, b_mixin, part="medium")
