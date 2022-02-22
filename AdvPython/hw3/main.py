from typing import List
import numpy as np


class Matrix:
    def __init__(self, data: List) -> None:
        self.data = data

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __getitem__(self, index):
        return self.data[index]

    def is_elementwise_compatible(self, other):
        assert (
            self.shape == other.shape
        ), f"operands could not be broadcast together with shapes {self.shape} {other.shape}"

    def is_matrixwise_compatible(self, other):
        assert (
            self.shape[1] == other.shape[0]
        ), f"operands could not be broadcast together with shapes {self.shape} {other.shape}"

    def __add__(self, other):
        self.is_elementwise_compatible(other)
        rows_len, cols_len = self.shape

        result = []

        for i in range(rows_len):
            tmp = []
            for j in range(cols_len):
                tmp.append(self[i][j] + other[i][j])
            result.append(tmp)

        return Matrix(result)

    def __mul__(self, other):
        self.is_elementwise_compatible(other)
        rows_len, cols_len = self.shape

        result = []

        for i in range(rows_len):
            tmp = []
            for j in range(cols_len):
                tmp.append(self[i][j] * other[i][j])
            result.append(tmp)

        return Matrix(result)

    @staticmethod
    def _multiply_row_by_col(row, col):
        result = 0
        for i in range(len(row)):
            result += row[i] * col[i]

        return result

    def __matmul__(self, other):
        self.is_matrixwise_compatible(other)

        rows_len, cols_len = self.shape

        result = []
        for i in range(rows_len):
            tmp = []
            for j in range(cols_len):
                tmp.append(self._multiply_row_by_col(self[i], [row[j] for row in other]))
            result.append(tmp)

        return Matrix(result)

    def __str__(self):
        return "\n".join(["\t".join([str(cell) for cell in row]) for row in self])


def check_correctness(first, second):
    assert first.shape == second.shape

    for i in range(first.shape[0]):
        for j in range(second.shape[0]):
            assert first[i][j] == second[i][j]

    return True


if __name__ == "__main__":
    np.random.seed(0)

    a = np.random.randint(0, 10, (3, 3))
    b = np.random.randint(0, 10, (3, 3))

    A = Matrix(a.tolist())
    B = Matrix(b.tolist())

    add_res = A + B
    mul_res = A * B
    matmul_res = A @ B

    # if check_correctness(a + b, add_res):
    #     print("+ passed!")
    # if check_correctness(a * b, mul_res):
    #     print("* passed!")
    # if check_correctness(a @ b, matmul_res):
    #     print("@ passed!")

    with open("artifacts/matrix+.txt", "w+") as art:
        art.write(str(add_res))

    with open("artifacts/matrix*.txt", "w+") as art:
        art.write(str(mul_res))

    with open("artifacts/matrix@.txt", "w+") as art:
        art.write(str(matmul_res))
