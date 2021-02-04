#!/usr/bin/env python3


from typing import List


def vec_product(vec1: List[int], vec2: List[int]) -> int:
    """
    >>> vec_product([1, 2, 3], [4, 5, 6])
    32
    """
    return sum([i * j for i, j in zip(vec1, vec2)])


# print(vec_product([1, 2, 3], [4, 5, 6]))


def matrix_transpose(mat: List[List]) -> List[List]:
    """
    >>> matrix_transpose([[1, 2], [3, 4], [5, 6]])
    [[1, 3, 5], [2, 4, 6]]
    """
    return list(map(list, zip(*mat)))


# print(matrix_transpose([[1, 2], [3, 4], [5, 6]]))


def matrix_product(mat1: List[List[int]], mat2: List[List[int]]):
    """
    >>> mat1 = [[1, 3, 2], [0, 4, -1]]
    >>> mat2 = [[2, 0, -1, 111], [3, -2, 1, 2], [0, 1, 2, 3]]
    >>> matrix_product(mat1, mat2)
    [[11, -4, 6, 123], [12, -9, 2, 5]]
    """
    for row in mat1:
        if len(row) != len(mat2):
            raise ValueError()
    return [
        [sum(i * j for i, j in zip(row, col)) for col in list(zip(*mat2))]
        for row in mat1
    ]


# mat1 = [[1, 3, 2], [0, 4, -1]]
# mat2 = [[2, 0, -1, 111], [3, -2, 1, 2], [0, 1, 2, 3]]
# print(matrix_product(mat1, mat2))


def matrix_pretty_print(mat: List[List[int]]):
    cell_size = 0
    for row in mat:
        for a in row:
            cell_size = max(cell_size, len(str(a)))
    cell_size += 3  # for both-side whitespaces and |

    sep = "----" * (cell_size) + "-"
    print(sep, sep="")
    for row in mat:
        print("|", end="")
        for a in row:
            cell = "{:{align}{width}}|".format(
                str(a), align="^", width=cell_size - 1
            )
            print(cell, end="")
        print("\n", sep, sep="", end="\n")


# matrix_pretty_print([[1, 2, 3, 4], [5, 6, 7, 8]])
# matrix_pretty_print([[11, 22, 33, 44], [55, 66, 77, 88]])
# matrix_pretty_print([[111, 222, 333, 444], [555, 666, 777, 888]])

# matrix_pretty_print([[111, 222, 333, 444], [5, 66, 777, 888]])
# matrix_pretty_print([[1, 22, 333, 444], [555, 666, 777, 888]])
# matrix_pretty_print([[111, 2, 333, 44], [555, 666, 777, 888]])

if __name__ == '__main__':
    import doctest

    doctest.testmod()
