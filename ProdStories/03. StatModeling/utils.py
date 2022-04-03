import numpy as np


def read_data(file_path: str) -> np.ndarray:
    xs, ys = [], []
    with open(file_path, "r") as input_file:
        """
        might be optimized if we can optimally count number of lines
        and create `out` with size of (number_of_lines, number_of_lines)
        """
        for line in input_file:
            x, y = map(float, line.split())
            xs.append(x)
            ys.append(y)

    xs, ys = map(np.array, (xs, ys))

    sorted_by_x_idx = np.argsort(xs)

    xs = xs[sorted_by_x_idx].reshape(-1, 1)
    ys = ys[sorted_by_x_idx].reshape(-1, 1)

    return np.hstack([xs, ys])


def save_data(file_path: str, difference: int, std_error: int, conjugate_degree: int):
    with open(file_path, "w+") as output_file:
        output_file.write(f"{difference} {std_error} {conjugate_degree}")
