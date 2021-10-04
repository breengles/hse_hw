#!/usr/bin/env python


from argparse import ArgumentParser
from datetime import datetime

import numpy as np
from utils import read_data, save_data
from scipy.stats import rankdata


def main(input_file_path: str, output_file_path: str):
    data = read_data(input_file_path)

    n = data.shape[0]
    p = int(round(n / 3))

    rs = rankdata(data[:, 1])
    rs = -(rs - rs.max() - 1)

    r1 = rs[:p].sum()
    r2 = rs[-p:].sum()

    ranks_diff = r1 - r2

    std_error = (n + 1 / 2) * np.sqrt(p / 6)
    conjugate_degree = ranks_diff / p / (n - p)

    # rounding
    ranks_diff = int(round(ranks_diff))
    std_error = int(round(std_error))
    conjugate_degree = round(conjugate_degree, 2)

    print(f"{ranks_diff} {std_error} {conjugate_degree}")
    save_data(output_file_path, ranks_diff, std_error, conjugate_degree)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("input", type=str, help="Path to input file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output file. If not specified execution timestamp: `out.TIMESTAMP.txt`. The output is duplicated in the stdout",
    )

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now()
        output_file_path = f"out.{timestamp.date()}_{timestamp.time()}.txt"
    else:
        output_file_path = args.output.strip()

    main(args.input, output_file_path)
