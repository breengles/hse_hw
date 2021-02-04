#!/usr/bin/env python3
import sys


def main():
    if len(sys.argv) < 2:
        for idx, line in enumerate(sys.stdin):
            print("  ", idx + 1, line, sep="  ", end="")
    else:
        cell_size_num_lines = len(
            str(sum(1 for line in open(str(sys.argv[1]), "r")))
        )
        with open(str(sys.argv[1]), "r") as fin:
            for idx, line in enumerate(fin):
                num_str = "{:{align}{width}}".format(
                    str(idx + 1), align=">", width=cell_size_num_lines
                )
                print("  ", num_str, line, sep="  ", end="")


if __name__ == "__main__":
    main()
