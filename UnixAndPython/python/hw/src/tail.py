#!/usr/bin/env python3
import sys


def main():
    if len(sys.argv) < 2:
        q = []
        for idx, line in enumerate(sys.stdin):
            if len(q) < 10:
                q.append(line)
            else:
                q.pop(0)
                q.append(line)
        for line in q:
            print(line, end="")
    else:
        for i in range(1, len(sys.argv)):
            if len(sys.argv) > 2:
                print(f"==> {str(sys.argv[i])} <==")
            num_lines = sum(1 for line in open(str(sys.argv[i]), "r")) - 9
            with open(str(sys.argv[i]), "r") as fin:
                for idx, line in enumerate(fin):
                    if idx + 1 >= num_lines:
                        print(line, end="")
                if i != len(sys.argv) - 1:
                    print()


if __name__ == "__main__":
    main()
