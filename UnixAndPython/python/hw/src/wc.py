#!/usr/bin/env python3
import re
import sys
from os.path import isfile


# ну тут больше кода, чтобы подогнать под формать wc
def main():
    num_lines_str = "0"
    num_words_str = "0"
    num_bytes_str = "0"
    if len(sys.argv) < 2:
        num_lines = 0
        num_words = 0
        num_bytes = 0
        for line in sys.stdin:
            num_lines += len(re.findall(r"\n", line))
            num_bytes += len(line.encode("utf-8"))
            num_words += len(line.split())

            num_lines_str = "{:{align}{width}}".format(
                str(num_lines), align=">", width=max(7, len(str(num_lines)))
            )
            num_words_str = "{:{align}{width}}".format(
                str(num_words), align=">", width=max(7, len(str(num_words)))
            )
            num_bytes_str = "{:{align}{width}}".format(
                str(num_bytes), align=">", width=max(7, len(str(num_bytes)))
            )
        print(num_lines_str, num_words_str, num_bytes_str)
    else:
        file_flg = False
        stat = []
        cell_line_size = 0
        cell_word_size = 0
        cell_byte_size = 0
        for i in range(1, len(sys.argv)):
            if isfile(str(sys.argv[i])):
                file_flg = True
                num_lines = 0
                cell_line_size = max(cell_line_size, len(str(num_lines)))
                with open(str(sys.argv[i]), "r") as fin:
                    num_words = 0
                    for line in fin:
                        num_lines += len(re.findall(r"\n", line))
                        num_words += len(line.split())
                    cell_word_size = max(cell_word_size, len(str(num_words)))
                with open(str(sys.argv[i]), "rb") as fin:
                    num_bytes = 0
                    while fin.read(1):
                        num_bytes += 1
                    cell_byte_size = max(cell_byte_size, len(str(num_bytes)))
                stat.append(
                    (num_lines, num_words, num_bytes, str(sys.argv[i]))
                )

        if file_flg:
            if len(stat) > 1:
                lines_total = 0
                words_total = 0
                bytes_total = 0
                for item in stat:
                    lines_total += item[0]
                    words_total += item[1]
                    bytes_total += item[2]

                cell_line_size = max(cell_line_size, len(str(lines_total)))
                cell_word_size = max(cell_word_size, len(str(words_total)))
                cell_byte_size = max(cell_byte_size, len(str(bytes_total)))
                stat.append((lines_total, words_total, bytes_total, "total"))

            for item in stat:
                num_lines_str = "{:{align}{width}}".format(
                    str(item[0]), align=">", width=max(4, cell_line_size)
                )
                num_words_str = "{:{align}{width}}".format(
                    str(item[1]), align=">", width=max(4, cell_word_size)
                )
                num_bytes_str = "{:{align}{width}}".format(
                    str(item[2]), align=">", width=max(4, cell_byte_size)
                )
                name = "{:{align}}".format(str(item[3]), align="<")
                print(num_lines_str, num_words_str, num_bytes_str, name)
        else:
            num_lines = 0
            num_words = 0
            num_bytes = 1

            for i in range(1, len(sys.argv)):
                num_lines += len(re.findall(r"\n", str(sys.argv[i])))
                num_words += len(str(sys.argv[i]).split())
                num_bytes += len(str(sys.argv[i]).encode("utf-8"))

            num_lines_str = "{:{align}{width}}".format(
                str(num_lines), align=">", width=max(7, len(str(num_lines)))
            )
            num_words_str = "{:{align}{width}}".format(
                str(num_words), align=">", width=max(7, len(str(num_words)))
            )
            num_bytes_str = "{:{align}{width}}".format(
                str(num_bytes), align=">", width=max(7, len(str(num_bytes)))
            )
            print(num_lines_str, num_words_str, num_bytes_str)


if __name__ == "__main__":
    main()
