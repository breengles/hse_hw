def generate_latex_document(input_file_path="table.tex"):
    preamble = "\\documentclass{article}\n"
    begin = "\\begin{document}\n"
    end = "\\end{document}\n"

    body = "\\input{" + input_file_path + "}\n"

    return preamble + begin + body + end


def generate_latex_table(data):
    num_of_cols = len(data[0])

    begin = r"\begin{tabular}{" + "|l|" + "c|" * (num_of_cols - 2) + "r|" + "}\n"
    begin += "\\hline\n"
    end = r" \\" + "\n\\hline \n\\end{tabular}"

    body = f" \\\\ \n".join(map(lambda row: " & ".join(row), data))

    return begin + body + end


if __name__ == "__main__":
    data = [["col1", "col2", "col3"], ["1", "2", "3"], ["word1", "word2", "word3"]]

    with open("artifacts/table.tex", "w+") as tablefile:
        tablefile.write(generate_latex_table(data))

    with open("artifacts/main.tex", "w+") as texfile:
        texfile.write(generate_latex_document("table.tex"))

