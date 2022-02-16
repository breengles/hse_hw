from ast_parser.tree import generate_and_save_ast
from fib import fib
import latex_utils as tex


def main():
    # generating ast for fib func from my pkg
    ast_fig_path = "artifacts/ast.png"
    generate_and_save_ast(fib, ast_fig_path)

    data = [["col1", "col2", "col3"], ["1", "2", "3"], ["word1", "word2", "word3"]]

    with open("artifacts/table.tex", "w+") as textable:
        textable.write(tex.generate_latex_table(data))

    with open("artifacts/astfig.tex", "w+") as texfig:
        texfig.write(tex.generate_latex_figure("ast.png"))

    with open("artifacts/main.tex", "w+") as texmain:
        texmain.write(tex.generate_latex_document(("table.tex", "astfig.tex")))


if __name__ == "__main__":
    main()
