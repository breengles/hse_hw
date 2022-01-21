import ast


def main():
    with open("fib.py", "r") as src:
        cnt = src.read()
        ast_object = ast.parse(cnt)

    print(ast.dump(ast_object))


if __name__ == "__main__":
    main()
