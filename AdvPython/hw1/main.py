import ast
import inspect
from tree import Tree
from fib import fib
import networkx as nx
import astunparse


def detect_redundant(graph):
    REDUNDANTS = ("Load", "Module", "Store")

    candidates = []
    for node in list(graph.nodes):
        for r in REDUNDANTS:
            if r in node:
                candidates.append(node)

    return candidates


def main():
    ast_object = ast.parse(inspect.getsource(fib))

    # print(astunparse.dump(ast_object))

    tree = Tree()
    tree.visit(ast_object)

    # print(tree.graph.nodes)

    tree.graph.remove_nodes_from(detect_redundant(tree.graph))

    p = nx.drawing.nx_pydot.to_pydot(tree.graph)
    p.write_png("artifacts/ast.png")


if __name__ == "__main__":
    main()
