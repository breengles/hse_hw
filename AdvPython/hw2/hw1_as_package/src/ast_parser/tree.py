import ast
import networkx as nx
import inspect


class Tree(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()

        self.stack = None
        self.graph = nx.DiGraph()
        self.level = 0

    @staticmethod
    def _get_node_properties(node):
        label = str(type(node)).split(".")[-1][:-2]
        color = "gray"
        shape = "rectangle"
        style = "filled"

        if isinstance(node, ast.FunctionDef):
            label = f"{label}: {node.name}"
            color = "#16a5f2"
            shape = "trapezium"

        if isinstance(node, ast.Assign):
            color = "#f29316"

        if isinstance(node, ast.Return):
            color = "#f21616"
            shape = "invtrapezium"

        if isinstance(node, ast.Sub):
            color = "#1651f2"
            shape = "rectangle"

        if isinstance(node, ast.Pow):
            color = "#ca16f2"
            shape = "egg"

        if isinstance(node, ast.Div):
            color = "#16e0f2"
            shape = "triangle"

        if isinstance(node, ast.Add):
            color = "#04b527"
            shape = "circle"

        if isinstance(node, ast.Mult):
            color = "#147823"
            shape = "circle"

        if isinstance(node, ast.BinOp):
            label = f"{label}: {str(node.op).split()[0][5:]}"
            color = "#b5a604"

        if isinstance(node, ast.Constant):
            label = f"{label}: {node.value}"
            color = "#575757"

        if isinstance(node, ast.Name):
            color = "#8a8a8a"

        if isinstance(node, ast.List):
            color = "#548199"

        return {"label": label, "color": color, "shape": shape, "style": style}

    def generic_visit(self, node):
        if self.stack is None:
            self.stack = []
            parent_name = None
        else:
            parent_name = self.stack[-1]

        self.stack.append(str(node))

        self.graph.add_node(str(node), **self._get_node_properties(node))

        if parent_name is not None:
            self.graph.add_edge(parent_name, str(node))

        super().generic_visit(node)
        self.stack.pop()


def detect_redundant(graph):
    REDUNDANTS = ("Load", "Module", "Store")

    candidates = []
    for node in list(graph.nodes):
        for r in REDUNDANTS:
            if r in node:
                candidates.append(node)

    return candidates


def generate_and_save_ast(func, path="artifacts/ast.png"):
    ast_object = ast.parse(inspect.getsource(func))

    tree = Tree()
    tree.visit(ast_object)

    tree.graph.remove_nodes_from(detect_redundant(tree.graph))

    p = nx.drawing.nx_pydot.to_pydot(tree.graph)
    p.write_png(path)

