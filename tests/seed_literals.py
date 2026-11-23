"""Seed-range literals in executable expressions, excluding declared historical ranges."""
import ast
import operator


def seed_values(source):
    tree = ast.parse(source)
    # Exclusions document other generators; do not ignore the rest of the declaring module.
    class WithoutExclusions(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'EARLIER' for t in node.targets):
                return None
            return self.generic_visit(node)

    tree = WithoutExclusions().visit(tree)
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Pow: operator.pow}

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            a, b = value(node.left), value(node.right)
            if a is not None and b is not None and (not isinstance(node.op, ast.Pow) or 0 <= b <= 64):
                return ops[type(node.op)](a, b)
        return None

    return [v for node in ast.walk(tree) if (v := value(node)) is not None and v >= 10 ** 18]
