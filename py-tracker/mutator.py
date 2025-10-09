import ast, argparse, copy, tokenize, io

# ---------------------
# Operator families
# ---------------------
ARITHMETIC = {
    ast.Add: ("+", [ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow]),
    ast.Sub: ("-", [ast.Add, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow]),
    ast.Mult: ("*", [ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow]),
    ast.Div: ("/", [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow]),
    ast.FloorDiv: ("//", [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow]),
    ast.Mod: ("%", [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow]),
    ast.Pow: ("**", [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod]),
}
BITWISE = {
    ast.BitAnd: ("&", [ast.BitOr, ast.BitXor, ast.LShift, ast.RShift]),
    ast.BitOr: ("|", [ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift]),
    ast.BitXor: ("^", [ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift]),
    ast.LShift: ("<<", [ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor]),
    ast.RShift: (">>", [ast.LShift, ast.BitAnd, ast.BitOr, ast.BitXor]),
}
BOOLEAN_OP = {ast.And: ("and", [ast.Or]), ast.Or: ("or", [ast.And])}
COMPARISONS = {
    ast.Lt: ("<", [ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]),
    ast.LtE: ("<=", [ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]),
    ast.Gt: (">", [ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq]),
    ast.GtE: (">=", [ast.Gt, ast.Lt, ast.LtE, ast.Eq, ast.NotEq]),
    ast.Eq: ("==", [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE]),
    ast.NotEq: ("!=", [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE]),
    ast.Is: ("is", [ast.IsNot]),
    ast.IsNot: ("is not", [ast.Is]),
    ast.In: ("in", [ast.NotIn]),
    ast.NotIn: ("not in", [ast.In]),
}
UNARY = {
    ast.UAdd: ("+", [ast.USub, ast.Not, ast.Invert]),
    ast.USub: ("-", [ast.UAdd, ast.Not, ast.Invert]),
    ast.Not: ("not", [ast.UAdd, ast.USub, ast.Invert]),
    ast.Invert: ("~", [ast.UAdd, ast.USub, ast.Not]),
}

# ---------------------
# Helpers
# ---------------------
def find_operator_span(line_text, op_str):
    if " " not in op_str:
        idx = line_text.find(op_str)
        if idx != -1:
            return idx, idx + len(op_str)
        return None, None
    parts = op_str.split()
    tokens = list(tokenize.generate_tokens(io.StringIO(line_text).readline))
    for i in range(len(tokens) - len(parts) + 1):
        if all(tokens[i + j].string == parts[j] for j in range(len(parts))):
            return tokens[i].start[1], tokens[i + len(parts) - 1].end[1]
    return None, None

class SingleMutator(ast.NodeTransformer):
    def __init__(self, target, make_new):
        self.target = target
        self.make_new = make_new
    def visit(self, node):
        if node is self.target:
            return self.make_new()
        return super().visit(node)

class Collector(ast.NodeVisitor):
    def __init__(self, source):
        self.source = source
        self.source_lines = source.splitlines()
        self.mutants = []
        self.root = ast.parse(source)

    def collect(self):
        self.visit(self.root)
        return self.mutants

    def record_mutant(self, node, make_new, mutation_type, before, after):
        line = getattr(node, "lineno", None)
        start = getattr(node, "col_offset", None)
        end = getattr(node, "end_col_offset", None)
        orig_line = self.source_lines[line - 1] if line else ""
        start_col, end_col = find_operator_span(orig_line, before)
        if start_col is None:
            start_col, end_col = start, end
        abs_start = self._linecol_to_offset(line, start_col)
        abs_end = self._linecol_to_offset(line, end_col)

        new_tree = copy.deepcopy(self.root)
        mutator = SingleMutator(self._find_equivalent_node(self.root, new_tree, node), make_new)
        new_tree = mutator.visit(new_tree)
        ast.fix_missing_locations(new_tree)
        mutated_src = ast.unparse(new_tree)

        self.mutants.append({
            "mutation_type": mutation_type,
            "line": line,
            "start_pos": abs_start,
            "end_pos": abs_end - 1,
            "original_line": orig_line,
            "before": before,
            "after": after,
            "mutated_code": mutated_src,
        })

    def _linecol_to_offset(self, lineno, col):
        if lineno is None or col is None:
            return None
        lines = self.source.splitlines(keepends=True)
        return sum(len(lines[i]) for i in range(lineno - 1)) + col

    def _find_equivalent_node(self, orig, copy_tree, target):
        if orig is target:
            return copy_tree
        for (ofield, oval), (cfield, cval) in zip(ast.iter_fields(orig), ast.iter_fields(copy_tree)):
            if isinstance(oval, list) and isinstance(cval, list):
                for o_item, c_item in zip(oval, cval):
                    if isinstance(o_item, ast.AST):
                        r = self._find_equivalent_node(o_item, c_item, target)
                        if r:
                            return r
            elif isinstance(oval, ast.AST) and isinstance(cval, ast.AST):
                r = self._find_equivalent_node(oval, cval, target)
                if r:
                    return r
        return None

    # ---- Visitors ----
    def visit_BinOp(self, node):
        self.generic_visit(node)
        op_map = {**ARITHMETIC, **BITWISE}
        if type(node.op) in op_map:
            before, alts = op_map[type(node.op)]
            for alt in alts:
                self.record_mutant(
                    node,
                    lambda alt=alt: ast.BinOp(left=copy.deepcopy(node.left), op=alt(), right=copy.deepcopy(node.right)),
                    f"BinOp {before}->{alt.__name__}",
                    before,
                    alt.__name__,
                )

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if type(node.op) in BOOLEAN_OP:
            before, alts = BOOLEAN_OP[type(node.op)]
            for alt in alts:
                self.record_mutant(
                    node,
                    lambda alt=alt: ast.BoolOp(op=alt(), values=[copy.deepcopy(v) for v in node.values]),
                    f"BoolOp {before}->{alt.__name__}",
                    before,
                    alt.__name__,
                )

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in COMPARISONS:
            before, alts = COMPARISONS[type(node.ops[0])]
            for alt in alts:
                self.record_mutant(
                    node,
                    lambda alt=alt: ast.Compare(left=copy.deepcopy(node.left), ops=[alt()],
                                                comparators=[copy.deepcopy(node.comparators[0])]),
                    f"Compare {before}->{alt.__name__}",
                    before,
                    alt.__name__,
                )

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if type(node.op) in UNARY:
            before, alts = UNARY[type(node.op)]
            for alt in alts:
                self.record_mutant(
                    node,
                    lambda alt=alt: ast.UnaryOp(op=alt(), operand=copy.deepcopy(node.operand)),
                    f"Unary {before}->{alt.__name__}",
                    before,
                    alt.__name__,
                )

    def visit_Constant(self, node):
        self.generic_visit(node)
        if isinstance(node.value, bool):
            self.record_mutant(
                node,
                lambda: ast.Constant(value=not node.value),
                "Bool flip", str(node.value), str(not node.value)
            )

def generate_mutants(source):
    return Collector(source).collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Python source file to mutate")
    parser.add_argument("--show-mutants", action="store_true")
    args = parser.parse_args()

    src = open(args.file, encoding="utf8").read()
    mutants = generate_mutants(src)

    for i, m in enumerate(mutants, 1):
        print(f"[{i}] {m['mutation_type']} line {m['line']} {m['before']} → {m['after']}")
        if args.show_mutants:
            print("--- Mutated Source ---\n" + m["mutated_code"] + "\n----------------------")

