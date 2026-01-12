import ast
import asttokens

COMPOUND = (
    ast.If, ast.For, ast.AsyncFor, ast.While,
    ast.With, ast.AsyncWith,
    ast.Try,
    ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Match,
)

def stmt_unit_span(atok, node):
    start = node.first_token.startpos
    full_end = node.last_token.endpos

    if isinstance(node, COMPOUND):
        for tok in atok.get_tokens(node, include_extra=False):
            if tok.string == ":":
                return start, tok.endpos
        return start, full_end

    return start, full_end

def display_units(src: str):
    atok = asttokens.ASTTokens(src, parse=True)
    tree = atok.tree

    counter = {"id": 0}  # mutable counter

    def rec(stmts, indent=""):
        for st in stmts:
            counter["id"] += 1
            sid = counter["id"]

            s, e = stmt_unit_span(atok, st)
            text = src[s:e].rstrip("\n")

            print(f"{indent}#{sid} {type(st).__name__} chars[{s},{e})")
            print(indent + text.replace("\n", "\n" + indent))
            print(indent + "-" * 50)

            # recurse into nested blocks
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(st, field, None)
                if inner:
                    rec(inner, indent + "  ")

            handlers = getattr(st, "handlers", None)
            if handlers:
                for h in handlers:
                    rec(h.body, indent + "  ")

    rec(tree.body)


if __name__ == "__main__":
    code = """\
class Wow:
  f=4
  def __init__(self):
     self.zz=3
def Pi():
   return 3.14
x = 1; y = 2
# Hello World!
if x > 0:
    y = x + 3
    z = 5
    while (z>y):
       print("hi"*z)
       z-=1
else:
    for x in range(3):
        print(x+2)
print(y)
"""
    display_units(code)
