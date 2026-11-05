"""Assign non-attributable mutants (mu_spec_IDs.csv) to fix-relative regions.

  r1: a fix-modified method
  r2: a direct caller or callee of a fix-modified method, excluding r1
  r3: another method in a fix-modified file, excluding r1 and r2
  r4: a remaining method in an imported or importing one-hop-neighbor file,
      excluding r1, r2, and r3

All analysis is done on the buggy-commit snapshot (the commit mutants were
injected into). Call edges are resolved by name, without type inference.

Usage (from py-tracker/RQs):  python rq2_region_assign.py
"""

import ast
import csv
import json
import os
import re
import subprocess
import sys
import warnings as pywarnings
from collections import Counter, defaultdict
from pathlib import Path

pywarnings.filterwarnings("ignore", category=SyntaxWarning)

BUGSINPY = Path("BUGSINPY Path here")
HERE = Path(__file__).resolve().parent
# RQs -> py-tracker -> tracker -> tools -> longmu
LONGMU = HERE.parents[3]
DATA_DIR = HERE / "data"
LOCAL_REPOS = HERE / "repos"
# Full status sequences already live next to this repo. The manual
# attributable / mutation-specific list is copied into RQs/data.
DATA_SEQ = LONGMU / "data_seq" if (LONGMU / "data_seq").is_dir() else HERE / "data_seq"
IN_CSV = DATA_DIR / "mu_spec_IDs.csv"
OUT_CSV = DATA_DIR / "mu_region.csv"
# Clones already made for projects that are not in the BugsInPy workspace.
EXTRA_REPOS = [
    LOCAL_REPOS,
    Path("REPOSITORIES Path here"),
]

REGIONS = ["r1", "r2", "r3", "r4", "outside", "module-level", "unresolved"]

warnings = []


def warn(msg):
    warnings.append(msg)
    print(f"[warn] {msg}", file=sys.stderr)


# ------------------------------------------------------------------
# Repository access
# ------------------------------------------------------------------

def repo_dir(project):
    ws = BUGSINPY / "workspace" / project
    if (ws / ".git").exists():
        return ws
    for base in EXTRA_REPOS:
        local = base / project
        if (local / ".git").exists():
            return local
    local = LOCAL_REPOS / project
    info = (BUGSINPY / "projects" / project / "project.info").read_text()
    url = re.search(r'github_url="([^"]+)"', info).group(1)
    LOCAL_REPOS.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", "-q", url, str(local)], check=True)
    return local


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="replace")


def read_blobs(repo, commit, paths):
    """Read many files at a commit in one `git cat-file --batch` call."""
    if not paths:
        return {}
    req = "".join(f"{commit}:{p}\n" for p in paths).encode()
    out = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=req, capture_output=True, check=True,
    ).stdout
    blobs, pos = {}, 0
    for p in paths:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].split()
        pos = nl + 1
        if header[-1] == b"missing":
            continue
        size = int(header[2])
        blobs[p] = out[pos:pos + size].decode("utf-8", errors="replace")
        pos += size + 1
    return blobs


def bug_info(project, bug_num):
    text = (BUGSINPY / "projects" / project / "bugs" / bug_num / "bug.info").read_text()
    info = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', text))
    return info["buggy_commit_id"], info["fixed_commit_id"]


def is_test_path(path):
    parts = path.split("/")
    name = parts[-1]
    return (
        any(p in ("test", "tests", "testing") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


# ------------------------------------------------------------------
# AST indexing of a snapshot
# ------------------------------------------------------------------

class FileIndex:
    def __init__(self, path, src):
        self.path = path
        self.src = src
        self.defs = {}          # qualname -> dict(start, end, calls, cls)
        self.toplevel = set()   # top-level function names
        self.classes = {}       # class name -> set(method names)
        self.import_targets = []  # raw (module, level, names) tuples
        self.name_imports = {}  # local name -> (module, level, orig_name)
        self.module_aliases = {}  # local name -> (module, level)
        self.tree = ast.parse(src)
        self._walk(self.tree, [], None)
        self._collect_imports()

    def _walk(self, node, scope, cls):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join(scope + [ch.name])
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                self.defs[qual] = {
                    "start": start,
                    "end": ch.end_lineno,
                    "calls": own_calls(ch),
                    "cls": cls,
                    "name": ch.name,
                }
                if not scope:
                    self.toplevel.add(ch.name)
                if cls is not None and scope and scope[-1] == cls:
                    self.classes.setdefault(cls, set()).add(ch.name)
                self._walk(ch, scope + [ch.name], None)
            elif isinstance(ch, ast.ClassDef):
                if not scope:
                    self.classes.setdefault(ch.name, set())
                self._walk(ch, scope + [ch.name], ch.name)
            else:
                self._walk(ch, scope, cls)

    def _collect_imports(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.import_targets.append((a.name, 0, []))
                    if a.asname:
                        self.module_aliases[a.asname] = (a.name, 0)
                    else:
                        self.module_aliases[a.name.split(".")[0]] = (a.name.split(".")[0], 0)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [a.name for a in node.names]
                self.import_targets.append((mod, node.level, names))
                for a in node.names:
                    local = a.asname or a.name
                    self.name_imports[local] = (mod, node.level, a.name)

    def innermost_def(self, line):
        best = None
        for qual, d in self.defs.items():
            if d["start"] <= line <= d["end"]:
                if best is None or d["start"] >= self.defs[best]["start"]:
                    best = qual
        return best


def own_calls(func_node):
    """Calls made directly in a def body, excluding nested defs/classes."""
    calls = []
    stack = list(ast.iter_child_nodes(func_node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if isinstance(n, ast.Lambda):
                stack.extend(ast.iter_child_nodes(n))
            continue
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                calls.append(("name", f.id, None))
            elif isinstance(f, ast.Attribute):
                base = f.value.id if isinstance(f.value, ast.Name) else None
                calls.append(("attr", f.attr, base))
        stack.extend(ast.iter_child_nodes(n))
    return calls


def module_name(path, all_paths):
    """Module name relative to the topmost package root."""
    parts = path[:-3].split("/")
    dirs = parts[:-1]
    root = len(dirs)
    while root > 0 and "/".join(dirs[:root] + ["__init__.py"]) in all_paths:
        root -= 1
    mod = parts[root:]
    if mod[-1] == "__init__":
        mod = mod[:-1]
    return ".".join(mod)


class Snapshot:
    def __init__(self, repo, commit):
        self.repo = repo
        self.commit = commit
        listing = git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
        self.all_py = {p for p in listing if p.endswith(".py")}
        src_paths = sorted(p for p in self.all_py if not is_test_path(p))
        blobs = read_blobs(repo, commit, src_paths)
        self.files = {}
        for p, src in blobs.items():
            try:
                self.files[p] = FileIndex(p, src)
            except SyntaxError as e:
                warn(f"{commit[:8]} {p}: unparseable ({e.msg})")
        self.mod2file = {}
        for p in self.files:
            self.mod2file.setdefault(module_name(p, self.all_py), p)
        self.file2mod = {p: module_name(p, self.all_py) for p in self.files}
        self._build_lookup()
        self._build_imports()
        self._build_call_graph()

    # --- name tables ---
    def _build_lookup(self):
        self.methods_by_name = defaultdict(set)
        for p, fi in self.files.items():
            for qual, d in fi.defs.items():
                if d["cls"] is not None:
                    self.methods_by_name[d["name"]].add((p, qual))

    def resolve_module(self, importer, mod, level):
        if level:
            base = self.file2mod[importer].split(".")
            is_pkg = importer.endswith("__init__.py")
            base = base if is_pkg else base[:-1]
            if level - 1:
                base = base[: len(base) - (level - 1)]
            mod = ".".join([x for x in base + ([mod] if mod else []) if x])
        parts = mod.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if cand in self.mod2file:
                return cand
        return None

    def _build_imports(self):
        self.imports = defaultdict(set)  # file -> imported files
        for p, fi in self.files.items():
            for mod, level, names in fi.import_targets:
                rm = self.resolve_module(p, mod, level)
                if rm is None:
                    continue
                hit_sub = False
                for n in names:
                    sub = f"{rm}.{n}" if rm else n
                    if sub in self.mod2file:
                        self.imports[p].add(self.mod2file[sub])
                        hit_sub = True
                if not hit_sub or not names:
                    self.imports[p].add(self.mod2file[rm])
            self.imports[p].discard(p)
        self.imported_by = defaultdict(set)
        for p, targets in self.imports.items():
            for t in targets:
                self.imported_by[t].add(p)

    # --- call resolution ---
    def func_in_file(self, path, name):
        fi = self.files.get(path)
        if fi is None:
            return set()
        if name in fi.toplevel:
            return {(path, name)}
        if name in fi.classes and "__init__" in fi.classes[name]:
            return {(path, f"{name}.__init__")}
        return set()

    def resolve_call(self, path, qual, call):
        fi = self.files[path]
        kind, name, base = call
        if kind == "name":
            parts = qual.split(".")
            for i in range(len(parts), 0, -1):
                nested = ".".join(parts[:i] + [name])
                if nested in fi.defs:
                    return {(path, nested)}
            local = self.func_in_file(path, name)
            if local:
                return local
            if name in fi.name_imports:
                mod, level, orig = fi.name_imports[name]
                rm = self.resolve_module(path, mod, level)
                if rm is not None:
                    return self.func_in_file(self.mod2file[rm], orig)
            return set()
        # attribute call
        if base in fi.module_aliases:
            mod, level = fi.module_aliases[base]
            rm = self.resolve_module(path, mod, level)
            if rm is not None:
                return self.func_in_file(self.mod2file[rm], name)
        if base in fi.name_imports:
            mod, level, orig = fi.name_imports[base]
            rm = self.resolve_module(path, mod, level)
            if rm is not None:
                sub = f"{rm}.{orig}"
                if sub in self.mod2file:
                    return self.func_in_file(self.mod2file[sub], name)
        if name.startswith("__") and name.endswith("__"):
            return set()
        if base in ("self", "cls"):
            cls = qual.split(".")[0]
            same = (path, f"{cls}.{name}")
            if same[1] in fi.defs:
                return {same}
        return set(self.methods_by_name.get(name, ()))

    def _build_call_graph(self):
        self.callees = defaultdict(set)
        self.callers = defaultdict(set)
        for p, fi in self.files.items():
            for qual, d in fi.defs.items():
                for call in d["calls"]:
                    for tgt in self.resolve_call(p, qual, call):
                        if tgt != (p, qual):
                            self.callees[(p, qual)].add(tgt)
                            self.callers[tgt].add((p, qual))


# ------------------------------------------------------------------
# Fix diff -> r1
# ------------------------------------------------------------------

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def diff_hunks(repo, buggy, fixed):
    """{path: [(old_start, old_len, new_start, new_len), ...]} for non-test .py files."""
    out = git(repo, "diff", "-U0", "--no-renames", buggy, fixed, "--", "*.py")
    hunks, cur, in_header = defaultdict(list), None, False
    for line in out.splitlines():
        if line.startswith("diff --git "):
            cur, in_header = None, True
            continue
        if in_header:
            if line.startswith("--- "):
                # Files added by the fix ("--- /dev/null") have no buggy-side methods.
                cur = line[6:] if line.startswith("--- a/") else None
            if not line.startswith("@@"):
                continue
            in_header = False
        m = HUNK.match(line)
        if m and cur and not is_test_path(cur):
            os_, ol, ns, nl = m.groups()
            hunks[cur].append((int(os_), int(ol or 1), int(ns), int(nl or 1)))
    return hunks


def patch_files(project, bug_num):
    text = (BUGSINPY / "projects" / project / "bugs" / bug_num / "bug_patch.txt").read_text(
        errors="replace")
    return {m for m in re.findall(r"^diff --git a/(\S+) b/", text, re.M)
            if m.endswith(".py") and not is_test_path(m)}


def r1_methods(snap, fixed_files, hunks):
    """Defs touched by the fix. Removed lines map via the buggy file; added lines
    map via the fixed file, keeping only defs whose qualname exists in the buggy file."""
    r1, new_defs = set(), {}
    for path, hs in hunks.items():
        old_fi = snap.files.get(path)
        new_fi = fixed_files.get(path)
        for os_, ol, ns, nl in hs:
            if old_fi and ol > 0:
                for line in range(os_, os_ + ol):
                    q = old_fi.innermost_def(line)
                    if q:
                        r1.add((path, q))
            if new_fi and nl > 0:
                for line in range(ns, ns + nl):
                    q = new_fi.innermost_def(line)
                    if not q:
                        continue
                    if old_fi and q in old_fi.defs:
                        r1.add((path, q))
                    else:
                        new_defs[(path, q)] = new_fi.defs[q]
    return r1, new_defs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def scope_matches(scope, qual):
    if not scope and not qual:
        return True
    if not scope or not qual:
        return False
    return scope == qual or qual.endswith("." + scope) or scope.endswith("." + qual)


def analyze_bug(project, bug, mutants):
    bug_num = re.match(rf"{re.escape(project)}-(\d+)b$", bug).group(1)
    repo = repo_dir(project)
    buggy, fixed = bug_info(project, bug_num)
    snap = Snapshot(repo, buggy)

    hunks = diff_hunks(repo, buggy, fixed)
    fix_files = {p for p in hunks if p in snap.files}
    expected = patch_files(project, bug_num)
    if expected and expected != set(hunks):
        warn(f"{bug}: git diff files {sorted(hunks)} != bug_patch.txt files {sorted(expected)}")

    fixed_blobs = read_blobs(repo, fixed, sorted(hunks))
    fixed_files = {}
    for p, src in fixed_blobs.items():
        try:
            fixed_files[p] = FileIndex(p, src)
        except SyntaxError as e:
            warn(f"{bug} fixed {p}: unparseable ({e.msg})")

    r1, new_defs = r1_methods(snap, fixed_files, hunks)

    # Callees of fix-modified methods: union of buggy- and fixed-version calls.
    r2 = set()
    for key in r1:
        r2 |= snap.callees.get(key, set())
        r2 |= snap.callers.get(key, set())
        fixed_fi = fixed_files.get(key[0])
        if fixed_fi and key[1] in fixed_fi.defs:
            for call in fixed_fi.defs[key[1]]["calls"]:
                r2 |= snap.resolve_call(key[0], key[1], call)
    # Defs added by the fix cannot host mutants, but their callees are callees
    # of fix-modified code.
    for (path, qual), d in new_defs.items():
        if path in snap.files:
            for call in d["calls"]:
                r2 |= snap.resolve_call(path, qual, call)
    r2 -= r1

    neighbor_files = set()
    for f in fix_files:
        neighbor_files |= snap.imports.get(f, set())
        neighbor_files |= snap.imported_by.get(f, set())
    neighbor_files -= fix_files

    data_path = DATA_SEQ / f"{bug}_rq1.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else None

    results = []
    for row in mutants:
        res = {**row, "file": "", "method": "", "region": "unresolved",
               "scope_match": "", "note": ""}
        mid = row["mutant_id"]
        res["file"] = mid.rsplit("#", 1)[0]
        if data is None or mid not in data:
            rel = "fix-modified" if res["file"] in fix_files else (
                "import-neighbor" if res["file"] in neighbor_files else "other")
            res["note"] = f"no position data; file is {rel}"
            if rel == "fix-modified":
                res["note"] += " (one of r1/r2/r3)"
            results.append(res)
            continue

        info = data[mid]["info"]
        path = info["file"]
        res["file"] = path
        fi = snap.files.get(path)
        if fi is None:
            res["note"] = "file missing or unparseable at buggy commit"
            results.append(res)
            continue

        line = fi.src.count("\n", 0, info["mutation_start_pos"]) + 1
        qual = fi.innermost_def(line)
        res["method"] = qual or ""
        scope = info.get("scope", "")
        if qual is None and scope in fi.classes:
            res["scope_match"] = "yes"
            res["note"] = f"class body of {scope}"
        elif scope_matches(scope, qual):
            res["scope_match"] = "yes"
        else:
            res["scope_match"] = "no"
            res["note"] = f"data scope={scope!r}"

        key = (path, qual)
        if qual is None:
            res["region"] = "module-level"
            rel = "fix-modified" if path in fix_files else (
                "import-neighbor" if path in neighbor_files else "other")
            res["note"] = (res["note"] + "; " if res["note"] else "") + f"file is {rel}"
        elif key in r1:
            res["region"] = "r1"
        elif key in r2:
            res["region"] = "r2"
        elif path in fix_files:
            res["region"] = "r3"
        elif path in neighbor_files:
            res["region"] = "r4"
        else:
            res["region"] = "outside"
        results.append(res)

    summary = {
        "fix_files": sorted(fix_files),
        "r1": sorted(f"{p}::{q}" for p, q in r1),
        "n_r2": len(r2),
        "n_neighbor_files": len(neighbor_files),
    }
    return results, summary


def main():
    rows = list(csv.DictReader(IN_CSV.open()))
    by_bug = defaultdict(list)
    for r in rows:
        by_bug[(r["project"], r["bug"])].append(r)

    all_results, bug_summaries = [], {}
    for (project, bug), mutants in sorted(by_bug.items()):
        print(f"analyzing {bug} ({len(mutants)} mutants)", file=sys.stderr)
        results, summary = analyze_bug(project, bug, mutants)
        all_results.extend(results)
        bug_summaries[bug] = summary

    fields = ["project", "bug", "mutant_id", "direction", "mutation_specific",
              "file", "method", "region", "scope_match", "note"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in fields})

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "mu_region_bugs.json").write_text(json.dumps(bug_summaries, indent=2))

    print(f"\nWrote {OUT_CSV.name} ({len(all_results)} mutants)\n")
    per_project = defaultdict(Counter)
    for r in all_results:
        per_project[r["project"]][r["region"]] += 1
    total = Counter(r["region"] for r in all_results)
    cols = [c for c in REGIONS if total[c]]
    print(f"{'project':<14}" + "".join(f"{c:>14}" for c in cols) + f"{'total':>8}")
    for p in sorted(per_project):
        c = per_project[p]
        print(f"{p:<14}" + "".join(f"{c[x]:>14}" for x in cols) + f"{sum(c.values()):>8}")
    print(f"{'ALL':<14}" + "".join(f"{total[x]:>14}" for x in cols) + f"{sum(total.values()):>8}")

    mism = sum(1 for r in all_results if r["scope_match"] == "no")
    print(f"\nscope mismatches: {mism}; warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
