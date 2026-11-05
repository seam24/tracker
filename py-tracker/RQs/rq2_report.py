import ast
import json
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


# ============================================================
# CONFIG
# ============================================================

RES_DIR = Path("D://res")
PATTERN = "*_rq1.json"

BUGSINPY_ROOT = Path("BUGSINPY projects path here")

COMMITS_ROOT = Path("commits2 path here")

REGIONS = [
    "r1_faulty",
    "r2_caller_callee",
    "r3_buggy_file_other",
    "r4_other",
]

REGION_LABELS = {
    "r1_faulty": "r1",
    "r2_caller_callee": "r2",
    "r3_buggy_file_other": "r3",
    "r4_other": "r4",
}

CATS = [
    "Stable S",
    "Stable K",
    "S->K",
    "K->S",
    "S->K->S",
    "K->S->K",
    "Recurrent",
]

STABLE_CATS = {"Stable S", "Stable K"}

TRANSITION_CATS = {
    "S->K",
    "K->S",
    "S->K->S",
    "K->S->K",
    "Recurrent",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def median_iqr(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return "NA"
    return f"{np.median(values):.1f} [{np.percentile(values, 25):.1f}, {np.percentile(values, 75):.1f}]"


def latex_escape(s):
    return str(s).replace("_", r"\_")


def fmt_pct(x):
    return "NA" if x is None else f"{x:.1f}%"


def latex_pct(x):
    return "NA" if x is None else f"{x:.1f}\\%"


def instance_from_path(path):
    name = path.name
    return name[:-len("_rq1.json")] if name.endswith("_rq1.json") else path.stem


def parse_instance(instance):
    m = re.match(r"(.+)-(\d+)b$", instance)
    if not m:
        raise ValueError(f"Cannot parse instance name: {instance}")
    pid = m.group(1)
    bid = m.group(2)
    return pid, bid, f"{bid}b"


def get_mutants(data):
    if isinstance(data, dict) and "mutants" in data and isinstance(data["mutants"], dict):
        return data["mutants"]
    return data


def get_count(m, *names):
    for name in names:
        if name in m:
            return int(m.get(name) or 0)
    return 0


def get_file(m):
    info = m.get("info", {}) if isinstance(m.get("info", {}), dict) else {}
    meta = m.get("meta", {}) if isinstance(m.get("meta", {}), dict) else {}

    return str(
        info.get("file")
        or meta.get("file")
        or m.get("file")
        or ""
    ).replace("\\", "/")


def get_element(m):
    info = m.get("info", {}) if isinstance(m.get("info", {}), dict) else {}
    meta = m.get("meta", {}) if isinstance(m.get("meta", {}), dict) else {}

    module = info.get("module") or meta.get("module") or m.get("module") or ""
    scope = info.get("scope") or meta.get("scope") or m.get("scope") or ""
    file_path = info.get("file") or meta.get("file") or m.get("file") or "UNKNOWN_FILE"

    module = str(module).strip()
    scope = str(scope).strip()
    file_path = str(file_path).replace("\\", "/").strip()

    if module and scope:
        return module

    if module and not scope:
        return f"{module}.<module>"

    return f"{file_path}.<module>"


def norm_name(s):
    return str(s).strip().replace("\\", ".").replace("/", ".").replace(":", ".")


def element_matches_any(element, targets):
    element = norm_name(element).replace(".<module>", "")

    for t in targets:
        t = norm_name(t)
        if element == t:
            return True
        if element.startswith(t + "."):
            return True
        if t.startswith(element + "."):
            return True

    return False


def file_matches(mut_file, target_files):
    mut_file = mut_file.replace("\\", "/")

    for f in target_files:
        f = f.replace("\\", "/")
        if mut_file == f or mut_file.endswith(f) or f.endswith(mut_file):
            return True

    return False


# ============================================================
# MUTANT CLASSIFICATION
# ============================================================

def classify_mutant(m):
    ss = get_count(m, "S_to_S", "S->S")
    kk = get_count(m, "K_to_K", "K->K")
    sk = get_count(m, "S_to_K", "S->K")
    ks = get_count(m, "K_to_S", "K->S")

    num_sks = get_count(m, "num_SKS", "S_K_S", "S->K->S")
    num_ksk = get_count(m, "num_KSK", "K_S_K", "K->S->K")

    valid = ss + kk + sk + ks

    if valid == 0:
        return "Noisy"

    if num_sks + num_ksk >= 2:
        return "Recurrent"

    if num_sks == 1 and num_ksk == 0:
        return "S->K->S"

    if num_ksk == 1 and num_sks == 0:
        return "K->S->K"

    if sk > 0 and ks == 0:
        return "S->K"

    if ks > 0 and sk == 0:
        return "K->S"

    if ss > 0 and kk == 0:
        return "Stable S"

    if kk > 0 and ss == 0:
        return "Stable K"

    return "Noisy"


# ============================================================
# PATCH PARSING
# ============================================================

def parse_patch_changed_ranges(patch_text):
    out = {}
    cur_new_file = None
    cur_old_file = None

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            cur_new_file = None
            cur_old_file = None
            continue

        if line.startswith("--- a/"):
            cur_old_file = line[len("--- a/"):].strip()
            continue

        if line.startswith("+++ b/"):
            cur_new_file = line[len("+++ b/"):].strip()
            out.setdefault(
                cur_new_file,
                {
                    "old_file": cur_old_file,
                    "new_file": cur_new_file,
                    "old_ranges": [],
                    "new_ranges": [],
                },
            )
            continue

        if line.startswith("@@") and cur_new_file:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                continue

            old_start = int(m.group(1))
            old_len = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_len = int(m.group(4) or "1")

            if old_len > 0:
                out[cur_new_file]["old_ranges"].append(
                    (old_start, old_start + old_len - 1)
                )

            if new_len > 0:
                out[cur_new_file]["new_ranges"].append(
                    (new_start, new_start + new_len - 1)
                )

    return out


def ranges_overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or b_end < a_start)


def span_overlaps_ranges(start, end, ranges):
    if start is None or end is None:
        return False

    return any(
        ranges_overlap(start, end, rs, re_)
        for rs, re_ in ranges
    )


# ============================================================
# AST INDEX
# ============================================================

def path_to_module(repo_root, py_path):
    rel = py_path.relative_to(repo_root).as_posix()

    if rel.endswith(".py"):
        rel = rel[:-3]

    parts = rel.split("/")

    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


class RepoIndexer(ast.NodeVisitor):
    def __init__(self, rel_file, module_name):
        self.rel_file = rel_file.as_posix()
        self.module_name = module_name
        self.stack = []
        self.functions = []
        self.import_aliases = {}

    def visit_Import(self, node):
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.import_aliases[local] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                self.import_aliases[local] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._add_scope(node, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._add_scope(node, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self._add_scope(node, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _add_scope(self, node, kind):
        qual_inside_module = ".".join(self.stack + [node.name])
        qualname = f"{self.module_name}.{qual_inside_module}"

        self.functions.append(
            {
                "kind": kind,
                "module": self.module_name,
                "qualname": qualname,
                "short_name": node.name,
                "file": self.rel_file,
                "start": getattr(node, "lineno", None),
                "end": getattr(node, "end_lineno", None),
                "depth": len(self.stack),
                "node": node,
                "import_aliases": dict(self.import_aliases),
            }
        )


def build_repo_index(repo_root):
    functions = []
    file_to_functions = defaultdict(list)
    qual_to_func = {}
    basename_to_quals = defaultdict(set)

    for py_path in repo_root.rglob("*.py"):
        try:
            text = read_text(py_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
        except Exception:
            continue

        rel_file = py_path.relative_to(repo_root)
        module_name = path_to_module(repo_root, py_path)

        indexer = RepoIndexer(rel_file, module_name)
        indexer.visit(tree)

        for f in indexer.functions:
            if f["start"] is None or f["end"] is None:
                continue

            functions.append(f)
            file_to_functions[f["file"]].append(f)
            qual_to_func[f["qualname"]] = f
            basename_to_quals[f["short_name"]].add(f["qualname"])

    return {
        "functions": functions,
        "file_to_functions": file_to_functions,
        "qual_to_func": qual_to_func,
        "basename_to_quals": basename_to_quals,
    }


def select_innermost_faulty_methods(repo_index, patch_files):
    faulty = set()
    buggy_files = set()
    file_to_functions = repo_index["file_to_functions"]

    for diff_file, info in patch_files.items():
        diff_file = diff_file.replace("\\", "/")
        buggy_files.add(diff_file)

        candidate_files = []

        if diff_file in file_to_functions:
            candidate_files.append(diff_file)
        else:
            for indexed_file in file_to_functions:
                if indexed_file.endswith(diff_file) or diff_file.endswith(indexed_file):
                    candidate_files.append(indexed_file)

        changed_ranges = info["old_ranges"] + info["new_ranges"]

        for changed_start, changed_end in changed_ranges:
            candidates = []

            for fpath in candidate_files:
                for f in file_to_functions[fpath]:
                    if f["kind"] == "class":
                        continue

                    if span_overlaps_ranges(
                        f["start"],
                        f["end"],
                        [(changed_start, changed_end)]
                    ):
                        candidates.append(f)

            if candidates:
                candidates.sort(key=lambda x: (-x["depth"], x["end"] - x["start"]))
                faulty.add(candidates[0]["qualname"])

    return faulty, buggy_files


# ============================================================
# CALL GRAPH
# ============================================================

class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        name = self.call_name(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)

    def call_name(self, func):
        if isinstance(func, ast.Name):
            return func.id

        if isinstance(func, ast.Attribute):
            parts = []
            cur = func

            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value

            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))

            return func.attr

        return None


def build_call_graph(repo_index):
    qual_to_func = repo_index["qual_to_func"]
    basename_to_quals = repo_index["basename_to_quals"]

    calls = defaultdict(set)
    called_by = defaultdict(set)

    for caller, fino in qual_to_func.items():
        if fino["kind"] == "class":
            continue

        collector = CallCollector()
        collector.visit(fino["node"])

        caller_module = fino["module"]
        caller_parts = caller.split(".")

        for raw_call in collector.calls:
            candidates = set()

            if raw_call.startswith("self.") or raw_call.startswith("cls."):
                attr = raw_call.split(".")[-1]
                class_prefix = ".".join(caller_parts[:-1])
                cand = f"{class_prefix}.{attr}"

                if cand in qual_to_func:
                    candidates.add(cand)

            if "." not in raw_call:
                cand = f"{caller_module}.{raw_call}"

                if cand in qual_to_func:
                    candidates.add(cand)

                candidates |= basename_to_quals.get(raw_call, set())

            if "." in raw_call:
                last = raw_call.split(".")[-1]
                candidates |= basename_to_quals.get(last, set())

                first = raw_call.split(".")[0]
                rest = raw_call.split(".")[1:]
                aliases = fino.get("import_aliases", {})

                if first in aliases:
                    resolved = ".".join([aliases[first]] + rest)

                    if resolved in qual_to_func:
                        candidates.add(resolved)

            for callee in candidates:
                if callee == caller:
                    continue

                if callee not in qual_to_func:
                    continue

                if qual_to_func[callee]["kind"] == "class":
                    continue

                calls[caller].add(callee)
                called_by[callee].add(caller)

    return calls, called_by


def direct_neighbours(faulty_methods, calls, called_by):
    out = set()

    for f in faulty_methods:
        out |= calls.get(f, set())
        out |= called_by.get(f, set())

    return out - faulty_methods


def assign_region(element, mut_file, faulty_methods, neighbour_methods, buggy_files):
    if element_matches_any(element, faulty_methods):
        return "r1_faulty"

    if element_matches_any(element, neighbour_methods):
        return "r2_caller_callee"

    if file_matches(mut_file, buggy_files):
        return "r3_buggy_file_other"

    return "r4_other"


# ============================================================
# MAIN COLLECTION
# ============================================================

files = sorted(RES_DIR.glob(PATTERN))

if not files:
    raise RuntimeError(f"No files found in {RES_DIR} matching {PATTERN}")

repo_cache = {}

pooled_region_counts = defaultdict(Counter)
project_region_counts = defaultdict(Counter)
element_region_counts = defaultdict(Counter)
element_region_set = defaultdict(set)

debug_transition_elements = []

total_noisy = 0
total_classified = 0

for rq1_path in files:
    instance = instance_from_path(rq1_path)
    pid, bid, bidb = parse_instance(instance)

    patch_path = BUGSINPY_ROOT / pid / "bugs" / bid / "bug_patch.txt"
    repo_root = COMMITS_ROOT / f"{pid}-{bidb}" / pid

    if not patch_path.exists():
        print(f"[WARN] missing patch: {patch_path}")
        continue

    if not repo_root.exists():
        print(f"[WARN] missing repo: {repo_root}")
        continue

    if str(repo_root) not in repo_cache:
        repo_index = build_repo_index(repo_root)
        calls, called_by = build_call_graph(repo_index)
        repo_cache[str(repo_root)] = (repo_index, calls, called_by)
    else:
        repo_index, calls, called_by = repo_cache[str(repo_root)]

    patch_files = parse_patch_changed_ranges(read_text(patch_path))
    faulty_methods, buggy_files = select_innermost_faulty_methods(repo_index, patch_files)
    neighbour_methods = direct_neighbours(faulty_methods, calls, called_by)

    mutants = get_mutants(load_json(rq1_path))

    for mid, m in mutants.items():
        cat = classify_mutant(m)

        if cat == "Noisy":
            total_noisy += 1
            continue

        total_classified += 1

        mut_file = get_file(m)
        element = get_element(m)

        region = assign_region(
            element=element,
            mut_file=mut_file,
            faulty_methods=faulty_methods,
            neighbour_methods=neighbour_methods,
            buggy_files=buggy_files,
        )

        element_key = (instance, pid, region, element)

        pooled_region_counts[region]["N"] += 1
        pooled_region_counts[region][cat] += 1

        project_region_counts[(pid, region)]["N"] += 1
        project_region_counts[(pid, region)][cat] += 1

        element_region_counts[element_key]["N"] += 1
        element_region_counts[element_key][cat] += 1

        element_region_set[region].add(element_key)

        if cat in TRANSITION_CATS:
            debug_transition_elements.append(
                {
                    "Project": pid,
                    "Instance": instance,
                    "Region": region,
                    "Element": element,
                    "Mutant": mid,
                    "Category": cat,
                }
            )


# ============================================================
# ELEMENT ROWS FOR TABLE IV
# ============================================================

element_rows = []

for (instance, pid, region, element), counts in element_region_counts.items():
    n = counts["N"]

    row = {
        "Instance": instance,
        "Project": pid,
        "Region": region,
        "Element": element,
        "N": n,
    }

    for c in CATS:
        row[c] = counts[c]
        row[f"{c}_density"] = pct(counts[c], n)

    row["AnyTransition"] = sum(counts[c] for c in TRANSITION_CATS)
    row["AnyTransition_density"] = pct(row["AnyTransition"], n)

    element_rows.append(row)


total_mutants = sum(pooled_region_counts[r]["N"] for r in REGIONS)
total_elements = sum(len(element_region_set[r]) for r in REGIONS)

projects = sorted({pid for pid, _region in project_region_counts})


# ============================================================
# TABLE IV: ELEMENT-LEVEL DENSITY
# ============================================================

print("\nTABLE IV. RQ2. Element-level density of mutant-status patterns by code region")
print("=" * 190)

print(
    f"{'Region':<24} "
    f"{'# Mutants (%)':>18} "
    f"{'# Elements (%)':>18} "
    f"{'Stable S':>18} "
    f"{'Stable K':>18} "
    f"{'S->K':>18} "
    f"{'K->S':>18} "
    f"{'S->K->S':>18} "
    f"{'K->S->K':>18} "
    f"{'Recurrent':>18}"
)

print("-" * 190)

table_iv_rows = []

for region in REGIONS:
    rows = [r for r in element_rows if r["Region"] == region and r["N"] > 0]

    n_mutants = pooled_region_counts[region]["N"]
    n_elements = len(element_region_set[region])

    out = {
        "Region": region,
        "# Mutants": n_mutants,
        "# Mutants %": pct(n_mutants, total_mutants),
        "# Elements": n_elements,
        "# Elements %": pct(n_elements, total_elements),
    }

    for c in CATS:
        out[c] = median_iqr([r[f"{c}_density"] for r in rows])

    table_iv_rows.append(out)

    print(
        f"{region:<24} "
        f"{f'{n_mutants} ({pct(n_mutants,total_mutants):.1f}%)':>18} "
        f"{f'{n_elements} ({pct(n_elements,total_elements):.1f}%)':>18} "
        f"{out['Stable S']:>18} "
        f"{out['Stable K']:>18} "
        f"{out['S->K']:>18} "
        f"{out['K->S']:>18} "
        f"{out['S->K->S']:>18} "
        f"{out['K->S->K']:>18} "
        f"{out['Recurrent']:>18}"
    )


# ============================================================
# TABLE V: COARSE STABLE VS TRANSITION
# ============================================================

TABLE_V_GROUPS = [
    ("Stable", STABLE_CATS),
    ("Status Transition", TRANSITION_CATS),
]

project_group_matrix = defaultdict(lambda: defaultdict(dict))

for pid in projects:
    for region in REGIONS:
        counts = project_region_counts.get((pid, region), Counter())
        n = counts["N"]

        for group_name, group_cats in TABLE_V_GROUPS:
            if n == 0:
                project_group_matrix[pid][group_name][region] = None
            else:
                group_count = sum(counts[c] for c in group_cats)
                project_group_matrix[pid][group_name][region] = pct(group_count, n)


def get_project_group_values(group_name, region):
    return [
        project_group_matrix[pid][group_name][region]
        for pid in projects
        if project_group_matrix[pid][group_name].get(region) is not None
    ]


def compute_group_summary(group_name, group_cats):
    overall = []
    mean = []
    median = []

    for region in REGIONS:
        counts = pooled_region_counts[region]
        n = counts["N"]
        group_count = sum(counts[c] for c in group_cats)

        overall.append(pct(group_count, n) if n else None)

        vals = get_project_group_values(group_name, region)
        mean.append(float(np.mean(vals)) if vals else None)
        median.append(float(np.median(vals)) if vals else None)

    return overall, mean, median


group_summary = {}

for group_name, group_cats in TABLE_V_GROUPS:
    group_summary[group_name] = compute_group_summary(group_name, group_cats)


print("\n\nTABLE V. Per-project percentage of mutants by coarse status category and code region")
print("=" * 150)

print(
    f"{'Project':<18} "
    f"{'Stable':^45} "
    f"{'Status Transition':^45}"
)
print(
    f"{'':<18} "
    f"{'r1':>10} {'r2':>10} {'r3':>10} {'r4':>10} "
    f"{'r1':>10} {'r2':>10} {'r3':>10} {'r4':>10}"
)
print("-" * 150)

for pid in projects:
    stable_cells = [
        fmt_pct(project_group_matrix[pid]["Stable"].get(region))
        for region in REGIONS
    ]
    trans_cells = [
        fmt_pct(project_group_matrix[pid]["Status Transition"].get(region))
        for region in REGIONS
    ]

    print(
        f"{pid:<18} "
        f"{stable_cells[0]:>10} {stable_cells[1]:>10} {stable_cells[2]:>10} {stable_cells[3]:>10} "
        f"{trans_cells[0]:>10} {trans_cells[1]:>10} {trans_cells[2]:>10} {trans_cells[3]:>10}"
    )

print("-" * 150)

for row_name, summary_index in [
    ("Overall pooled", 0),
    ("Project mean", 1),
    ("Project median", 2),
]:
    stable_cells = group_summary["Stable"][summary_index]
    trans_cells = group_summary["Status Transition"][summary_index]

    print(
        f"{row_name:<18} "
        f"{fmt_pct(stable_cells[0]):>10} {fmt_pct(stable_cells[1]):>10} "
        f"{fmt_pct(stable_cells[2]):>10} {fmt_pct(stable_cells[3]):>10} "
        f"{fmt_pct(trans_cells[0]):>10} {fmt_pct(trans_cells[1]):>10} "
        f"{fmt_pct(trans_cells[2]):>10} {fmt_pct(trans_cells[3]):>10}"
    )


# ============================================================
# TABLE VI: DETAILED CATEGORY TABLE
# ============================================================

project_cat_matrix = defaultdict(lambda: defaultdict(dict))

for pid in projects:
    for cat in CATS:
        for region in REGIONS:
            counts = project_region_counts.get((pid, region), Counter())
            n = counts["N"]

            if n == 0:
                project_cat_matrix[pid][cat][region] = None
            else:
                project_cat_matrix[pid][cat][region] = pct(counts[cat], n)


def get_project_cat_values(cat, region):
    return [
        project_cat_matrix[pid][cat][region]
        for pid in projects
        if project_cat_matrix[pid][cat].get(region) is not None
    ]


def compute_cat_summary(cat):
    overall = []
    mean = []
    median = []

    for region in REGIONS:
        counts = pooled_region_counts[region]
        n = counts["N"]

        overall.append(pct(counts[cat], n) if n else None)

        vals = get_project_cat_values(cat, region)
        mean.append(float(np.mean(vals)) if vals else None)
        median.append(float(np.median(vals)) if vals else None)

    return overall, mean, median


cat_summary = {}

for cat in CATS:
    cat_summary[cat] = compute_cat_summary(cat)


print("\n\nTABLE VI. Per-project percentage of mutants by detailed status category and code region")
print("=" * 260)

header = f"{'Project':<18} "
for cat in CATS:
    header += f"{cat:^36} "
print(header)

subheader = f"{'':<18} "
for _cat in CATS:
    subheader += f"{'r1':>8} {'r2':>8} {'r3':>8} {'r4':>8} "
print(subheader)

print("-" * 260)

for pid in projects:
    line = f"{pid:<18} "

    for cat in CATS:
        cells = [
            fmt_pct(project_cat_matrix[pid][cat].get(region))
            for region in REGIONS
        ]
        line += f"{cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8} "

    print(line)

print("-" * 260)

for row_name, summary_index in [
    ("Overall pooled", 0),
    ("Project mean", 1),
    ("Project median", 2),
]:
    line = f"{row_name:<18} "

    for cat in CATS:
        cells = cat_summary[cat][summary_index]
        line += (
            f"{fmt_pct(cells[0]):>8} {fmt_pct(cells[1]):>8} "
            f"{fmt_pct(cells[2]):>8} {fmt_pct(cells[3]):>8} "
        )

    print(line)


# ============================================================
# LATEX TABLE IV
# ============================================================

print("\n\nLaTeX Table IV")
print("=" * 130)

print(r"\begin{table*}[t]")
print(r"\centering")
print(
    r"\caption{RQ2. Element-level density of mutant-status patterns by code region. "
    r"Each cell reports the median and interquartile range of $D_c(m)$ across elements in the region. "
    r"The $S{\to}K{\to}S$ and $K{\to}S{\to}K$ columns denote exactly two status changes, "
    r"whereas Recurrent denotes mutants with more than two status changes.}"
)
print(r"\label{tab:rq2_element_density_region}")
print(r"\begin{tabular}{lrrlllllll}")
print(r"\toprule")
print(
    r"Region & \# Mutants (\%) & \# Elements (\%) & Stable $S$ & Stable $K$ & "
    r"$S{\to}K$ & $K{\to}S$ & $S{\to}K{\to}S$ & $K{\to}S{\to}K$ & Recurrent \\"
)
print(r"\midrule")

for r in table_iv_rows:
    print(
        f"{latex_escape(r['Region'])} & "
        f"{r['# Mutants']} ({r['# Mutants %']:.1f}\\%) & "
        f"{r['# Elements']} ({r['# Elements %']:.1f}\\%) & "
        f"{r['Stable S']} & "
        f"{r['Stable K']} & "
        f"{r['S->K']} & "
        f"{r['K->S']} & "
        f"{r['S->K->S']} & "
        f"{r['K->S->K']} & "
        f"{r['Recurrent']} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")


# ============================================================
# LATEX TABLE V
# ============================================================

print("\n\nLaTeX Table V")
print("=" * 130)

print(r"\begin{table*}[t]")
print(r"\centering")
print(
    r"\caption{Per-project percentage of mutants exhibiting stable behavior or at least one status transition "
    r"within each code region. Stable combines mutants classified as Stable $S$ or Stable $K$; "
    r"Status Transition combines mutants classified as $S{\to}K$, $K{\to}S$, "
    r"$S{\to}K{\to}S$, $K{\to}S{\to}K$, or Recurrent. "
    r"Percentages are computed independently for each project-region using the number of classified mutants "
    r"assigned to that region as the denominator. The pooled row aggregates all mutants across projects, "
    r"whereas the project mean and median weight projects equally.}"
)
print(r"\label{tab:rq2_per_project_stable_transition_region}")
print(r"\resizebox{\linewidth}{!}{")
print(r"\begin{tabular}{lrrrrrrrr}")
print(r"\toprule")
print(r"\multirow{2}{*}{Project} & \multicolumn{4}{c}{Stable} & \multicolumn{4}{c}{Status Transition} \\")
print(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
print(r"& r1 & r2 & r3 & r4 & r1 & r2 & r3 & r4 \\")
print(r"\midrule")

for pid in projects:
    stable_cells = [
        latex_pct(project_group_matrix[pid]["Stable"].get(region))
        for region in REGIONS
    ]
    trans_cells = [
        latex_pct(project_group_matrix[pid]["Status Transition"].get(region))
        for region in REGIONS
    ]

    print(
        f"{latex_escape(pid)} & "
        f"{stable_cells[0]} & {stable_cells[1]} & {stable_cells[2]} & {stable_cells[3]} & "
        f"{trans_cells[0]} & {trans_cells[1]} & {trans_cells[2]} & {trans_cells[3]} \\\\"
    )

print(r"\midrule")

for row_name, summary_index in [
    ("Overall pooled", 0),
    ("Project mean", 1),
    ("Project median", 2),
]:
    stable_cells = [latex_pct(x) for x in group_summary["Stable"][summary_index]]
    trans_cells = [latex_pct(x) for x in group_summary["Status Transition"][summary_index]]

    print(
        f"{row_name} & "
        f"{stable_cells[0]} & {stable_cells[1]} & {stable_cells[2]} & {stable_cells[3]} & "
        f"{trans_cells[0]} & {trans_cells[1]} & {trans_cells[2]} & {trans_cells[3]} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"}")
print(r"\end{table*}")


# ============================================================
# LATEX TABLE VI
# ============================================================

print("\n\nLaTeX Table VI")
print("=" * 130)

cat_latex_names = {
    "Stable S": r"Stable $S$",
    "Stable K": r"Stable $K$",
    "S->K": r"$S{\to}K$",
    "K->S": r"$K{\to}S$",
    "S->K->S": r"$S{\to}K{\to}S$",
    "K->S->K": r"$K{\to}S{\to}K$",
    "Recurrent": r"Recur.",
}

print(r"\begin{table*}[t]")
print(r"\centering")
print(
    r"\caption{Per-project percentage of mutants exhibiting each longitudinal status category within each code region. "
    r"Percentages are computed independently for each project-region using the number of classified mutants assigned "
    r"to that region as the denominator; therefore, values across categories within the same project-region sum to "
    r"100\%, except for rounding. The pooled row aggregates all mutants across projects, whereas the project mean "
    r"and median weight projects equally.}"
)
print(r"\label{tab:rq2_per_project_detailed_status_region}")
print(r"\scriptsize")
print(r"\setlength{\tabcolsep}{2pt}")
print(r"\resizebox{\linewidth}{!}{")
print(r"\begin{tabular}{l" + "rrrr" * len(CATS) + r"}")
print(r"\toprule")

top = r"\multirow{2}{*}{Project}"
col_start = 2
cmidrules = []

for cat in CATS:
    col_end = col_start + 3
    top += rf" & \multicolumn{{4}}{{c}}{{{cat_latex_names[cat]}}}"
    cmidrules.append(rf"\cmidrule(lr){{{col_start}-{col_end}}}")
    col_start = col_end + 1

top += r" \\"
print(top)
print("".join(cmidrules))

second = ""
for _cat in CATS:
    second += r" & r1 & r2 & r3 & r4"
second += r" \\"
print(second)

print(r"\midrule")

for pid in projects:
    row = latex_escape(pid)

    for cat in CATS:
        cells = [
            latex_pct(project_cat_matrix[pid][cat].get(region))
            for region in REGIONS
        ]
        row += f" & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]}"

    row += r" \\"
    print(row)

print(r"\midrule")

for row_name, summary_index in [
    ("Overall pooled", 0),
    ("Project mean", 1),
    ("Project median", 2),
]:
    row = row_name

    for cat in CATS:
        cells = [latex_pct(x) for x in cat_summary[cat][summary_index]]
        row += f" & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]}"

    row += r" \\"
    print(row)

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"}")
print(r"\end{table*}")


# ============================================================
# SANITY CHECKS
# ============================================================

max_stable_transition_gap = 0.0

for pid in projects:
    for region in REGIONS:
        s = project_group_matrix[pid]["Stable"].get(region)
        t = project_group_matrix[pid]["Status Transition"].get(region)

        if s is not None and t is not None:
            max_stable_transition_gap = max(
                max_stable_transition_gap,
                abs((s + t) - 100.0)
            )

max_detailed_sum_gap = 0.0

for pid in projects:
    for region in REGIONS:
        vals = [
            project_cat_matrix[pid][cat].get(region)
            for cat in CATS
        ]
        vals = [v for v in vals if v is not None]

        if vals:
            max_detailed_sum_gap = max(
                max_detailed_sum_gap,
                abs(sum(vals) - 100.0)
            )


print("\n\nSANITY")
print("=" * 80)
print("classified mutants:", total_classified)
print("noisy / unclassified mutants:", total_noisy)
print("elements:", len(element_rows))
print("transition / non-stable-pattern mutants:", len(debug_transition_elements))
print(
    "max Stable+Transition deviation from 100% in non-empty project-regions:",
    f"{max_stable_transition_gap:.8f}",
)
print(
    "max detailed-category sum deviation from 100% in non-empty project-regions:",
    f"{max_detailed_sum_gap:.8f}",
)

print("\nTop transition-containing elements")
print("=" * 120)

top = sorted(
    [r for r in element_rows if r["AnyTransition"] > 0],
    key=lambda r: (r["AnyTransition"], r["AnyTransition_density"]),
    reverse=True,
)[:30]

for r in top:
    print(
        f"{r['Project']:<14} "
        f"{r['Instance']:<18} "
        f"{r['Region']:<22} "
        f"N={r['N']:<4} "
        f"T={r['AnyTransition']:<4} "
        f"Tdens={r['AnyTransition_density']:.1f}% "
        f"{r['Element']}"
    )