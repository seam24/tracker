import json
import re
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from statistics import median


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("D:/data_from_hpc")

# Expected repo path:
# C:/Users/tahar/Desktop/shared/LatentMu/commits2/black-16b/black
REPO_ROOT_TEMPLATE = Path(
    "C:/Users/tahar/Desktop/shared/LatentMu/commits2/{instance}/{pid}"
)

OUT_DIR = Path("D:/res/rq3_commit_pair_context")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRE_PROGRAM_EXITSTATUS_0_OR_1 = True
REMOVE_UNINFORMATIVE_COMMITS = True

# None = all projects under DATA_ROOT
# Or for table only:
# SELECTED_PROJECTS = {"black", "youtube-dl", "tqdm"}
SELECTED_PROJECTS = None


# ============================================================
# BASIC HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def git(repo_dir, args, quiet=True):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir)] + args,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL if quiet else None,
        )
    except subprocess.CalledProcessError:
        return None


def result_commit_id(path, data):
    commit = data.get("program", {}).get("meta", {}).get("commit")
    if commit:
        return commit

    name = path.name
    if name.startswith("results_") and name.endswith(".json"):
        return name[len("results_"):-len(".json")]

    return None


def is_usable_result(data):
    exitstatus = data.get("program", {}).get("exitstatus")
    if REQUIRE_PROGRAM_EXITSTATUS_0_OR_1:
        return exitstatus in (0, 1)
    return True


def mutant_status_from_tests(minfo):
    counts = Counter()

    for _, tinfo in minfo.get("tests", {}).items():
        st = (tinfo.get("status") or "unknown").lower().strip()
        counts[st] += 1

    if counts["killed"] > 0:
        return "K"
    if counts["survived"] > 0:
        return "S"
    if counts["anomalous"] > 0:
        return "A"
    if counts["incompetent"] > 0:
        return "I"
    return "U"


def status_label(mu_hist, mid, commit):
    status = mu_hist[mid]["dict_history"][commit].get("status")

    if status is None:
        raise KeyError(f"No status for mutant={mid}, commit={commit}")

    if isinstance(status, str):
        return status

    if isinstance(status, dict):
        return status.get("label", "U")

    if isinstance(status, (list, tuple)) and len(status) >= 1:
        return "K" if status[0] > 0 else "S"

    return "U"


def pct(x, denom):
    return 0.0 if denom == 0 else 100.0 * x / denom


def fmt_pct(x, denom):
    return f"{pct(x, denom):.1f}"


def fmt_num(x):
    if x is None:
        return "--"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def quantile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]

    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def fmt_median_iqr(vals):
    vals = sorted(vals)
    if not vals:
        return "--"

    m = median(vals)
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)

    return f"{fmt_num(m)} [{fmt_num(q1)}, {fmt_num(q3)}]"


# ============================================================
# DISCOVERY
# ============================================================
def discover_instances():
    """
    Finds folders like:
      black-16b
      tqdm-5b
      youtube-dl-10b

    Project name is everything before the final -<number>b.
    """
    pattern = re.compile(r"^(.+)-(\d+)b$")

    instances = []
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir():
            continue

        m = pattern.match(path.name)
        if not m:
            continue

        pid = m.group(1)
        bid = m.group(2)

        if SELECTED_PROJECTS is not None and pid not in SELECTED_PROJECTS:
            continue

        instances.append({
            "pid": pid,
            "bid": bid,
            "instance": path.name,
            "base": path,
        })

    return instances


# ============================================================
# CHURN / COMMIT MAGNITUDE
# ============================================================
def normalize_path(p):
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def is_test_path(path):
    p = normalize_path(path).lower()
    name = p.rsplit("/", 1)[-1]

    return (
        p.startswith("test/")
        or p.startswith("tests/")
        or "/test/" in p
        or "/tests/" in p
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def is_python_file(path):
    return normalize_path(path).lower().endswith(".py")


def parse_numstat_num(x):
    if x == "-":
        return 0
    try:
        return int(x)
    except Exception:
        return 0


def commit_magnitude(repo_dir, commit_before, commit_after):
    """
    Table VI needs:
      Size = all line churn in the commit pair
      Code = line churn in production Python files
      Test = line churn in test Python files

    Change type:
      code-only
      test-only
      code+test
      no-code/test
    """
    raw = git(repo_dir, ["diff", "--numstat", commit_before, commit_after], quiet=True)

    if raw is None:
        return {
            "size": None,
            "code": None,
            "test": None,
            "change_type": "git-error",
        }

    size = 0
    code = 0
    test = 0

    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        added = parse_numstat_num(parts[0])
        deleted = parse_numstat_num(parts[1])
        path = normalize_path(parts[-1])
        churn = added + deleted

        size += churn

        if is_python_file(path):
            if is_test_path(path):
                test += churn
            else:
                code += churn

    has_code = code > 0
    has_test = test > 0

    if has_code and has_test:
        change_type = "code+test"
    elif has_code:
        change_type = "code-only"
    elif has_test:
        change_type = "test-only"
    else:
        change_type = "no-code/test"

    return {
        "size": size,
        "code": code,
        "test": test,
        "change_type": change_type,
    }


# ============================================================
# INSTANCE LOADING
# ============================================================
def load_instance(pid, bid, base):
    with_lbl_dir = base / "with_lbl"
    commits_path = base / "commits_to_run.txt"
    mutant_history_path = base / "mutant_history.json"

    if not with_lbl_dir.exists():
        raise RuntimeError(f"Missing with_lbl: {with_lbl_dir}")
    if not commits_path.exists():
        raise RuntimeError(f"Missing commits_to_run.txt: {commits_path}")
    if not mutant_history_path.exists():
        raise RuntimeError(f"Missing mutant_history.json: {mutant_history_path}")

    mu_hist = load_json(mutant_history_path)

    for mid in mu_hist:
        mu_hist[mid]["dict_history"] = {
            item["commit"]: item
            for item in mu_hist[mid].get("history", [])
        }

    commits_newest_to_oldest = [
        line.strip()
        for line in commits_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]

    commits_oldest_to_newest = list(reversed(commits_newest_to_oldest))

    results_by_commit = {}

    for commit in commits_newest_to_oldest:
        path = with_lbl_dir / f"results_{commit}.json"
        if not path.exists():
            continue

        data = load_json(path)
        real_commit = result_commit_id(path, data)

        if real_commit is None:
            continue
        if not is_usable_result(data):
            continue

        results_by_commit[real_commit] = data

    for commit, data in results_by_commit.items():
        for mid, minfo in data.get("mutants", {}).items():
            if mid not in mu_hist:
                continue
            if commit not in mu_hist[mid]["dict_history"]:
                continue

            mu_hist[mid]["dict_history"][commit]["status"] = mutant_status_from_tests(minfo)

    ordered_result_commits = [
        c for c in commits_oldest_to_newest
        if c in results_by_commit
    ]

    if REMOVE_UNINFORMATIVE_COMMITS:
        filtered = {}

        for commit in ordered_result_commits:
            labels = []

            for mid in results_by_commit[commit].get("mutants", {}):
                try:
                    labels.append(status_label(mu_hist, mid, commit))
                except Exception:
                    continue

            if not labels:
                continue

            c = Counter(labels)

            if c["K"] == len(labels):
                continue
            if c["S"] == len(labels):
                continue
            if c["A"] == len(labels):
                continue
            if c["I"] == len(labels):
                continue
            if c["U"] == len(labels):
                continue
            if c["K"] + c["S"] == 0:
                continue

            filtered[commit] = results_by_commit[commit]

        results_by_commit = filtered

    ordered_commits = [
        c for c in commits_oldest_to_newest
        if c in results_by_commit
    ]

    return mu_hist, results_by_commit, ordered_commits


# ============================================================
# COMMIT-PAIR CLASSIFICATION
# ============================================================
def classify_commit_pair(mu_hist, results_by_commit, commit_before, commit_after):
    """
    A stable pair:
      adjacent pair with no S->K or K->S transition among categorized mutants.

    A transition pair:
      adjacent pair with at least one S->K or K->S event.

    Only-S->K, only-K->S, mixed are mutually exclusive subgroups.
    """
    before_mutants = set(results_by_commit[commit_before].get("mutants", {}))
    after_mutants = set(results_by_commit[commit_after].get("mutants", {}))
    common = before_mutants & after_mutants

    counts = Counter()

    for mid in common:
        if mid not in mu_hist:
            continue

        try:
            st_before = status_label(mu_hist, mid, commit_before)
            st_after = status_label(mu_hist, mid, commit_after)
        except Exception:
            continue

        if st_before not in {"K", "S"}:
            continue
        if st_after not in {"K", "S"}:
            continue

        counts[f"{st_before}->{st_after}"] += 1

    sk = counts["S->K"]
    ks = counts["K->S"]

    if sk == 0 and ks == 0:
        group = "All stable"
    elif sk > 0 and ks == 0:
        group = "Only S -> K"
    elif sk == 0 and ks > 0:
        group = "Only K -> S"
    else:
        group = "Mixed transition"

    return {
        "group": group,
        "is_transition": sk > 0 or ks > 0,
        "S_to_S": counts["S->S"],
        "K_to_K": counts["K->K"],
        "S_to_K": sk,
        "K_to_S": ks,
        "categorized_mutant_pairs": sum(counts.values()),
    }


def analyze_instance(info):
    pid = info["pid"]
    bid = info["bid"]
    instance = info["instance"]
    base = info["base"]

    repo_dir = Path(str(REPO_ROOT_TEMPLATE).format(
        pid=pid,
        bid=bid,
        instance=instance,
    ))

    if not repo_dir.exists():
        raise RuntimeError(f"Repo does not exist: {repo_dir}")

    mu_hist, results_by_commit, ordered_commits = load_instance(pid, bid, base)

    pair_rows = []

    for commit_before, commit_after in zip(ordered_commits, ordered_commits[1:]):
        pair_info = classify_commit_pair(
            mu_hist=mu_hist,
            results_by_commit=results_by_commit,
            commit_before=commit_before,
            commit_after=commit_after,
        )

        if pair_info["categorized_mutant_pairs"] == 0:
            continue

        mag = commit_magnitude(repo_dir, commit_before, commit_after)

        row = {
            "pid": pid,
            "bid": bid,
            "instance": instance,
            "commit_before": commit_before,
            "commit_after": commit_after,
            **pair_info,
            **mag,
        }

        pair_rows.append(row)

    return pair_rows


# ============================================================
# AGGREGATION
# ============================================================
GROUP_ORDER = [
    "All stable",
    "All transition",
    "Only S -> K",
    "Only K -> S",
    "Mixed transition",
]

CHANGE_TYPES = [
    "code-only",
    "test-only",
    "code+test",
    "no-code/test",
]


def empty_agg():
    return {
        "pairs": 0,
        "size_values": [],
        "code_values": [],
        "test_values": [],
        "change_type_counts": Counter(),
        "S_to_K_events": 0,
        "K_to_S_events": 0,
        "S_to_S_events": 0,
        "K_to_K_events": 0,
    }


def add_to_agg(agg, row):
    agg["pairs"] += 1

    if row["size"] is not None:
        agg["size_values"].append(row["size"])
        agg["code_values"].append(row["code"])
        agg["test_values"].append(row["test"])

    agg["change_type_counts"][row["change_type"]] += 1

    agg["S_to_K_events"] += row["S_to_K"]
    agg["K_to_S_events"] += row["K_to_S"]
    agg["S_to_S_events"] += row["S_to_S"]
    agg["K_to_K_events"] += row["K_to_K"]


def build_aggregates(pair_rows):
    by_project = defaultdict(lambda: defaultdict(empty_agg))
    total = defaultdict(empty_agg)

    for row in pair_rows:
        pid = row["pid"]

        add_to_agg(by_project[pid][row["group"]], row)
        add_to_agg(total[row["group"]], row)

        if row["is_transition"]:
            add_to_agg(by_project[pid]["All transition"], row)
            add_to_agg(total["All transition"], row)

    return by_project, total


def agg_to_table_row(project, group, agg):
    n = agg["pairs"]
    c = agg["change_type_counts"]

    return {
        "Project": project,
        "Commit-pair group": group,
        "# Pairs": n,
        "Size": fmt_median_iqr(agg["size_values"]),
        "Code": fmt_median_iqr(agg["code_values"]),
        "Test": fmt_median_iqr(agg["test_values"]),
        "Code-only": fmt_pct(c["code-only"], n),
        "Test-only": fmt_pct(c["test-only"], n),
        "Code+test": fmt_pct(c["code+test"], n),
        "No code/test": fmt_pct(c["no-code/test"], n),
    }


# ============================================================
# OUTPUT
# ============================================================
def print_markdown_table(rows):
    headers = [
        "Project",
        "Commit-pair group",
        "# Pairs",
        "Size",
        "Code",
        "Test",
        "Code-only",
        "Test-only",
        "Code+test",
        "No code/test",
    ]

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")

    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in headers) + " |")


def write_csv(rows, path):
    headers = [
        "Project",
        "Commit-pair group",
        "# Pairs",
        "Size",
        "Code",
        "Test",
        "Code-only",
        "Test-only",
        "Code+test",
        "No code/test",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = [str(r[h]).replace(",", ";") for h in headers]
            f.write(",".join(vals) + "\n")


def latex_escape(x):
    return str(x).replace("_", "\\_")


def write_latex_table(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrlllrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Project & Commit-pair group & \\# Pairs "
            "& Size & Code & Test "
            "& Code-only & Test-only & Code+test & No code/test \\\\\n"
        )
        f.write("\\midrule\n")

        last_project = None

        for r in rows:
            project = r["Project"]
            shown_project = project if project != last_project else ""
            last_project = project

            f.write(
                f"{latex_escape(shown_project)} & "
                f"{latex_escape(r['Commit-pair group'])} & "
                f"{r['# Pairs']} & "
                f"{r['Size']} & "
                f"{r['Code']} & "
                f"{r['Test']} & "
                f"{r['Code-only']} & "
                f"{r['Test-only']} & "
                f"{r['Code+test']} & "
                f"{r['No code/test']} \\\\\n"
            )

            if r["Commit-pair group"] == "Mixed transition":
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    instances = discover_instances()

    if not instances:
        raise RuntimeError(f"No instances found under {DATA_ROOT}")

    all_pair_rows = []
    skipped = []

    for info in instances:
        label = info["instance"]

        try:
            rows = analyze_instance(info)
            all_pair_rows.extend(rows)
            print(f"[OK] {label}: {len(rows)} commit pairs")

        except Exception as exc:
            skipped.append({
                "instance": label,
                "reason": str(exc),
            })
            print(f"[SKIP] {label}: {exc}")

    by_project, total = build_aggregates(all_pair_rows)

    table_rows = []

    for pid in sorted(by_project):
        for group in GROUP_ORDER:
            table_rows.append(
                agg_to_table_row(pid, group, by_project[pid][group])
            )

    for group in GROUP_ORDER:
        table_rows.append(
            agg_to_table_row("Total", group, total[group])
        )

    print()
    print_markdown_table(table_rows)

    summary_csv = OUT_DIR / "rq3_commit_pair_context_summary.csv"
    summary_tex = OUT_DIR / "rq3_commit_pair_context_summary.tex"
    pair_json = OUT_DIR / "rq3_commit_pair_context_pairs.json"
    skipped_json = OUT_DIR / "rq3_commit_pair_context_skipped.json"

    write_csv(table_rows, summary_csv)
    write_latex_table(table_rows, summary_tex)

    with open(pair_json, "w", encoding="utf-8") as f:
        json.dump(all_pair_rows, f, indent=2, ensure_ascii=False)

    with open(skipped_json, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_tex}")
    print(f"Wrote {pair_json}")
    print(f"Wrote {skipped_json}")

    if skipped:
        print()
        print("Skipped instances:")
        for s in skipped:
            print(f"- {s['instance']}: {s['reason']}")


if __name__ == "__main__":
    main()