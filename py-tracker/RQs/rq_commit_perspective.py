import json
import subprocess
import statistics
import math
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================
pid = "tqdm"

DATA_ROOT = Path("D://data_from_hpc")
REPO_ROOT = Path("commits2 path here")

REQUIRE_PROGRAM_EXITSTATUS_0_OR_1 = True
REMOVE_UNINFORMATIVE_COMMITS = True
PRINT_INSTANCE_DETAILS = True


# ============================================================
# BASIC HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def short(c):
    return c[:12]


def bid_from_instance_dir(path):
    name = path.name
    prefix = f"{pid}-"
    suffix = "b"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return None


def pct(num, den):
    return 100 * num / den if den else 0.0


def median_iqr(values):
    values = sorted([v for v in values if v is not None])

    if not values:
        return "NA"

    med = statistics.median(values)

    if len(values) < 4:
        return f"{med:.1f} [NA, NA]"

    q1 = statistics.quantiles(values, n=4, method="inclusive")[0]
    q3 = statistics.quantiles(values, n=4, method="inclusive")[2]
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def cliffs_delta(xs, ys):
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]

    if not xs or not ys:
        return None

    gt = 0
    lt = 0

    for x in xs:
        for y in ys:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1

    return (gt - lt) / (len(xs) * len(ys))


def cliffs_label(delta):
    if delta is None:
        return "NA"

    ad = abs(delta)

    if ad < 0.147:
        label = "negligible"
    elif ad < 0.33:
        label = "small"
    elif ad < 0.474:
        label = "medium"
    else:
        label = "large"

    return f"{delta:.3f} ({label})"


def odds_ratio(a, b, c, d):
    """
    Table:
                    feature yes   feature no
    transition yes      a            b
    transition no       c            d

    Uses Haldane-Anscombe correction if any cell is zero.
    """
    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    return (a * d) / (b * c) if b * c else float("inf")


def fmt_or(x):
    if x == float("inf"):
        return "inf"
    return f"{x:.2f}"


# ============================================================
# PROJECT / TEST HELPERS
# ============================================================
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
    elif counts["survived"] > 0:
        return "S"
    elif counts["anomalous"] > 0:
        return "A"
    elif counts["incompetent"] > 0:
        return "I"
    else:
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


def is_test_file(path_str):
    p = path_str.replace("\\", "/").lower()
    name = Path(p).name.lower()

    return (
        "/test/" in p
        or "/tests/" in p
        or p.startswith("test/")
        or p.startswith("tests/")
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("test.py")
    )


def is_code_file(path_str):
    p = path_str.replace("\\", "/").lower()
    return p.endswith(".py") and not is_test_file(p)


def run_git(repo_path, args):
    result = subprocess.run(
        ["git", "-C", str(repo_path)] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Git failed:\n"
            f"git -C {repo_path} {' '.join(args)}\n"
            f"{result.stderr}"
        )

    return result.stdout


# ============================================================
# COMMIT FEATURES
# ============================================================
def has_assertion_change(repo_path, prev_commit, curr_commit):
    diff = run_git(repo_path, [
        "diff",
        "--unified=0",
        "--ignore-space-change",
        prev_commit,
        curr_commit,
        "--",
        "*.py",
    ])

    markers = [
        "assert ",
        "assert(",
        "pytest.raises",
        "self.assert",
        ".assert",
        "unittest.",
        "raises(",
        "equal(",
        "equals(",
    ]

    current_file_is_test = False

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file_is_test = is_test_file(line[len("+++ b/"):])
            continue

        if not current_file_is_test:
            continue

        if line.startswith("+") or line.startswith("-"):
            if line.startswith("+++") or line.startswith("---"):
                continue

            body = line[1:].strip().lower()

            if any(m in body for m in markers):
                return True

    return False


def get_commit_features(repo_path, prev_commit, curr_commit):
    numstat = run_git(repo_path, [
        "diff",
        "--numstat",
        "--find-renames",
        "--find-copies",
        "--ignore-space-change",
        prev_commit,
        curr_commit,
    ])

    name_status = run_git(repo_path, [
        "diff",
        "--name-status",
        "--find-renames",
        "--find-copies",
        prev_commit,
        curr_commit,
    ])

    test_added = 0
    test_deleted = 0
    code_added = 0
    code_deleted = 0
    other_added = 0
    other_deleted = 0

    test_files_changed = 0
    code_files_changed = 0
    other_files_changed = 0
    total_files_changed = 0

    added_test_files = 0
    added_code_files = 0

    for line in numstat.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        added_raw, deleted_raw, file_path = parts[0], parts[1], parts[2]
        total_files_changed += 1

        if added_raw == "-" or deleted_raw == "-":
            added = 0
            deleted = 0
        else:
            added = int(added_raw)
            deleted = int(deleted_raw)

        if is_test_file(file_path):
            test_files_changed += 1
            test_added += added
            test_deleted += deleted
        elif is_code_file(file_path):
            code_files_changed += 1
            code_added += added
            code_deleted += deleted
        else:
            other_files_changed += 1
            other_added += added
            other_deleted += deleted

    for line in name_status.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        status = parts[0]

        if status.startswith("A") and len(parts) >= 2:
            file_path = parts[1]

            if is_test_file(file_path):
                added_test_files += 1
            elif is_code_file(file_path):
                added_code_files += 1

    test_churn = test_added + test_deleted
    code_churn = code_added + code_deleted
    other_churn = other_added + other_deleted

    return {
        "total_files_changed": total_files_changed,
        "test_files_changed": test_files_changed,
        "code_files_changed": code_files_changed,
        "other_files_changed": other_files_changed,

        "test_added": test_added,
        "test_deleted": test_deleted,
        "test_churn": test_churn,
        "test_net_change": test_added - test_deleted,

        "code_added": code_added,
        "code_deleted": code_deleted,
        "code_churn": code_churn,
        "code_net_change": code_added - code_deleted,

        "other_churn": other_churn,

        "has_test_change": test_churn > 0,
        "has_code_change": code_churn > 0,

        "is_test_only": test_churn > 0 and code_churn == 0,
        "is_code_only": code_churn > 0 and test_churn == 0,
        "is_mixed_test_code": test_churn > 0 and code_churn > 0,

        "added_test_files": added_test_files,
        "added_code_files": added_code_files,
        "has_added_test_file": added_test_files > 0,
        "has_added_code_file": added_code_files > 0,

        "has_assertion_change": has_assertion_change(repo_path, prev_commit, curr_commit),
    }


# ============================================================
# PROCESS ONE INSTANCE
# ============================================================
def process_instance(instance_dir):
    bid = bid_from_instance_dir(instance_dir)

    with_lbl_dir = instance_dir / "with_lbl"
    commits_path = instance_dir / "commits_to_run.txt"
    mutant_history_path = instance_dir / "mutant_history.json"
    repo_path = REPO_ROOT / f"{pid}-{bid}b" / pid

    if not with_lbl_dir.exists() or not commits_path.exists() or not mutant_history_path.exists():
        print(f"[SKIP] Missing files for {instance_dir}")
        return []

    if not repo_path.exists():
        print(f"[SKIP] Missing repo: {repo_path}")
        return []

    mu_hist = load_json(mutant_history_path)

    for mid in mu_hist:
        mu_hist[mid]["dict_history"] = {
            item["commit"]: item
            for item in mu_hist[mid].get("history", [])
        }

    with open(commits_path, "r", encoding="utf-8", errors="replace") as f:
        commits_newest_to_oldest = [line.strip() for line in f if line.strip()]

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
        filtered_results_by_commit = {}

        for commit in ordered_result_commits:
            labels = []

            for mid in results_by_commit[commit].get("mutants", {}):
                try:
                    labels.append(status_label(mu_hist, mid, commit))
                except Exception:
                    continue

            if not labels:
                continue

            counter = Counter(labels)

            if counter["K"] == len(labels):
                continue
            if counter["S"] == len(labels):
                continue
            if counter["A"] == len(labels):
                continue
            if counter["I"] == len(labels):
                continue
            if counter["U"] == len(labels):
                continue

            if counter["K"] + counter["S"] == 0:
                continue

            filtered_results_by_commit[commit] = results_by_commit[commit]

        results_by_commit = filtered_results_by_commit

    ordered_commits = [
        c for c in commits_oldest_to_newest
        if c in results_by_commit
    ]

    commit_features = {}

    for i in range(len(ordered_commits) - 1):
        prev_commit = ordered_commits[i]
        curr_commit = ordered_commits[i + 1]

        try:
            commit_features[(prev_commit, curr_commit)] = get_commit_features(
                repo_path,
                prev_commit,
                curr_commit,
            )
        except Exception as e:
            print(f"[WARN] Git feature failed for {instance_dir.name}: {short(prev_commit)} -> {short(curr_commit)}")
            print(e)

    commit_transition_counts = defaultdict(Counter)

    for mid, minfo in mu_hist.items():
        observations = []

        for commit in ordered_commits:
            if mid not in results_by_commit[commit].get("mutants", {}):
                continue

            try:
                st = status_label(mu_hist, mid, commit)
            except Exception:
                continue

            if st in {"K", "S"}:
                observations.append((commit, st))

        for i in range(len(observations) - 1):
            prev_commit, prev_status = observations[i]
            curr_commit, curr_status = observations[i + 1]

            if (prev_commit, curr_commit) not in commit_features:
                continue

            transition = f"{prev_status}->{curr_status}"
            commit_transition_counts[(prev_commit, curr_commit)][transition] += 1

    rows = []

    for (prev_commit, curr_commit), feat in commit_features.items():
        tc = commit_transition_counts[(prev_commit, curr_commit)]

        s_to_k = tc["S->K"]
        k_to_s = tc["K->S"]
        s_to_s = tc["S->S"]
        k_to_k = tc["K->K"]

        changing = s_to_k + k_to_s
        stable = s_to_s + k_to_k
        total = changing + stable

        rows.append({
            "project": pid,
            "bug": bid,
            "instance": instance_dir.name,
            "prev_commit": prev_commit,
            "curr_commit": curr_commit,

            "S_to_K": s_to_k,
            "K_to_S": k_to_s,
            "S_to_S": s_to_s,
            "K_to_K": k_to_k,

            "changing_transitions": changing,
            "stable_transitions": stable,
            "total_transitions": total,

            "has_S_to_K": s_to_k > 0,
            "has_K_to_S": k_to_s > 0,
            "has_any_change": changing > 0,

            **feat,
        })

    if PRINT_INSTANCE_DETAILS:
        print(
            f"{instance_dir.name}: commit_pairs={len(rows)}, "
            f"S->K={sum(r['S_to_K'] for r in rows)}, "
            f"K->S={sum(r['K_to_S'] for r in rows)}"
        )

    return rows


# ============================================================
# PROCESS ALL INSTANCES
# ============================================================
instance_dirs = sorted(DATA_ROOT.glob(f"{pid}-*b"))

all_rows = []

for instance_dir in instance_dirs:
    if bid_from_instance_dir(instance_dir) is None:
        continue

    all_rows.extend(process_instance(instance_dir))

transition_rows = [r for r in all_rows if r["has_any_change"]]
stable_rows = [r for r in all_rows if not r["has_any_change"]]

s_to_k_rows = [r for r in all_rows if r["has_S_to_K"]]
k_to_s_rows = [r for r in all_rows if r["has_K_to_S"]]

print("\nTotal commit-pair rows:", len(all_rows))
print("Transition commit rows:", len(transition_rows))
print("Stable commit rows:", len(stable_rows))


# ============================================================
# TABLE 1: TRANSITION VS STABLE NUMERIC CHARACTERISTICS
# ============================================================
numeric_metrics = [
    ("Test churn", "test_churn"),
    ("Code churn", "code_churn"),
    ("Test net", "test_net_change"),
    ("Code net", "code_net_change"),
    ("Test files", "test_files_changed"),
    ("Code files", "code_files_changed"),
    ("Total files", "total_files_changed"),
]

print("\n=== TABLE 1: Numeric commit characteristics: transition vs stable ===")
print("Project\tMetric\tTransition commits\tStable commits\tCliff's delta")

table1 = []

for label, key in numeric_metrics:
    xs = [r[key] for r in transition_rows]
    ys = [r[key] for r in stable_rows]
    d = cliffs_delta(xs, ys)

    row = {
        "project": pid,
        "metric": label,
        "transition": median_iqr(xs),
        "stable": median_iqr(ys),
        "cliffs": cliffs_label(d),
    }
    table1.append(row)

    print(
        f"{row['project']}\t{row['metric']}\t"
        f"{row['transition']}\t{row['stable']}\t{row['cliffs']}"
    )


# ============================================================
# TABLE 2: BOOLEAN FEATURES AND ODDS RATIOS
# ============================================================
boolean_features = [
    ("Test changed", "has_test_change"),
    ("Code changed", "has_code_change"),
    ("Mixed code+test", "is_mixed_test_code"),
    ("Code-only", "is_code_only"),
    ("Added test file", "has_added_test_file"),
    ("Assertion changed", "has_assertion_change"),
    ("Added code file", "has_added_code_file"),
]

print("\n=== TABLE 2: Boolean commit features: transition vs stable ===")
print(
    "Project\tFeature\t"
    "Transition yes\tTransition no\tStable yes\tStable no\t"
    "% transition given feature\t% transition given no feature\tOdds ratio"
)

table2 = []

for label, key in boolean_features:
    a = sum(1 for r in all_rows if r["has_any_change"] and r[key])
    b = sum(1 for r in all_rows if r["has_any_change"] and not r[key])
    c = sum(1 for r in all_rows if (not r["has_any_change"]) and r[key])
    d = sum(1 for r in all_rows if (not r["has_any_change"]) and not r[key])

    feature_yes_total = a + c
    feature_no_total = b + d

    pct_feature_yes = pct(a, feature_yes_total)
    pct_feature_no = pct(b, feature_no_total)

    or_value = odds_ratio(a, b, c, d)

    row = {
        "project": pid,
        "feature": label,
        "transition_yes": a,
        "transition_no": b,
        "stable_yes": c,
        "stable_no": d,
        "pct_transition_feature_yes": pct_feature_yes,
        "pct_transition_feature_no": pct_feature_no,
        "odds_ratio": or_value,
    }
    table2.append(row)

    print(
        f"{row['project']}\t{row['feature']}\t"
        f"{a}\t{b}\t{c}\t{d}\t"
        f"{pct_feature_yes:.1f}\t{pct_feature_no:.1f}\t{fmt_or(or_value)}"
    )


# ============================================================
# TABLE 3: S->K VS STABLE NUMERIC CHARACTERISTICS
# ============================================================
print("\n=== TABLE 3: Numeric commit characteristics: S->K vs no S->K ===")
print("Project\tMetric\tS->K commits\tNo S->K commits\tCliff's delta")

no_s_to_k_rows = [r for r in all_rows if not r["has_S_to_K"]]

for label, key in numeric_metrics:
    xs = [r[key] for r in s_to_k_rows]
    ys = [r[key] for r in no_s_to_k_rows]
    d = cliffs_delta(xs, ys)

    print(
        f"{pid}\t{label}\t"
        f"{median_iqr(xs)}\t{median_iqr(ys)}\t{cliffs_label(d)}"
    )


# ============================================================
# TABLE 4: K->S VS STABLE NUMERIC CHARACTERISTICS
# ============================================================
print("\n=== TABLE 4: Numeric commit characteristics: K->S vs no K->S ===")
print("Project\tMetric\tK->S commits\tNo K->S commits\tCliff's delta")

no_k_to_s_rows = [r for r in all_rows if not r["has_K_to_S"]]

for label, key in numeric_metrics:
    xs = [r[key] for r in k_to_s_rows]
    ys = [r[key] for r in no_k_to_s_rows]
    d = cliffs_delta(xs, ys)

    print(
        f"{pid}\t{label}\t"
        f"{median_iqr(xs)}\t{median_iqr(ys)}\t{cliffs_label(d)}"
    )


# ============================================================
# LATEX: TABLE 1
# ============================================================
print("\n=== LATEX TABLE 1 ===")
print(r"\begin{table*}[t]")
print(r"\centering")
print(r"\caption{Commit-level characteristics of transition and stable commits. Churn is computed as added plus deleted lines; net change is added minus deleted lines.}")
print(r"\label{tab:transition_vs_stable_commit_metrics}")
print(r"\small")
print(r"\begin{tabular}{lllll}")
print(r"\toprule")
print(r"Project & Metric & Transition commits & Stable commits & Cliff's $\delta$ \\")
print(r"\midrule")

for r in table1:
    print(
        f"{r['project']} & {r['metric']} & {r['transition']} & "
        f"{r['stable']} & {r['cliffs']} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")


# ============================================================
# LATEX: TABLE 2
# ============================================================
print("\n=== LATEX TABLE 2 ===")
print(r"\begin{table*}[t]")
print(r"\centering")
print(r"\caption{Association between commit features and mutant-status transitions. Odds ratios above 1 indicate that commits with the feature are more likely to induce at least one mutant-status transition.}")
print(r"\label{tab:commit_feature_transition_odds}")
print(r"\small")
print(r"\begin{tabular}{llrrrrr}")
print(r"\toprule")
print(
    r"Project & Feature & "
    r"\% Trans. with feature & \% Trans. without feature & "
    r"Trans. yes/no & Stable yes/no & Odds ratio \\"
)
print(r"\midrule")

for r in table2:
    print(
        f"{r['project']} & {r['feature']} & "
        f"{r['pct_transition_feature_yes']:.1f} & "
        f"{r['pct_transition_feature_no']:.1f} & "
        f"{r['transition_yes']}/{r['transition_no']} & "
        f"{r['stable_yes']}/{r['stable_no']} & "
        f"{fmt_or(r['odds_ratio'])} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")