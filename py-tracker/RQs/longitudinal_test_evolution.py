import ast
import json
import subprocess
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================
pid, bid = "fastapi", "4"

BASE = Path(f"D://data_from_hpc//{pid}-{bid}b")
WITH_LBL_DIR = BASE / "with_lbl"
COMMITS_PATH = BASE / "commits_to_run.txt"
MUTANT_HISTORY_PATH = BASE / "mutant_history.json"

REPO_DIR = Path(f"C://Users//tahar//Desktop//shared//LatentMu//commits2//{pid}-{bid}b/{pid}")
OUT_JSON = Path(f"D://res") / f"{pid}-{bid}b_buggy_commit_analysis.json"

BUGSINPY=Path(f"C:\\Users\\tahar\\Desktop\\shared\\LatentMu\\BugsInPy2\\BugsInPy\\projects\\{pid}\\bugs\\{bid}")

REQUIRE_PROGRAM_EXITSTATUS_0_OR_1 = True
SAVE_FULL_JSON = False
PRINT_FAILING_TEST_IDS = False


# ============================================================
# BASIC HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def git(args, quiet=True):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_DIR)] + args,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL if quiet else None,
        )
    except subprocess.CalledProcessError:
        return None


def git_file(commit, file_path):
    return git(["show", f"{commit}:{file_path}"], quiet=True)


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


def pct(x, denom):
    return 0.0 if denom == 0 else 100.0 * x / denom


def utestu_status_is_killed(tinfo):
    return (tinfo.get("status") or "").lower().strip() == "killed"


# ============================================================
# PYTEST NODEID NORMALIZATION
# ============================================================
def strip_param_part(nodeid):
    return nodeid.split("[", 1)[0]


def normalize_pytest_nodeid(nodeid):
    clean = strip_param_part(nodeid)
    parts = clean.split("::")

    if not parts:
        return None

    file_path = parts[0].replace("\\", "/")
    if not file_path.endswith(".py"):
        return None

    rest = parts[1:]
    rest = [x for x in rest if x != "()"]

    if len(rest) == 1:
        class_name = None
        func_name = rest[0]
    elif len(rest) >= 2:
        class_name = rest[-2]
        func_name = rest[-1]
    else:
        return None

    if class_name:
        function_id = f"{file_path}::{class_name}::{func_name}"
    else:
        function_id = f"{file_path}::{func_name}"

    return {
        "original_nodeid": nodeid,
        "clean_nodeid": clean,
        "file_path": file_path,
        "class_name": class_name,
        "func_name": func_name,
        "function_id": function_id,
    }


def normalized_function_ids(test_ids):
    out = set()
    for tid in test_ids:
        norm = normalize_pytest_nodeid(tid)
        if norm is not None:
            out.add(norm["function_id"])
    return out


# ============================================================
# RENAME-AWARE FILE RESOLUTION
# ============================================================
def find_possible_old_path(commit_before, commit_after, new_path):
    raw = git([
        "diff",
        "--name-status",
        "--find-renames",
        "--find-copies",
        commit_before,
        commit_after,
        "--",
        new_path,
    ])

    if not raw:
        return None

    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            status = parts[0]
            old_path = parts[1].replace("\\", "/")
            new_path2 = parts[2].replace("\\", "/")

            if (status.startswith("R") or status.startswith("C")) and new_path2 == new_path:
                return old_path

    return None


def git_file_before_rename_aware(commit_before, commit_after, new_path):
    before_text = git_file(commit_before, new_path)
    if before_text is not None:
        return before_text, new_path, False

    old_path = find_possible_old_path(commit_before, commit_after, new_path)
    if old_path:
        old_text = git_file(commit_before, old_path)
        if old_text is not None:
            return old_text, old_path, True

    return None, new_path, False


# ============================================================
# AST HELPERS
# ============================================================
def find_test_function_ast(source, norm):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, "syntax_error"

    class_name = norm["class_name"]
    func_name = norm["func_name"]

    if class_name is None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                return node, "ok"
        return None, "function_not_found"

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == func_name:
                    return item, "ok"
            return None, "method_not_found"

    return None, "class_not_found"


def ast_dump(node):
    if node is None:
        return None
    return ast.dump(node, include_attributes=False)


def is_assertion_node(node):
    if isinstance(node, ast.Assert):
        return True

    if isinstance(node, ast.With):
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                f = ctx.func
                if isinstance(f, ast.Attribute) and f.attr == "raises":
                    return True
                if isinstance(f, ast.Name) and f.id == "raises":
                    return True

    if isinstance(node, ast.Call):
        f = node.func

        if isinstance(f, ast.Attribute):
            if f.attr.startswith("assert") or f.attr == "raises":
                return True

        if isinstance(f, ast.Name):
            if f.id.startswith("assert") or f.id == "raises":
                return True

    return False


def collect_assertion_dumps(func_node):
    dumps = []
    for node in ast.walk(func_node):
        if is_assertion_node(node):
            dumps.append(ast.dump(node, include_attributes=False))
    return sorted(dumps)


def classify_test_function_change(commit_before, commit_after, tid, before_test_ids):
    norm = normalize_pytest_nodeid(tid)

    if norm is None:
        return {
            "test_id": tid,
            "function_id": None,
            "category": "unknown_bad_nodeid",
            "reason": "cannot_parse_pytest_nodeid",
        }

    before_function_ids = normalized_function_ids(before_test_ids)

    exact_nodeid_existed_before = tid in before_test_ids
    function_existed_before = norm["function_id"] in before_function_ids

    file_path = norm["file_path"]

    before_source, before_path_resolved, renamed_or_copied = git_file_before_rename_aware(
        commit_before, commit_after, file_path
    )
    after_source = git_file(commit_after, file_path)

    base = {
        "test_id": tid,
        "function_id": norm["function_id"],
        "test_file": file_path,
        "resolved_before_file_path": before_path_resolved,
        "resolved_after_file_path": file_path,
        "rename_or_copy_detected": renamed_or_copied,
    }

    if not exact_nodeid_existed_before and function_existed_before:
        return {
            **base,
            "category": "updated_test_new_parameter_case",
            "reason": "pytest_nodeid_absent_before_but_test_function_existed",
        }

    if before_source is None and after_source is not None:
        return {
            **base,
            "category": "new_test_function",
            "reason": "test_file_or_function_absent_before",
        }

    if before_source is None and after_source is None:
        return {
            **base,
            "category": "unknown_missing_file_both",
            "reason": "file_missing_in_both_commits",
        }

    if after_source is None:
        return {
            **base,
            "category": "unknown_missing_after",
            "reason": "file_missing_after",
        }

    before_func, before_reason = find_test_function_ast(before_source, norm)
    after_func, after_reason = find_test_function_ast(after_source, norm)

    if before_func is None and after_func is not None:
        return {
            **base,
            "category": "new_test_function",
            "reason": before_reason,
        }

    if before_func is None and after_func is None:
        return {
            **base,
            "category": "unknown_function_missing_both",
            "reason": f"before={before_reason}, after={after_reason}",
        }

    if before_func is not None and after_func is None:
        return {
            **base,
            "category": "unknown_function_missing_after",
            "reason": after_reason,
        }

    if ast_dump(before_func) == ast_dump(after_func):
        return {
            **base,
            "category": "available_unchanged_test",
            "reason": "test_function_ast_unchanged",
        }

    before_asserts = collect_assertion_dumps(before_func)
    after_asserts = collect_assertion_dumps(after_func)

    if before_asserts != after_asserts:
        return {
            **base,
            "category": "updated_test_assertion_or_oracle_changed",
            "reason": "assertion_ast_changed",
            "num_assertions_before": len(before_asserts),
            "num_assertions_after": len(after_asserts),
        }

    return {
        **base,
        "category": "updated_test_non_assertion_changed",
        "reason": "function_ast_changed_but_assertions_same",
        "num_assertions_before": len(before_asserts),
        "num_assertions_after": len(after_asserts),
    }


def coarse_test_category(cat):
    if cat.startswith("new_test"):
        return "new_test"
    if cat.startswith("updated_test"):
        return "updated_existing_test"
    if cat == "available_unchanged_test":
        return "available_unchanged_test"
    return "unknown"


# ============================================================
# STATUS SEQUENCE HELPERS
# ============================================================
def nearest_previous_ks_status(mu_hist, mid, buggy_commit, commits_newest_to_oldest, results_by_commit):
    """
    Find the nearest previous non-anomalous status before buggy_commit.
    Skips missing commits and A/I/U. Returns (status, commit) where status is K or S.
    """
    seen_buggy = False

    for commit in commits_newest_to_oldest:
        if commit == buggy_commit:
            seen_buggy = True
            continue

        if not seen_buggy:
            continue

        if commit not in results_by_commit:
            continue

        if mid not in results_by_commit[commit].get("mutants", {}):
            continue

        try:
            st = status_label(mu_hist, mid, commit)
        except Exception:
            continue

        if st in {"K", "S"}:
            return st, commit

    return None, None


def mutant_killing_tests_at_commit(results_by_commit, commit, mid):
    minfo = results_by_commit.get(commit, {}).get("mutants", {}).get(mid, {})
    return sorted([
        tid
        for tid, tinfo in minfo.get("tests", {}).items()
        if utestu_status_is_killed(tinfo)
    ])


# ============================================================
# LOAD COMMITS
# ============================================================
with open(COMMITS_PATH, "r", encoding="utf-8", errors="replace") as f:
    commits_newest_to_oldest = [line.strip() for line in f if line.strip()]

if len(commits_newest_to_oldest) < 2:
    raise RuntimeError("Need at least two commits in commits_to_run.txt")

buggy_commit = commits_newest_to_oldest[0]
previous_commit = commits_newest_to_oldest[1]


# ============================================================
# LOAD RESULTS
# ============================================================
results_by_commit = {}

for commit in commits_newest_to_oldest:
    path = WITH_LBL_DIR / f"results_{commit}.json"
    if not path.exists():
        continue

    data = load_json(path)
    real_commit = result_commit_id(path, data)

    if real_commit is None:
        continue

    if not is_usable_result(data):
        continue

    results_by_commit[real_commit] = data

if buggy_commit not in results_by_commit:
    raise RuntimeError(f"Buggy/newest commit result not found or not usable: {buggy_commit}")

if previous_commit not in results_by_commit:
    raise RuntimeError(f"Previous commit result not found or not usable: {previous_commit}")

buggy_data = results_by_commit[buggy_commit]
previous_data = results_by_commit[previous_commit]


# ============================================================
# FAILING TESTS AT BUGGY COMMIT: new / updated / unchanged
# ============================================================
buggy_program_tests = buggy_data.get("program", {}).get("tests", {})
previous_program_tests = previous_data.get("program", {}).get("tests", {})

failing_tests_buggy = sorted([
    tid
    for tid, tinfo in buggy_program_tests.items()
    if (tinfo.get("outcome") or "").lower().strip() == "failed"
])

if PRINT_FAILING_TEST_IDS:
    print("Failing tests at buggy commit:")
    for tid in failing_tests_buggy:
        print(" ", tid)

previous_test_ids = set(previous_program_tests.keys())

failing_test_classifications = [
    classify_test_function_change(
        commit_before=previous_commit,
        commit_after=buggy_commit,
        tid=tid,
        before_test_ids=previous_test_ids,
    )
    for tid in failing_tests_buggy
]

fine_fail_counts = Counter(x["category"] for x in failing_test_classifications)
coarse_fail_counts = Counter(coarse_test_category(x["category"]) for x in failing_test_classifications)
num_failing_tests = len(failing_tests_buggy)
failing_test_set = set(failing_tests_buggy)


# ============================================================
# LOAD MUTANT HISTORY + POPULATE STATUS LABELS
# ============================================================
mu_hist = load_json(MUTANT_HISTORY_PATH)

for mid in mu_hist:
    mu_hist[mid]["dict_history"] = {
        item["commit"]: item
        for item in mu_hist[mid].get("history", [])
    }

for commit, data in results_by_commit.items():
    for mid, minfo in data.get("mutants", {}).items():
        if mid not in mu_hist:
            continue
        if commit not in mu_hist[mid]["dict_history"]:
            continue
        mu_hist[mid]["dict_history"][commit]["status"] = mutant_status_from_tests(minfo)


# ============================================================
# MUTANT QUESTION:
# killed at buggy commit, nearest previous K/S was S,
# and whether buggy kill intersects failing tests.
# ============================================================
buggy_mutant_ids = set(buggy_data.get("mutants", {}).keys())
available_buggy_mutants = sorted(mid for mid in buggy_mutant_ids if mid in mu_hist)

buggy_status_counts = Counter()
buggy_k_mutants = []

recent_s_to_k_mutants = []
recent_s_to_k_killed_by_failing = []
recent_s_to_k_killed_only_by_failing = []
recent_s_to_k_killed_by_passing_too = []

buggy_k_nearest_prev_counts = Counter()

for mid in available_buggy_mutants:
    try:
        st_buggy = status_label(mu_hist, mid, buggy_commit)
    except Exception:
        continue

    buggy_status_counts[st_buggy] += 1

    if st_buggy != "K":
        continue

    buggy_k_mutants.append(mid)

    prev_st, prev_commit = nearest_previous_ks_status(
        mu_hist=mu_hist,
        mid=mid,
        buggy_commit=buggy_commit,
        commits_newest_to_oldest=commits_newest_to_oldest,
        results_by_commit=results_by_commit,
    )

    buggy_k_nearest_prev_counts[prev_st or "none"] += 1

    if prev_st != "S":
        continue

    killers = mutant_killing_tests_at_commit(results_by_commit, buggy_commit, mid)
    killers_set = set(killers)
    failing_killers = sorted(killers_set & failing_test_set)
    nonfailing_killers = sorted(killers_set - failing_test_set)

    rec = {
        "mutant_id": mid,
        "buggy_status": "K",
        "nearest_previous_status": prev_st,
        "nearest_previous_commit": prev_commit,
        "num_killing_tests_at_buggy": len(killers),
        "killing_tests_at_buggy": killers,
        "num_failing_killers_at_buggy": len(failing_killers),
        "failing_killers_at_buggy": failing_killers,
        "num_nonfailing_killers_at_buggy": len(nonfailing_killers),
        "nonfailing_killers_at_buggy": nonfailing_killers,
    }

    recent_s_to_k_mutants.append(rec)

    if failing_killers:
        recent_s_to_k_killed_by_failing.append(rec)

        if not nonfailing_killers:
            recent_s_to_k_killed_only_by_failing.append(rec)
        else:
            recent_s_to_k_killed_by_passing_too.append(rec)


# ============================================================
# SUMMARY
# ============================================================
summary = {
    "project": f"{pid}-{bid}b",
    "buggy_commit": buggy_commit,
    "previous_commit": previous_commit,

    "failing_tests_at_buggy_commit": {
        "total": num_failing_tests,
        "coarse_counts": dict(coarse_fail_counts),
        "coarse_percentages": {
            k: round(pct(v, num_failing_tests), 2)
            for k, v in coarse_fail_counts.items()
        },
        "fine_counts": dict(fine_fail_counts),
        "fine_percentages": {
            k: round(pct(v, num_failing_tests), 2)
            for k, v in fine_fail_counts.items()
        },
    },

    "buggy_commit_mutants": {
        "available_mutants": len(available_buggy_mutants),
        "status_counts": dict(buggy_status_counts),
        "killed_at_buggy": len(buggy_k_mutants),
        "nearest_previous_status_among_buggy_K": dict(buggy_k_nearest_prev_counts),

        "recent_S_to_K_at_buggy": len(recent_s_to_k_mutants),
        "recent_S_to_K_at_buggy_pct_of_buggy_available": round(
            pct(len(recent_s_to_k_mutants), len(available_buggy_mutants)), 2
        ),
        "recent_S_to_K_at_buggy_pct_of_buggy_K": round(
            pct(len(recent_s_to_k_mutants), len(buggy_k_mutants)), 2
        ),

        "recent_S_to_K_killed_by_at_least_one_failing_test": len(recent_s_to_k_killed_by_failing),
        "recent_S_to_K_killed_by_at_least_one_failing_test_pct_of_recent_S_to_K": round(
            pct(len(recent_s_to_k_killed_by_failing), len(recent_s_to_k_mutants)), 2
        ),
        "recent_S_to_K_killed_by_at_least_one_failing_test_pct_of_buggy_K": round(
            pct(len(recent_s_to_k_killed_by_failing), len(buggy_k_mutants)), 2
        ),
        "recent_S_to_K_killed_by_at_least_one_failing_test_pct_of_buggy_available": round(
            pct(len(recent_s_to_k_killed_by_failing), len(available_buggy_mutants)), 2
        ),

        "recent_S_to_K_killed_only_by_failing_tests": len(recent_s_to_k_killed_only_by_failing),
        "recent_S_to_K_killed_only_by_failing_tests_pct_of_recent_S_to_K": round(
            pct(len(recent_s_to_k_killed_only_by_failing), len(recent_s_to_k_mutants)), 2
        ),

        "recent_S_to_K_killed_by_failing_and_nonfailing_tests": len(recent_s_to_k_killed_by_passing_too),
        "recent_S_to_K_killed_by_failing_and_nonfailing_tests_pct_of_recent_S_to_K": round(
            pct(len(recent_s_to_k_killed_by_passing_too), len(recent_s_to_k_mutants)), 2
        ),
    },
}


details = {
    "summary": summary,
    "failing_tests": failing_tests_buggy,
    "failing_test_classifications": failing_test_classifications,
    "recent_S_to_K_mutants": recent_s_to_k_mutants,
    "recent_S_to_K_killed_by_at_least_one_failing_test": recent_s_to_k_killed_by_failing,
    "recent_S_to_K_killed_only_by_failing_tests": recent_s_to_k_killed_only_by_failing,
    "recent_S_to_K_killed_by_failing_and_nonfailing_tests": recent_s_to_k_killed_by_passing_too,
}

if SAVE_FULL_JSON:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, ensure_ascii=False))
