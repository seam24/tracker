import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
pid = "cookiecutter"
DATA_ROOT = Path("D:/data_from_hpc")

BASE = None
WITH_LBL_DIR = None
COMMITS_PATH = None
MUTANT_HISTORY_PATH = None
REPO_DIR = None
BUGSINPY = None

REQUIRE_PROGRAM_EXITSTATUS_0_OR_1 = True
SAVE_FULL_JSON = False
PRINT_FAILING_TEST_IDS = False
PRINT_PER_BUG_ROWS = False


# ============================================================
# BASIC HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def read_text_if_exists(path):
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


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
        return name[len("results_") : -len(".json")]

    return None


def is_usable_result(data):
    exitstatus = data.get("program", {}).get("exitstatus")
    if REQUIRE_PROGRAM_EXITSTATUS_0_OR_1:
        return exitstatus in (0, 1)
    return True


def pct(x, denom):
    return 0.0 if denom == 0 else 100.0 * x / denom


def pct_str(x, denom):
    return f"{pct(x, denom):.1f}%"


def discover_bids(project_id):
    out = []
    pattern = re.compile(rf"^{re.escape(project_id)}-(\d+)b$")
    for path in sorted(DATA_ROOT.iterdir()):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            out.append(match.group(1))
    return out


# ============================================================
# BUGSINPY HELPERS
# ============================================================
def parse_bug_info(path):
    text = read_text_if_exists(path)
    info = {}
    if not text:
        return info

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        info[key] = value
    return info


def parse_patch_changed_files(path):
    text = read_text_if_exists(path)
    files = []
    if not text:
        return files

    for line in text.splitlines():
        match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
        if match:
            old_path, new_path = match.groups()
            files.append(
                {
                    "old_path": old_path.replace("\\", "/"),
                    "new_path": new_path.replace("\\", "/"),
                }
            )
    return files


def load_bug_metadata():
    info = parse_bug_info(BUGSINPY / "bug.info")
    patch_files = parse_patch_changed_files(BUGSINPY / "bug_patch.txt")
    return {
        "bug_info": info,
        "patch_changed_files": patch_files,
        "patch_changed_new_paths": {x["new_path"] for x in patch_files},
        "patch_changed_old_paths": {x["old_path"] for x in patch_files},
    }


# ============================================================
# MUTANT STATUS HELPERS
# ============================================================
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


def utestu_status_is_killed(tinfo):
    return (tinfo.get("status") or "").lower().strip() == "killed"


# ============================================================
# PYTEST NODEID NORMALIZATION
# ============================================================
def strip_param_part(nodeid):
    return nodeid.split("[", 1)[0]


def normalize_path(path):
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def normalize_pytest_nodeid(nodeid):
    clean = strip_param_part(nodeid)
    parts = clean.split("::")

    if not parts:
        return None

    file_path = normalize_path(parts[0])
    if not file_path.endswith(".py"):
        return None

    rest = [x for x in parts[1:] if x != "()"]

    if len(rest) == 1:
        class_name = None
        func_name = rest[0]
    elif len(rest) >= 2:
        class_name = rest[-2]
        func_name = rest[-1]
    else:
        return None

    function_id = (
        f"{file_path}::{class_name}::{func_name}"
        if class_name
        else f"{file_path}::{func_name}"
    )

    return {
        "original_nodeid": nodeid,
        "clean_nodeid": clean,
        "file_path": file_path,
        "class_name": class_name,
        "func_name": func_name,
        "function_id": function_id,
    }


def build_previous_test_index(test_ids):
    index = {}
    for tid in test_ids:
        norm = normalize_pytest_nodeid(tid)
        if norm is None:
            continue
        index.setdefault(norm["function_id"], []).append(norm)
    return index


# ============================================================
# RENAME-AWARE FILE RESOLUTION
# ============================================================
def find_possible_old_path(commit_before, commit_after, new_path):
    raw = git(
        [
            "diff",
            "--name-status",
            "--find-renames",
            "--find-copies",
            commit_before,
            commit_after,
        ]
    )

    if not raw:
        return None

    new_path = normalize_path(new_path)
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            status = parts[0]
            old_path = normalize_path(parts[1])
            new_path2 = normalize_path(parts[2])
            if (status.startswith("R") or status.startswith("C")) and new_path2 == new_path:
                return old_path

    return None


def git_file_before_rename_aware(commit_before, commit_after, new_path, fallback_old_path=None):
    before_text = git_file(commit_before, new_path)
    if before_text is not None:
        return before_text, new_path, False

    for old_path in [fallback_old_path, find_possible_old_path(commit_before, commit_after, new_path)]:
        if not old_path:
            continue
        old_text = git_file(commit_before, old_path)
        if old_text is not None:
            return old_text, old_path, True

    return None, new_path, False


# ============================================================
# AST HELPERS
# ============================================================
def iter_classes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


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

    for cls in iter_classes(tree):
        if cls.name != class_name:
            continue
        for item in cls.body:
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
                func = ctx.func
                if isinstance(func, ast.Attribute) and func.attr == "raises":
                    return True
                if isinstance(func, ast.Name) and func.id == "raises":
                    return True

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and (func.attr.startswith("assert") or func.attr == "raises"):
            return True
        if isinstance(func, ast.Name) and (func.id.startswith("assert") or func.id == "raises"):
            return True

    return False


def collect_assertion_dumps(func_node):
    return sorted(
        ast.dump(node, include_attributes=False)
        for node in ast.walk(func_node)
        if is_assertion_node(node)
    )


def classify_test_function_change(
    commit_before,
    commit_after,
    tid,
    before_test_ids,
    before_test_index,
    patch_changed_paths,
):
    norm = normalize_pytest_nodeid(tid)

    if norm is None:
        return {
            "test_id": tid,
            "function_id": None,
            "category": "unknown_bad_nodeid",
            "reason": "cannot_parse_pytest_nodeid",
        }

    exact_nodeid_existed_before = tid in before_test_ids
    function_existed_before = norm["function_id"] in before_test_index
    fallback_old_path = None
    if function_existed_before:
        fallback_old_path = before_test_index[norm["function_id"]][0]["file_path"]

    file_path = norm["file_path"]
    before_source, before_path_resolved, renamed_or_copied = git_file_before_rename_aware(
        commit_before, commit_after, file_path, fallback_old_path=fallback_old_path
    )
    after_source = git_file(commit_after, file_path)

    base = {
        "test_id": tid,
        "function_id": norm["function_id"],
        "test_file": file_path,
        "resolved_before_file_path": before_path_resolved,
        "resolved_after_file_path": file_path,
        "rename_or_copy_detected": renamed_or_copied,
        "patch_changed_test_file": file_path in patch_changed_paths
        or before_path_resolved in patch_changed_paths,
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
            "reason": "test_file_absent_before_present_after",
        }

    if before_source is None and after_source is None:
        return {
            **base,
            "category": "unknown_missing_file_both",
            "reason": "file_missing_in_before_and_after_commits",
        }

    if after_source is None:
        return {
            **base,
            "category": "unknown_missing_after",
            "reason": "file_missing_after_commit",
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
    return sorted(
        tid
        for tid, tinfo in minfo.get("tests", {}).items()
        if utestu_status_is_killed(tinfo)
    )



# ============================================================
# ONE-BUG ANALYSIS
# ============================================================
def analyze_bid(project_id, bug_id):
    global BASE, WITH_LBL_DIR, COMMITS_PATH, MUTANT_HISTORY_PATH, REPO_DIR, BUGSINPY

    BASE = DATA_ROOT / f"{project_id}-{bug_id}b"
    WITH_LBL_DIR = BASE / "with_lbl"
    COMMITS_PATH = BASE / "commits_to_run.txt"
    MUTANT_HISTORY_PATH = BASE / "mutant_history.json"
    REPO_DIR = Path(f"commits2 path here/{project_id}-{bug_id}b/{project_id}")
    BUGSINPY = Path(
        f"BUGSINPY projects path here/{project_id}/bugs/{bug_id}"
    )

    metadata = load_bug_metadata()
    bug_info = metadata["bug_info"]

    with open(COMMITS_PATH, "r", encoding="utf-8", errors="replace") as f:
        commits_newest_to_oldest = [line.strip() for line in f if line.strip()]

    if len(commits_newest_to_oldest) < 2:
        raise RuntimeError("Need at least two commits in commits_to_run.txt")

    buggy_commit = bug_info.get("buggy_commit_id") or commits_newest_to_oldest[0]
    fixed_commit = bug_info.get("fixed_commit_id")

    if buggy_commit not in commits_newest_to_oldest:
        raise RuntimeError(f"buggy_commit_id is not in commits_to_run.txt: {buggy_commit}")

    buggy_index = commits_newest_to_oldest.index(buggy_commit)
    if buggy_index + 1 >= len(commits_newest_to_oldest):
        raise RuntimeError(f"No previous commit exists after buggy commit in commits_to_run.txt: {buggy_commit}")

    previous_commit = commits_newest_to_oldest[buggy_index + 1]
    test_source_after_commit = fixed_commit or buggy_commit

    results_by_commit = {}
    for commit in commits_newest_to_oldest:
        result_path = WITH_LBL_DIR / f"results_{commit}.json"
        if not result_path.exists():
            continue

        data = load_json(result_path)
        real_commit = result_commit_id(result_path, data)

        if real_commit is None:
            continue
        if not is_usable_result(data):
            continue

        results_by_commit[real_commit] = data

    if buggy_commit not in results_by_commit:
        raise RuntimeError(f"Buggy commit result not found or not usable: {buggy_commit}")

    if previous_commit not in results_by_commit:
        raise RuntimeError(f"Previous commit result not found or not usable: {previous_commit}")

    buggy_data = results_by_commit[buggy_commit]
    previous_data = results_by_commit[previous_commit]

    buggy_program_tests = buggy_data.get("program", {}).get("tests", {})
    previous_program_tests = previous_data.get("program", {}).get("tests", {})

    failing_tests_buggy = sorted(
        tid
        for tid, tinfo in buggy_program_tests.items()
        if (tinfo.get("outcome") or "").lower().strip() == "failed"
    )

    if PRINT_FAILING_TEST_IDS:
        print(f"Failing tests at {project_id}-{bug_id}b:")
        for tid in failing_tests_buggy:
            print(" ", tid)

    previous_test_ids = set(previous_program_tests.keys())
    previous_test_index = build_previous_test_index(previous_test_ids)
    patch_changed_paths = metadata["patch_changed_new_paths"] | metadata["patch_changed_old_paths"]

    failing_test_classifications = [
        classify_test_function_change(
            commit_before=buggy_commit,
            commit_after=test_source_after_commit,
            tid=tid,
            before_test_ids=previous_test_ids,
            before_test_index=previous_test_index,
            patch_changed_paths=patch_changed_paths,
        )
        for tid in failing_tests_buggy
    ]

    fine_fail_counts = Counter(x["category"] for x in failing_test_classifications)
    coarse_fail_counts = Counter(coarse_test_category(x["category"]) for x in failing_test_classifications)
    num_failing_tests = len(failing_tests_buggy)
    failing_test_set = set(failing_tests_buggy)

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

    summary = {
        "project": f"{project_id}-{bug_id}b",
        "pid": project_id,
        "bid": bug_id,
        "buggy_commit": buggy_commit,
        "fixed_commit": fixed_commit,
        "previous_commit_for_mutants": previous_commit,
        "test_classification_compare": {
            "before": buggy_commit,
            "after": test_source_after_commit,
        },
        "bug_patch_changed_files": metadata["patch_changed_files"],
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
        out_json = Path("D:/res") / f"{project_id}-{bug_id}b_buggy_commit_analysis.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    return summary


# ============================================================
# PROJECT REPORT
# ============================================================
def counts_from_summary(summary):
    mutants = summary["buggy_commit_mutants"]
    tests = summary["failing_tests_at_buggy_commit"]
    coarse = Counter(tests.get("coarse_counts", {}))

    return Counter(
        {
            "bugs": 1,
            "available_mutants": mutants.get("available_mutants", 0),
            "recent_s_to_k": mutants.get("recent_S_to_K_at_buggy", 0),
            "recent_s_to_k_killed_by_failing": mutants.get(
                "recent_S_to_K_killed_by_at_least_one_failing_test", 0
            ),
            "recent_s_to_k_killed_only_by_failing": mutants.get(
                "recent_S_to_K_killed_only_by_failing_tests", 0
            ),
            "failing_tests": tests.get("total", 0),
            "new_tests": coarse.get("new_test", 0),
            "updated_tests": coarse.get("updated_existing_test", 0),
            "existing_tests": coarse.get("available_unchanged_test", 0),
            "unknown_tests": coarse.get("unknown", 0),
        }
    )


def report_row(label, counts):
    recent = counts["recent_s_to_k"]
    failing_tests = counts["failing_tests"]
    return {
        "prj": label,
        "Recent S->K (%)": pct_str(recent, counts["available_mutants"]),
        "Recent S->K killed by failing tests": pct_str(
            counts["recent_s_to_k_killed_by_failing"], recent
        ),
        "Recent S->K Killed Only by Failing Tests": pct_str(
            counts["recent_s_to_k_killed_only_by_failing"], recent
        ),
        "New Tests": pct_str(counts["new_tests"], failing_tests),
        "Updated Tests": pct_str(counts["updated_tests"], failing_tests),
        "Existing Tests": pct_str(counts["existing_tests"], failing_tests),
        "? Tests": pct_str(counts["unknown_tests"], failing_tests),
    }


def print_markdown_table(rows):
    headers = [
        "prj",
        "Recent S->K (%)",
        "Recent S->K killed by failing tests",
        "Recent S->K Killed Only by Failing Tests",
        "New Tests",
        "Updated Tests",
        "Existing Tests",
        "? Tests",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in headers) + " |")


def main():
    bids = discover_bids(pid)
    if not bids:
        raise RuntimeError(f"No directories found under {DATA_ROOT} matching {pid}-<bid>b")

    project_counts = Counter()
    rows = []
    skipped = []

    for bid in bids:
        label = f"{pid}-{bid}b"
        try:
            summary = analyze_bid(pid, bid)
        except Exception as exc:
            skipped.append((label, str(exc)))
            continue

        counts = counts_from_summary(summary)
        project_counts.update(counts)

        if PRINT_PER_BUG_ROWS:
            rows.append(report_row(label, counts))

    rows.append(report_row(pid, project_counts))
    print_markdown_table(rows)

    if skipped:
        print()
        print("Skipped bugs:")
        for label, reason in skipped:
            print(f"- {label}: {reason}")

    print()
    print(json.dumps({"project": pid, "raw_counts": dict(project_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
