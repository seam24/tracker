from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import PurePosixPath
from typing import Dict, Optional, Set, List, Iterable, Any
from datetime import timedelta
from git import Repo


# ============================================================
# Debug logging
# ============================================================

DEBUG = True
_t0 = time.time()


def log(msg: str) -> None:
    if DEBUG:
        dt = time.time() - _t0
        print(f"[{dt:8.2f}s] {msg}", file=sys.stderr, flush=True)


# ============================================================
# Git helpers
# ============================================================

def list_files(repo: Repo, commit: str) -> List[str]:
    out = repo.git.ls_tree("-r", "--name-only", commit)
    return [x.strip().replace("\\", "/") for x in out.splitlines() if x.strip()]


def read_file(repo: Repo, commit: str, path: str) -> Optional[str]:
    try:
        return repo.git.show(f"{commit}:{path}")
    except Exception:
        return None


# ============================================================
# Python module mapping
# ============================================================

def module_for_path(path: str) -> str:
    p = PurePosixPath(path)
    if p.name == "__init__.py":
        return ".".join(p.parent.parts)
    return ".".join(p.with_suffix("").parts)


def build_module_index(py_files: Iterable[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for f in py_files:
        idx[module_for_path(f)] = f
    return idx


# ============================================================
# Import parsing
# ============================================================

def _resolve_relative(current_module: str, level: int, module: Optional[str]) -> Optional[str]:
    if level <= 0:
        return module

    base_parts = current_module.split(".")[:-1]
    if level > len(base_parts) + 1:
        return None

    prefix_len = max(0, len(base_parts) - (level - 1))
    prefix = base_parts[:prefix_len]

    if module:
        return ".".join(prefix + module.split("."))
    return ".".join(prefix)


def parse_import_modules(source: str, current_module: str) -> Set[str]:
    mods: Set[str] = set()
    try:
        tree = ast.parse(source)
    except Exception:
        return mods

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name:
                    mods.add(a.name)

        elif isinstance(n, ast.ImportFrom):
            base = _resolve_relative(current_module, n.level or 0, n.module)
            if base:
                mods.add(base)
                for a in n.names:
                    if a.name and a.name != "*":
                        mods.add(f"{base}.{a.name}")

    return mods


def resolve_import_to_file(mod: str, module_index: Dict[str, str]) -> Optional[str]:
    cur = mod
    while cur:
        if cur in module_index:
            return module_index[cur]
        if "." not in cur:
            break
        cur = cur.rsplit(".", 1)[0]
    return None


# ============================================================
# Test detection
# ============================================================

def is_test_file(path: str) -> bool:
    p = path.lower()
    name = PurePosixPath(p).name
    return (
        "/test" in p
        or "/tests" in p
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


# ============================================================
# Dependency logic
# ============================================================

def imports_of_file(repo: Repo, commit: str, path: str, module_index: Dict[str, str]) -> Set[str]:
    if not path.endswith(".py"):
        return set()

    src = read_file(repo, commit, path)
    if not src:
        return set()

    current_module = module_for_path(path)
    mods = parse_import_modules(src, current_module)

    out: Set[str] = set()
    for m in mods:
        f = resolve_import_to_file(m, module_index)
        if f and f != path:
            out.add(f)
    return out


def build_S(repo: Repo, commit: str, targets: Set[str]):
    files = list_files(repo, commit)
    py_files = [f for f in files if f.endswith(".py")]
    module_index = build_module_index(py_files)

    imports_map: Dict[str, Set[str]] = {f: set() for f in py_files}
    importers_map: Dict[str, Set[str]] = {f: set() for f in py_files}

    for f in py_files:
        imps = imports_of_file(repo, commit, f, module_index)
        imports_map[f] = imps
        for g in imps:
            importers_map[g].add(f)

    S: Set[str] = set()
    for t in targets:
        S.add(t)
        S |= imports_map.get(t, set())
        S |= importers_map.get(t, set())

    return S, imports_map


def tests_that_import_S(files: List[str], imports_map: Dict[str, Set[str]], S: Set[str]) -> Set[str]:
    tests: Set[str] = set()
    for f in files:
        if not (f.endswith(".py") and is_test_file(f)):
            continue
        if imports_map.get(f, set()) & S:
            tests.add(f)
    return tests


def neighborhood_and_tests_for_commit(repo: Repo, commit: str, targets: Set[str]) -> Set[str]:
    files = list_files(repo, commit)
    files_set = set(files)
    present_targets = {t for t in targets if t in files_set}
    if not present_targets:
        return set()

    S, imports_map = build_S(repo, commit, present_targets)
    T = tests_that_import_S(files, imports_map, S)
    return S | T


# ============================================================
# Commit range helpers
# ============================================================

def list_commits_to_head(repo_or_path, *, start_commit=None, days=None):
    repo = repo_or_path if isinstance(repo_or_path, Repo) else Repo(repo_or_path)

    if start_commit and days is not None:
        raise ValueError("Use either start_commit or days, not both.")

    cutoff = None
    if days is not None:
        cutoff = repo.head.commit.committed_datetime - timedelta(days=days)

    out = []
    for c in repo.iter_commits("HEAD", first_parent=True):
        out.append(c)

        if len(out) % 200 == 0:
            log(f"list_commits: collected {len(out)} commits")

        if start_commit and c.hexsha.startswith(start_commit):
            break
        if cutoff and c.committed_datetime < cutoff:
            break

    log(f"list_commits: total={len(out)}")
    return out


# ============================================================
# Diff analysis
# ============================================================

def parse_name_status(repo: Repo, parent_hexsha: str, commit_hexsha: str) -> List[Dict[str, Any]]:
    """
    Returns entries like:
      {"status": "A", "old_path": None, "new_path": "x.py"}
      {"status": "D", "old_path": "x.py", "new_path": None}
      {"status": "M", "old_path": "x.py", "new_path": "x.py"}
      {"status": "R", "old_path": "old.py", "new_path": "new.py"}
    """
    out = repo.git.diff("--name-status", "-M", parent_hexsha, commit_hexsha)
    entries: List[Dict[str, Any]] = []

    for line in out.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        status_raw = parts[0]

        if status_raw.startswith("R") and len(parts) >= 3:
            entries.append({
                "status": "R",
                "old_path": parts[1].replace("\\", "/"),
                "new_path": parts[2].replace("\\", "/"),
            })
            continue

        status = status_raw[:1]

        if status == "A" and len(parts) >= 2:
            entries.append({
                "status": "A",
                "old_path": None,
                "new_path": parts[1].replace("\\", "/"),
            })
        elif status == "D" and len(parts) >= 2:
            entries.append({
                "status": "D",
                "old_path": parts[1].replace("\\", "/"),
                "new_path": None,
            })
        elif status in {"M", "T", "C"} and len(parts) >= 2:
            path = parts[1].replace("\\", "/")
            entries.append({
                "status": "M",
                "old_path": path,
                "new_path": path,
            })

    return entries


def changed_paths_in_commit(repo: Repo, commit) -> Set[str]:
    if not commit.parents:
        return set(list_files(repo, commit.hexsha))

    parent = commit.parents[0]
    touched: Set[str] = set()

    for e in parse_name_status(repo, parent.hexsha, commit.hexsha):
        if e["old_path"]:
            touched.add(e["old_path"])
        if e["new_path"]:
            touched.add(e["new_path"])

    return touched


def classify_diff_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = {
        "added": [],
        "modified": [],
        "deleted": [],
        "renamed": [],
    }
    tests = {
        "added": [],
        "modified": [],
        "deleted": [],
        "renamed": [],
    }

    def bucket_for(path: Optional[str]) -> Optional[Dict[str, List[Any]]]:
        if not path or not path.endswith(".py"):
            return None
        return tests if is_test_file(path) else source

    for e in entries:
        status = e["status"]
        old_path = e["old_path"]
        new_path = e["new_path"]

        if status == "R":
            old_bucket = bucket_for(old_path)
            new_bucket = bucket_for(new_path)

            if old_bucket is tests or new_bucket is tests:
                tests["renamed"].append({"old": old_path, "new": new_path})
            elif old_bucket is source or new_bucket is source:
                source["renamed"].append({"old": old_path, "new": new_path})

        elif status == "A":
            b = bucket_for(new_path)
            if b is not None:
                b["added"].append(new_path)

        elif status == "D":
            b = bucket_for(old_path)
            if b is not None:
                b["deleted"].append(old_path)

        elif status == "M":
            b = bucket_for(new_path or old_path)
            if b is not None:
                b["modified"].append(new_path or old_path)

    return {
        "source": source,
        "tests": tests,
        "has_source_evolution": any(source[k] for k in source),
        "has_test_evolution": any(tests[k] for k in tests),
    }


def evolution_info_for_commit(repo: Repo, commit) -> Dict[str, Any]:
    if not commit.parents:
        files = [f for f in list_files(repo, commit.hexsha) if f.endswith(".py")]
        entries = [{"status": "A", "old_path": None, "new_path": f} for f in files]
    else:
        parent = commit.parents[0]
        entries = parse_name_status(repo, parent.hexsha, commit.hexsha)

    cls = classify_diff_entries(entries)
    return {
        "commit": commit.hexsha,
        "summary": commit.summary,
        **cls,
    }


# ============================================================
# Relevance logic (original dynamic mode)
# ============================================================

def is_relevant_commit(repo: Repo, commit, targets: Set[str]) -> bool:
    touched = changed_paths_in_commit(repo, commit)
    sel_c = neighborhood_and_tests_for_commit(repo, commit.hexsha, targets)
    return bool(touched & sel_c)


# ============================================================
# Relevance logic (new explicit-path mode with rename tracking)
# ============================================================

def commit_touches_watched_paths(repo: Repo, commit, watched_paths: Set[str]) -> bool:
    """
    True iff this commit touches any currently watched path.
    watched_paths are interpreted in the namespace of `commit` itself
    (the newer side when walking from HEAD backwards).
    """
    if not commit.parents:
        touched = set(list_files(repo, commit.hexsha))
        return bool(touched & watched_paths)

    parent = commit.parents[0]
    entries = parse_name_status(repo, parent.hexsha, commit.hexsha)

    for e in entries:
        old_path = e["old_path"]
        new_path = e["new_path"]
        if (old_path and old_path in watched_paths) or (new_path and new_path in watched_paths):
            return True

    return False


def rewind_watched_paths_through_commit(repo: Repo, commit, watched_paths: Set[str]) -> Set[str]:
    """
    Move watched paths one step backward in history:
    commit namespace  --->  parent namespace

    If a watched file was renamed in this commit from old -> new,
    and we are traversing backward, then we must replace new by old.
    """
    if not commit.parents:
        return set(watched_paths)

    parent = commit.parents[0]
    entries = parse_name_status(repo, parent.hexsha, commit.hexsha)

    out = set(watched_paths)

    for e in entries:
        if e["status"] != "R":
            continue

        old_path = e["old_path"]
        new_path = e["new_path"]

        if new_path in out:
            out.remove(new_path)
            out.add(old_path)

    return out


def collect_commits_for_explicit_paths(
    repo_or_path,
    watched_paths: Set[str],
    *,
    max_commits: int = 100,
    start_commit: Optional[str] = None,
    days: Optional[int] = None,
    include_head: bool = True,
    include_boundary: bool = True,
    step_log_every: int = 25,
):
    """
    Collect commits from HEAD backwards where one of the explicitly provided
    watched paths is changed. Renames are followed backward, so if utils.py
    used to live somewhere else, older commits still match that old location.

    `watched_paths` should usually be given as HEAD-era paths.
    """
    repo = repo_or_path if isinstance(repo_or_path, Repo) else Repo(repo_or_path)

    if start_commit is not None and days is not None:
        raise ValueError("Use either start_commit or days, not both.")

    cutoff = None
    if days is not None:
        cutoff = repo.head.commit.committed_datetime - timedelta(days=days)

    kept = []
    scanned = 0
    relevant_count = 0
    anchor_count = 0
    watched_now = set(watched_paths)

    log(
        f"collect_commits_for_explicit_paths: "
        f"max_commits={max_commits}, start_commit={start_commit}, days={days}, "
        f"include_head={include_head}, include_boundary={include_boundary}, "
        f"initial_watched={sorted(watched_now)}"
    )

    for c in repo.iter_commits("HEAD", first_parent=True):
        scanned += 1
        boundary_hit = bool(start_commit and c.hexsha.startswith(start_commit))

        if cutoff is not None and c.committed_datetime < cutoff:
            log(f"cutoff reached after scanning {scanned} commits")
            break

        if scanned == 1 and include_head:
            kept.append(c)
            anchor_count += 1
        else:
            hit = commit_touches_watched_paths(repo, c, watched_now)

            if hit:
                kept.append(c)
                relevant_count += 1
            elif boundary_hit and include_boundary:
                kept.append(c)
                anchor_count += 1

            if step_log_every and scanned % step_log_every == 0:
                log(
                    f"[{scanned}] relevant={hit} boundary={boundary_hit} "
                    f"kept={len(kept)} relevant_count={relevant_count} "
                    f"anchors={anchor_count} watched={sorted(watched_now)}"
                )

        if len(kept) >= max_commits:
            break

        if boundary_hit:
            break

        watched_now = rewind_watched_paths_through_commit(repo, c, watched_now)

    log(
        f"collect_commits_for_explicit_paths: done scanned={scanned} "
        f"kept={len(kept)} relevant_count={relevant_count} anchors={anchor_count} "
        f"final_watched={sorted(watched_now)}"
    )
    return kept


# ============================================================
# Collector (original dynamic mode)
# ============================================================

def collect_relevant_commits(
    repo_or_path,
    targets: Set[str],
    *,
    max_commits: Optional[int] = None,
    start_commit: Optional[str] = None,
    days: Optional[int] = None,
    include_head: bool = True,
    include_boundary: bool = True,
    step_log_every: int = 25,
):
    repo = repo_or_path if isinstance(repo_or_path, Repo) else Repo(repo_or_path)

    if start_commit is not None and days is not None:
        raise ValueError("Use either start_commit or days, not both.")

    cutoff = None
    if days is not None:
        cutoff = repo.head.commit.committed_datetime - timedelta(days=days)

    kept = []
    scanned = 0
    relevant_count = 0
    anchor_count = 0

    log(
        f"collect_relevant_commits: "
        f"max_commits={max_commits}, start_commit={start_commit}, days={days}, "
        f"include_head={include_head}, include_boundary={include_boundary}"
    )

    for c in repo.iter_commits("HEAD", first_parent=True):
        scanned += 1
        boundary_hit = bool(start_commit and c.hexsha.startswith(start_commit))

        if cutoff is not None and c.committed_datetime < cutoff:
            log(f"cutoff reached after scanning {scanned} commits")
            break

        if scanned == 1 and include_head:
            kept.append(c)
            anchor_count += 1

            if max_commits is not None and len(kept) >= max_commits:
                break

            if boundary_hit:
                break

            continue

        hit = is_relevant_commit(repo, c, targets)

        if hit:
            kept.append(c)
            relevant_count += 1
        elif boundary_hit and include_boundary:
            kept.append(c)
            anchor_count += 1

        if step_log_every and scanned % step_log_every == 0:
            log(
                f"[{scanned}] relevant={hit} boundary={boundary_hit} "
                f"kept={len(kept)} relevant_count={relevant_count} anchors={anchor_count}"
            )

        if max_commits is not None and len(kept) >= max_commits:
            break

        if boundary_hit:
            break

    log(
        f"collect_relevant_commits: done scanned={scanned} "
        f"kept={len(kept)} relevant_count={relevant_count} anchors={anchor_count}"
    )
    return kept


# ============================================================
# Save outputs
# ============================================================

def save_commits_to_run(path: str, commits) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in commits:
            f.write(c.hexsha + "\n")


def save_commit_evolution_json(path: str, repo: Repo, commits) -> None:
    data = [evolution_info_for_commit(repo, c) for c in commits]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_commit_evolution_jsonl(path: str, repo: Repo, commits) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in commits:
            rec = evolution_info_for_commit(repo, c)
            f.write(json.dumps(rec) + "\n")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    prj_name = "youtube-dl"
    bug_id = "43"
    repo_path = f"../../chk/{prj_name}-{bug_id}b/{prj_name}"

    repo = Repo(repo_path)

    # dynamic one-hop neighbours mode
    targets = {
        "youtube_dl/utils.py",
    }

    kept_dynamic = collect_relevant_commits(
        repo,
        targets,
        max_commits=25,
        include_head=True,
        include_boundary=True,
        step_log_every=10,
    )

    save_commits_to_run("commits_to_run_dynamic.txt", kept_dynamic)
    save_commit_evolution_json("commit_evolution_dynamic.json", repo, kept_dynamic)

    # static explicit-path mode
    # watched_paths = {
    #     "youtube_dl/utils.py",
    #     "test/test_utils.py",
    #     # "youtube_dl/compat.py",
    #     # "test/test_compat.py",
    # }

    # kept_explicit = collect_commits_for_explicit_paths(
    #     repo,
    #     watched_paths,
    #     max_commits=100,
    #     include_head=True,
    #     include_boundary=False,
    #     step_log_every=10,
    # )

    # save_commits_to_run(f"commits_to_run.txt", kept_explicit)
    # save_commit_evolution_json(f"commit_evolution.json", repo, kept_explicit)

    # log(f"saved {len(kept_explicit)} commits to commits_to_run.txt")
    # log(f"saved evolution metadata to commit_evolution.json")
