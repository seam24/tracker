#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button, TextBox, CheckButtons
import matplotlib as mpl

mpl.rcParams["keymap.save"] = []
mpl.rcParams["keymap.quit"] = []
mpl.rcParams["keymap.fullscreen"] = []
mpl.rcParams["keymap.grid"] = []
mpl.rcParams["keymap.pan"] = []
mpl.rcParams["keymap.zoom"] = []
mpl.rcParams["keymap.back"] = []
mpl.rcParams["keymap.forward"] = []

# ================== CONFIG ==================
DATA_ROOT = Path("/home/rooti/shared/LatentMu/data/data_from_hpc")
DEFAULT_COMMIT_FILTER_ROOT = Path("/home/rooti/shared/commit_filter/commit_filter")
DEFAULT_BUGSINPY_ROOT = Path("/home/rooti/shared/LatentMu/BugsInPy")

PRINT_UNIFIED_DIFF = True
MAX_RELEVANT_TESTS_TO_PRINT = 10
MAX_PROGRAM_FAILURES_TO_PRINT = 100
MAX_XTICKS = 30

LEFT_KILLISH = {"killed", "incompetent"}
RIGHT_KILLISH = {"killed", "incompetent"}
PROBLEMATIC_KILLISH = {"killed", "incompetent"}

# Program-side pytest-ish outcomes
PROGRAM_NON_FAILING_OUTCOMES = {
    "passed",
    "skipped",
    "xfailed",
    "deselected",
}
PROGRAM_FAILING_OUTCOMES = {
    "failed",
    "error",
    "xpassed",
}

SAVE_REASON_OPTIONS = [
    ("survived_to_killed", "Survived -> Killed"),
    ("killed_to_survived", "Killed -> Survived"),
    ("survived_for_good_reason", "Survived for good reason"),
    ("survived_killed_survived", "Surv -> Kill -> Surv"),
    ("killed_survived_killed", "Kill -> Surv -> Kill"),
]

# visual encoding for one-row mutant timeline
CELL_UNKNOWN = 0
CELL_SURVIVED = 1
CELL_KILLED = 2
CELL_INCOMPETENT = 3
CELL_ANOMALOUS = 4
CELL_NOT_EXPECTED = 5
CELL_PROBLEMATIC = 6
CELL_MISSING = 7

CELL_TO_LABEL = {
    CELL_UNKNOWN: "unknown",
    CELL_SURVIVED: "survived",
    CELL_KILLED: "killed",
    CELL_INCOMPETENT: "incompetent",
    CELL_ANOMALOUS: "anomalous",
    CELL_NOT_EXPECTED: "not_expected",
    CELL_PROBLEMATIC: "problematic_commit",
    CELL_MISSING: "missing",
}

STATUS_TO_CELL = {
    "unknown": CELL_UNKNOWN,
    "survived": CELL_SURVIVED,
    "killed": CELL_KILLED,
    "incompetent": CELL_INCOMPETENT,
    "anomalous": CELL_ANOMALOUS,
    "not_expected": CELL_NOT_EXPECTED,
    "problematic_commit": CELL_PROBLEMATIC,
    "missing": CELL_MISSING,
}


# ================== PATH RESOLUTION ==================
def parse_project_bug_from_base_dir(base_dir: Path) -> Tuple[str, str]:
    """
    Expects directories like:
      /home/rooti/shared/LatentMu/data/data_from_hpc/tqdm-8b

    Returns:
      ("tqdm", "8b")
    """
    name = base_dir.name
    if "-" not in name:
        raise SystemExit(
            f"Could not infer project/bug from base-dir name '{name}'. "
            f"Expected something like 'tqdm-8b'."
        )
    project, bug = name.rsplit("-", 1)
    if not project or not bug:
        raise SystemExit(
            f"Could not infer project/bug from base-dir name '{name}'. "
            f"Expected something like 'tqdm-8b'."
        )
    return project, bug


def normalize_bug_for_bugsinpy_and_commit_filter(bug: str) -> str:
    """
    commit_filter and BugsInPy paths use 8 instead of 8b.
    """
    m = re.match(r"(\d+)", bug)
    return m.group(1) if m else bug


def derive_default_paths(
    *,
    base_dir: Path,
    project: Optional[str],
    bug: Optional[str],
    save_file: Optional[Path],
    mutant_history: Optional[Path],
    commit_evolution_file: Optional[Path],
    bug_patch_file: Optional[Path],
) -> Dict[str, Any]:
    inferred_project, inferred_bug = parse_project_bug_from_base_dir(base_dir)

    project = project or inferred_project
    bug = bug or inferred_bug
    bug_num = normalize_bug_for_bugsinpy_and_commit_filter(bug)

    commits_file = base_dir / "commits_to_run.txt"
    results_dir = base_dir / "with_lbl"

    if mutant_history is None:
        candidate = base_dir / "mutant_history.json"
        mutant_history = candidate if candidate.exists() else None

    if commit_evolution_file is None:
        candidate = DEFAULT_COMMIT_FILTER_ROOT / project / bug_num / f"{project}_{bug_num}_commit_evolution.json"
        commit_evolution_file = candidate if candidate.exists() else None

    if bug_patch_file is None:
        candidate = DEFAULT_BUGSINPY_ROOT / "projects" / project / "bugs" / bug_num / "bug_patch.txt"
        bug_patch_file = candidate if candidate.exists() else None

    if save_file is None:
        save_file = base_dir / "saved_mutants.json"

    return {
        "project": project,
        "bug": bug,
        "bug_num": bug_num,
        "commits_file": commits_file,
        "results_dir": results_dir,
        "mutant_history": mutant_history,
        "commit_evolution_file": commit_evolution_file,
        "bug_patch_file": bug_patch_file,
        "save_file": save_file,
    }


# ================== BASIC HELPERS ==================
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return load_json(path)


def maybe_read_text(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def clip_msg(x: Any, limit: int = 240) -> str:
    if x is None:
        return ""
    s = str(x).replace("\n", "\\n")
    return s if len(s) <= limit else s[:limit] + "…"


def editor_hint(path: Path) -> str:
    ed = os.environ.get("EDITOR")
    if ed:
        return f"{ed} {path}"
    return f"less {path}   # or: jq . {path} | less"


def load_saved_records(path: Path) -> List[Dict[str, Any]]:
    """
    Preferred format: JSON array of objects.

    Backward compatibility:
    - JSONL (one JSON object per line)
    - a single JSON object
    - {"records": [...]} wrapper
    """
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("records"), list):
                return [x for x in obj["records"] if isinstance(x, dict)]
            return [obj]
    except Exception:
        pass

    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def write_saved_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


# ================== STATUS LOGIC ==================
def mutant_status_from_tests(mtests: Dict[str, Any]) -> str:
    if not mtests:
        return "unknown"

    statuses = [(t.get("status") or "unknown").lower() for t in mtests.values()]

    if any(s == "incompetent" for s in statuses):
        return "incompetent"
    if any(s == "killed" for s in statuses):
        return "killed"
    if all(s in ("survived", "unknown") for s in statuses) and any(s == "survived" for s in statuses):
        return "survived"
    if any(s == "anomalous" for s in statuses):
        return "anomalous"
    return "unknown"


def summarize_status_counts(mtests: Dict[str, Any]) -> Dict[str, int]:
    c = Counter()
    for td in (mtests or {}).values():
        c[(td.get("status") or "unknown").lower()] += 1
    return dict(c)


def normalize_program_test_outcome(td: Dict[str, Any]) -> str:
    """
    Program test rows use 'outcome'. Be tolerant and fall back to 'status'
    if some file deviates.
    """
    return (td.get("outcome") or td.get("status") or "").lower().strip()


def is_program_failure_outcome(outcome: str) -> bool:
    if not outcome:
        return False
    if outcome in PROGRAM_FAILING_OUTCOMES:
        return True
    if outcome in PROGRAM_NON_FAILING_OUTCOMES:
        return False
    # Conservative fallback: any other explicit non-empty outcome is treated as failing/non-passing.
    return True


# ================== EXPECTED EXISTENCE ==================
def compute_expected_from_history(
    mutant_ids: List[str],
    commits_newest_to_oldest: List[str],
    mutant_history: Dict[str, Any],
) -> Dict[str, Dict[str, bool]]:
    commit_to_idx = {c: i for i, c in enumerate(commits_newest_to_oldest)}
    expected = {mid: {c: False for c in commits_newest_to_oldest} for mid in mutant_ids}

    for mid in mutant_ids:
        mh = mutant_history.get(mid)
        if not mh:
            continue
        hist = mh.get("history", []) or []
        idxs = sorted({commit_to_idx[h["commit"]] for h in hist if h.get("commit") in commit_to_idx})
        if not idxs:
            continue

        if idxs[0] != 0:
            continue

        s = set(idxs)
        k = 0
        while k < len(commits_newest_to_oldest) and k in s:
            k += 1

        for i in range(k):
            expected[mid][commits_newest_to_oldest[i]] = True

    return expected


def compute_expected_from_presence(
    mutant_ids: List[str],
    commits_newest_to_oldest: List[str],
    commit_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, bool]]:
    expected = {mid: {c: False for c in commits_newest_to_oldest} for mid in mutant_ids}
    for c in commits_newest_to_oldest:
        data = commit_cache.get(c)
        if not data:
            continue
        mutants = data.get("mutants", {}) or {}
        for mid in mutants:
            if mid in expected:
                expected[mid][c] = True
    return expected


# ================== DIFF SUPPORT ==================
def reconstruct_mutant_code(original: str, start: int, end: int, val: str) -> str:
    return original[:start] + val + original[end:]


def git_show_file(repo: Path, commit: str, rel_path: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{rel_path}"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        return None


def unified_diff(original: str, mutated: str, path: str, commit: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(True),
            mutated.splitlines(True),
            fromfile=f"{path}@{commit[:8]} (original)",
            tofile=f"{path}@{commit[:8]} (mutant)",
            lineterm="",
            n=4,
        )
    )


def get_mutant_meta(history: Dict[str, Any], mid: str, commit: str) -> Optional[Tuple[str, int, int, str]]:
    mh = history.get(mid)
    if not mh:
        return None

    info = mh.get("info") or {}
    file_path = info.get("file") or mh.get("file")
    mutant_value = info.get("mutant_value")

    if not file_path or mutant_value is None:
        return None

    for h in (mh.get("history") or []):
        if h.get("commit") == commit:
            sp = h.get("start_pos")
            ep = h.get("end_pos")
            if isinstance(sp, int) and isinstance(ep, int):
                return file_path, sp, ep, str(mutant_value)

    sp = info.get("mutation_start_pos")
    ep = info.get("mutation_end_pos")
    if isinstance(sp, int) and isinstance(ep, int):
        return file_path, sp, ep, str(mutant_value)

    return None


def show_unified_diff(
    mutant_history: Optional[Dict[str, Any]],
    repo_root: Optional[Path],
    commit: str,
    mid: str,
) -> None:
    if not PRINT_UNIFIED_DIFF:
        return

    if mutant_history is None:
        print("\nUNIFIED DIFF: unavailable (mutant_history missing)")
        return
    if repo_root is None:
        print("\nUNIFIED DIFF: unavailable (repo_root missing)")
        return

    meta = get_mutant_meta(mutant_history, mid, commit)
    if not meta:
        print("\nUNIFIED DIFF: ❌ No mutant metadata available.")
        return

    path, start, end, val = meta
    original = git_show_file(repo_root, commit, path)
    if original is None:
        print("\nUNIFIED DIFF: ❌ Could not load source file from git show.")
        return

    if not (0 <= start <= end <= len(original)):
        print("\nUNIFIED DIFF: ❌ Offsets invalid for this commit/file.")
        print(f"  file_len={len(original)} start={start} end={end}")
        return

    mutated = reconstruct_mutant_code(original, start, end, val)
    print(unified_diff(original, mutated, path, commit))


# ================== COMMIT EVOLUTION ==================
def load_commit_evolution_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    data = load_json(path)
    if not isinstance(data, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        c = item.get("commit")
        if c:
            out[c] = item
    return out


def print_commit_evolution_info(commit: str, commit_evolution: Dict[str, Dict[str, Any]]) -> None:
    info = commit_evolution.get(commit)
    if not info:
        print("Commit evolution: unavailable")
        return

    print("\nCOMMIT EVOLUTION:")
    print(f"  commit               : {commit}")
    print(f"  summary              : {info.get('summary', '')}")
    print(f"  has_source_evolution : {info.get('has_source_evolution')}")
    print(f"  has_test_evolution   : {info.get('has_test_evolution')}")

    source = info.get("source") or {}
    tests = info.get("tests") or {}

    print("  source:")
    print(f"    modified : {source.get('modified', [])}")
    print(f"    added    : {source.get('added', [])}")
    print(f"    deleted  : {source.get('deleted', [])}")
    print(f"    renamed  : {source.get('renamed', [])}")

    print("  tests:")
    print(f"    modified : {tests.get('modified', [])}")
    print(f"    added    : {tests.get('added', [])}")
    print(f"    deleted  : {tests.get('deleted', [])}")
    print(f"    renamed  : {tests.get('renamed', [])}")


# ================== LOADING DATA ==================
def read_commits_file(commits_file: Path) -> List[str]:
    commits = [c.strip() for c in commits_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    if not commits:
        raise SystemExit(f"No commits in {commits_file}")
    return commits


def resolve_existing_result_paths(commits: List[str], results_dir: Path) -> Tuple[Dict[str, Path], List[str]]:
    existing_result_paths: Dict[str, Path] = {}
    missing_commits: List[str] = []

    for c in commits:
        p = results_dir / f"results_{c}.json"
        if p.exists():
            existing_result_paths[c] = p
        else:
            missing_commits.append(c)

    return existing_result_paths, missing_commits


def build_commit_cache(existing_result_paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    return {commit: load_json(path) for commit, path in existing_result_paths.items()}


def collect_mutant_ids(commit_cache: Dict[str, Dict[str, Any]]) -> List[str]:
    mids = set()
    for data in commit_cache.values():
        mids.update((data.get("mutants") or {}).keys())
    return sorted(mids)


def build_status_map(
    mutant_ids: List[str],
    commits_newest_to_oldest: List[str],
    commit_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    status_map: Dict[str, Dict[str, str]] = {mid: {} for mid in mutant_ids}

    for commit in commits_newest_to_oldest:
        data = commit_cache.get(commit)
        if not data:
            continue
        mutants = data.get("mutants", {}) or {}
        for mid, md in mutants.items():
            if mid in status_map:
                status_map[mid][commit] = mutant_status_from_tests((md.get("tests", {}) or {}))

    return status_map


# ================== PROBLEMATIC COMMITS ==================
def detect_problematic_commits(
    commits_chronological: List[str],
    mutant_ids: List[str],
    expected_map: Dict[str, Dict[str, bool]],
    status_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for commit in commits_chronological:
        observed: List[str] = []
        for mid in mutant_ids:
            if not expected_map.get(mid, {}).get(commit, False):
                continue
            st = status_map.get(mid, {}).get(commit)
            if st is None or st == "unknown":
                continue
            observed.append(st)

        is_problematic = bool(observed) and all(s in PROBLEMATIC_KILLISH for s in observed)

        out[commit] = {
            "problematic": is_problematic,
            "observed_expected_count": len(observed),
            "counts": dict(Counter(observed)),
        }

    return out


# ================== DISPLAY COMMIT FILTER ==================
def compute_display_commits(
    commits_chronological: List[str],
    mutant_ids: List[str],
    expected_map: Dict[str, Dict[str, bool]],
    status_map: Dict[str, Dict[str, str]],
) -> List[str]:
    kept: List[str] = []

    for commit in commits_chronological:
        observed: List[str] = []
        for mid in mutant_ids:
            if not expected_map.get(mid, {}).get(commit, False):
                continue
            st = status_map.get(mid, {}).get(commit, "unknown")
            observed.append(st)

        if not observed:
            continue

        if all(st in {"killed", "unknown"} for st in observed):
            continue

        kept.append(commit)

    return kept


# ================== MUTANT-LEVEL INTEREST ==================
def row_display_cells(
    mid: str,
    commits_chronological: List[str],
    expected_map: Dict[str, Dict[str, bool]],
    status_map: Dict[str, Dict[str, str]],
    problematic_info: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    row: List[Dict[str, Any]] = []
    for commit in commits_chronological:
        if not expected_map.get(mid, {}).get(commit, False):
            st = "not_expected"
        elif problematic_info.get(commit, {}).get("problematic", False):
            st = "problematic_commit"
        else:
            st = status_map.get(mid, {}).get(commit, "missing")
        row.append({"commit": commit, "status": st})
    return row


def filtered_sequence_for_transition_detection(
    mid: str,
    commits_chronological: List[str],
    expected_map: Dict[str, Dict[str, bool]],
    status_map: Dict[str, Dict[str, str]],
    problematic_info: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seq: List[Dict[str, Any]] = []

    for commit in commits_chronological:
        if not expected_map.get(mid, {}).get(commit, False):
            continue
        if problematic_info.get(commit, {}).get("problematic", False):
            continue

        st = status_map.get(mid, {}).get(commit)
        if st is None or st == "unknown":
            continue

        seq.append({"commit": commit, "status": st})

    return seq


def build_mutant_candidate(
    mid: str,
    commits_chronological: List[str],
    expected_map: Dict[str, Dict[str, bool]],
    status_map: Dict[str, Dict[str, str]],
    problematic_info: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    row_cells = row_display_cells(mid, commits_chronological, expected_map, status_map, problematic_info)
    seq = filtered_sequence_for_transition_detection(mid, commits_chronological, expected_map, status_map, problematic_info)

    interesting_points: List[Dict[str, Any]] = []

    if not seq:
        return None

    if all(x["status"] == "survived" for x in seq):
        # interesting_points.append(
        #     {
        #         "kind": "always_survived",
        #         "anchor_commit": seq[0]["commit"],
        #         "anchor_commit_to": seq[-1]["commit"],
        #         "from_commit": None,
        #         "to_commit": None,
        #         "from_status": None,
        #         "to_status": None,
        #         "transition": None,
        #     }
        # )
        return None

    for i in range(1, len(seq)):
        prev_ = seq[i - 1]
        curr_ = seq[i]
        a = prev_["status"]
        b = curr_["status"]

        if a in LEFT_KILLISH and b == "survived":
            interesting_points.append(
                {
                    "kind": "killed_or_incompetent_to_survived",
                    "anchor_commit": curr_["commit"],
                    "anchor_commit_to": curr_["commit"],
                    "from_commit": prev_["commit"],
                    "to_commit": curr_["commit"],
                    "from_status": a,
                    "to_status": b,
                    "transition": f"{a} -> {b}",
                }
            )
        elif a == "survived" and b in RIGHT_KILLISH:
            interesting_points.append(
                {
                    "kind": "survived_to_killed_or_incompetent",
                    "anchor_commit": curr_["commit"],
                    "anchor_commit_to": curr_["commit"],
                    "from_commit": prev_["commit"],
                    "to_commit": curr_["commit"],
                    "from_status": a,
                    "to_status": b,
                    "transition": f"{a} -> {b}",
                }
            )

    if not interesting_points:
        return None

    score = 0
    for p in interesting_points:
        if p["kind"] == "survived_to_killed_or_incompetent":
            score += 3
        elif p["kind"] == "killed_or_incompetent_to_survived":
            score += 2
        elif p["kind"] == "always_survived":
            score += 1

    return {
        "mutant_id": mid,
        "row_cells": row_cells,
        "interesting_points": interesting_points,
        "score": score,
        "filtered_seq": seq,
    }


def collect_mutant_candidates(
    mutant_ids: List[str],
    commits_chronological: List[str],
    expected_map: Dict[str, Dict[str, bool]],
    status_map: Dict[str, Dict[str, str]],
    problematic_info: Dict[str, Dict[str, Any]],
    saved_mutants: Set[str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for mid in mutant_ids:
        if mid in saved_mutants:
            continue
        cand = build_mutant_candidate(mid, commits_chronological, expected_map, status_map, problematic_info)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda x: (-x["score"], x["mutant_id"]))
    return candidates


# ================== INSPECTION ==================
def explain_program_failures(commit_data: Dict[str, Any], max_tests: int = MAX_PROGRAM_FAILURES_TO_PRINT) -> None:
    """
    Program-side logic:
    - program.tests[*] uses outcome, not mutant status
    - print a robust summary first
    - show only explicit failing/non-passing rows
    """
    program = commit_data.get("program") or {}
    program_tests = (program.get("tests") or {}) or {}

    outcome_counts = Counter()
    failing: List[Tuple[str, str, Dict[str, Any]]] = []

    for tid, td in program_tests.items():
        outcome = normalize_program_test_outcome(td)
        outcome_counts[outcome or "<missing>"] += 1

        if is_program_failure_outcome(outcome):
            failing.append((tid, outcome or "<missing>", td))

    print(f"Original program exitstatus    : {program.get('exitstatus')}")
    print(f"Original program collection_ok : {program.get('collection_ok')}")
    print(f"Original program test rows     : {len(program_tests)}")
    print(f"Original program outcome counts: {dict(sorted(outcome_counts.items()))}")
    print(f"Original failing test rows     : {len(failing)}")

    if not failing:
        print("\nNo failing test rows found for the original program at this commit.")
        if program.get("exitstatus") not in (None, 0):
            print("Note: exitstatus is non-zero, so there may still be a session-level failure not attached to a test row.")
        if program.get("collection_ok") is False:
            print("Note: collection_ok is False, so collection/setup may have failed even if no per-test failing row is present.")
        return

    print(f"\nORIGINAL PROGRAM FAILURES (up to {max_tests}):")
    for tid, outcome, td in failing[:max_tests]:
        msg = clip_msg(td.get("raw_short_msg"))
        print(f"\n- Test: {tid}")
        print(f"  outcome: {outcome}")
        print(f"  raw_short_msg: {msg!r}" if msg else "  raw_short_msg: (null/empty)")

    if len(failing) > max_tests:
        print(f"\n(+ {len(failing) - max_tests} more failing tests not shown)")


def explain_mutant_at_commit(commit_data: Dict[str, Any], mid: str, max_tests: int = 10) -> None:
    program_tests = ((commit_data.get("program") or {}).get("tests")) or {}
    md = ((commit_data.get("mutants") or {}).get(mid)) or {}
    mtests = (md.get("tests") or {}) or {}

    agg = mutant_status_from_tests(mtests)
    counts = summarize_status_counts(mtests)

    print(f"Mutant aggregated status: {agg}")
    print(f"Per-status counts       : {counts}")
    print(f"Mutant test rows        : {len(mtests)}")
    print(f"Program test rows       : {len(program_tests)}")

    relevant = []
    for tid, td in mtests.items():
        st = (td.get("status") or "unknown").lower()
        if st in ("killed", "incompetent", "anomalous"):
            relevant.append((tid, st, td))

    if not relevant:
        print("\nNo killed/incompetent/anomalous test rows for this commit.")
        return

    print(f"\nDETAILS (up to {max_tests} relevant tests):")
    for tid, st, mtd in relevant[:max_tests]:
        ptd = program_tests.get(tid) or {}
        pm = clip_msg(ptd.get("raw_short_msg"))
        mm = clip_msg(mtd.get("raw_short_msg"))

        print(f"\n- Test: {tid}")
        print(f"  status: {st}")
        print(f"  program raw_short_msg: {pm!r}" if pm else "  program raw_short_msg: (null/empty)")
        print(f"  mutant  raw_short_msg: {mm!r}" if mm else "  mutant  raw_short_msg: (null/empty)")


# ================== VISUAL REVIEWER ==================
class MutantReviewer:
    def __init__(
        self,
        *,
        project: str,
        bug: str,
        results_dir: Path,
        commits_chronological: List[str],
        display_commits: List[str],
        candidates: List[Dict[str, Any]],
        problematic_info: Dict[str, Dict[str, Any]],
        commit_cache: Dict[str, Dict[str, Any]],
        commit_evolution: Dict[str, Dict[str, Any]],
        mutant_history: Optional[Dict[str, Any]],
        repo_root: Optional[Path],
        save_file: Path,
        already_saved: Set[str],
        saved_records: List[Dict[str, Any]],
        bug_patch_text: Optional[str],
        effective_head_commit: str,
    ) -> None:
        self.project = project
        self.bug = bug
        self.results_dir = results_dir
        self.commits_chronological = commits_chronological
        self.display_commits = display_commits
        self.candidates = candidates
        self.problematic_info = problematic_info
        self.commit_cache = commit_cache
        self.commit_evolution = commit_evolution
        self.mutant_history = mutant_history
        self.repo_root = repo_root
        self.save_file = save_file
        self.saved_mutants = set(already_saved)
        self.saved_records = list(saved_records)
        self.bug_patch_text = bug_patch_text
        self.effective_head_commit = effective_head_commit

        self.idx = 0
        self.selected_point_idx = 0
        self.selected_commit: Optional[str] = None

        self.save_dialog_fig = None
        self.save_dialog_textbox = None
        self.save_dialog_checks = None
        self.save_dialog_candidate_mid: Optional[str] = None

        self.fig, self.ax = plt.subplots(figsize=(16, 3.8), dpi=180)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.cmap = ListedColormap([
            "#bdbdbd",  # unknown
            "#2ca02c",  # survived
            "#d62728",  # killed
            "#ff7f0e",  # incompetent
            "#9467bd",  # anomalous
            "#ffffff",  # not expected
            "#808080",  # problematic
            "#d9d9d9",  # missing
        ])
        self.norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), self.cmap.N)

    def current(self) -> Optional[Dict[str, Any]]:
        if not self.candidates:
            return None
        self.idx = max(0, min(self.idx, len(self.candidates) - 1))
        cand = self.candidates[self.idx]
        points = cand["interesting_points"]
        if points:
            self.selected_point_idx = max(0, min(self.selected_point_idx, len(points) - 1))
        else:
            self.selected_point_idx = 0
        return cand

    def print_separator(self, title: str) -> None:
        bar = "=" * 140
        print(f"\n{bar}\n{title}\n{bar}")

    def show_original_failures_on_effective_head(self) -> None:
        commit = self.effective_head_commit
        data = self.commit_cache.get(commit)

        self.print_separator(
            f"ORIGINAL PROGRAM FAILURES | newest available commit={commit} ({commit[:8]})"
        )

        if data is None:
            print("Unavailable: no cached results for effective head commit.")
            return

        explain_program_failures(data, max_tests=MAX_PROGRAM_FAILURES_TO_PRINT)

    def show_bug_patch(self) -> None:
        if self.bug_patch_text:
            self.print_separator(f"BUG PATCH | {self.project}/{self.bug}")
            print(self.bug_patch_text)
        else:
            print("\nBUG PATCH: unavailable")

    def show_diff_for_selected_commit(self) -> None:
        cand = self.current()
        if cand is None:
            return

        commit = self.selected_commit
        if commit is None:
            points = cand["interesting_points"]
            if not points:
                print("No selected commit and no interesting points.")
                return
            commit = points[self.selected_point_idx]["anchor_commit"]

        self.print_separator(f"UNIFIED DIFF | mutant={cand['mutant_id']} | commit={commit[:8]}")
        show_unified_diff(self.mutant_history, self.repo_root, commit, cand["mutant_id"])

    def show_patch_and_diff_for_selected_commit(self) -> None:
        self.show_original_failures_on_effective_head()
        self.show_bug_patch()
        self.show_diff_for_selected_commit()

    def announce_current_mutant(self) -> None:
        cand = self.current()
        if cand is None:
            return

        points = cand["interesting_points"]
        if points:
            commit = points[self.selected_point_idx]["anchor_commit"]
            self.selected_commit = commit
        else:
            self.selected_commit = None

        self.print_separator(
            f"NEW MUTANT | {self.project}/{self.bug} | "
            f"{cand['mutant_id']} | idx={self.idx + 1}/{len(self.candidates)}"
        )

        if points:
            p = points[self.selected_point_idx]
            print(f"kind       : {p['kind']}")
            print(f"anchor     : {p['anchor_commit']} ({p['anchor_commit'][:8]})")
            if p.get("transition"):
                print(f"transition : {p['transition']}")
            if p.get("from_commit"):
                print(f"from       : {p['from_commit']} ({p['from_commit'][:8]})")
            if p.get("to_commit"):
                print(f"to         : {p['to_commit']} ({p['to_commit'][:8]})")

            self.show_patch_and_diff_for_selected_commit()
        else:
            print("No interesting points for this mutant.")

    def render(self) -> None:
        cand = self.current()
        self.ax.clear()

        if cand is None:
            self.ax.set_title("No candidate mutants.")
            self.fig.canvas.draw_idle()
            return

        row = cand["row_cells"]
        points = cand["interesting_points"]
        mid = cand["mutant_id"]

        row_by_commit = {cell["commit"]: cell["status"] for cell in row}
        vals = np.array(
            [[STATUS_TO_CELL[row_by_commit.get(c, "missing")] for c in self.display_commits]],
            dtype=np.uint8,
        )
        self.ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=self.cmap, norm=self.norm)

        n_cols = len(self.display_commits)
        if n_cols <= MAX_XTICKS:
            xt = np.arange(n_cols)
        else:
            xt = np.linspace(0, n_cols - 1, MAX_XTICKS, dtype=int)

        self.ax.set_xticks(xt)
        self.ax.set_xticklabels([self.display_commits[i][:8] for i in xt], rotation=90, fontsize=7)
        self.ax.set_yticks([0])
        self.ax.set_yticklabels([""])
        self.ax.set_xlabel("Commits (oldest -> newest, filtered display)")

        for i, p in enumerate(points):
            anchor = p["anchor_commit"]
            if anchor not in self.display_commits:
                continue
            x = self.display_commits.index(anchor)
            marker = "*" if p["kind"] != "always_survived" else "D"
            size = 180 if i == self.selected_point_idx else 90
            edge = "black" if i == self.selected_point_idx else "white"
            self.ax.scatter([x], [0], marker=marker, s=size, edgecolors=edge, linewidths=1.5, zorder=5)

        if self.selected_commit and self.selected_commit in self.display_commits:
            xc = self.display_commits.index(self.selected_commit)
            self.ax.axvline(x=xc, color="black", linestyle="--", linewidth=1.0, zorder=4)

        current_point = points[self.selected_point_idx] if points else None
        point_text = ""
        if current_point:
            if current_point["kind"] == "always_survived":
                point_text = (
                    f"selected point: always_survived "
                    f"[{current_point['anchor_commit'][:8]} .. {current_point['anchor_commit_to'][:8]}]"
                )
            else:
                point_text = (
                    f"selected point: {current_point['transition']} at {current_point['anchor_commit'][:8]}"
                )

        clicked_text = ""
        if self.selected_commit:
            clicked_text = f" | selected commit: {self.selected_commit[:8]}"

        saved_mark = " [SAVED]" if mid in self.saved_mutants else ""
        title = (
            f"{self.project}/{self.bug} | mutant {self.idx + 1}/{len(self.candidates)} | "
            f"{mid}{saved_mark} | score={cand['score']} | points={len(points)}{clicked_text}"
        )
        subtitle = (
            "keys: left/right=mutant  up/down=point  enter=inspect point  click=inspect commit  "
            "d=diff  p=patch  f=orig-failures  a=patch+diff+failures  s=save  x=skip  q=quit"
        )

        self.ax.set_title(title + "\n" + point_text + "\n" + subtitle, fontsize=10)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def inspect_commit(self, commit: str) -> None:
        cand = self.current()
        if cand is None:
            return

        mid = cand["mutant_id"]
        row_map = {cell["commit"]: cell["status"] for cell in cand["row_cells"]}

        print("\n" + "=" * 120)
        print(f"MUTANT: {mid}")
        print(f"COMMIT: {commit} ({commit[:8]})")
        print(f"ROW STATUS: {row_map.get(commit, 'missing')}")

        print_commit_evolution_info(commit, self.commit_evolution)

        results_path = self.results_dir / f"results_{commit}.json"
        print(f"\nResults file: {results_path}")
        print(f"Open with   : {editor_hint(results_path)}")

        data = self.commit_cache.get(commit)
        if data is not None:
            print(f"\nCOMMIT DETAILS: {commit} ({commit[:8]})")
            explain_mutant_at_commit(data, mid, max_tests=MAX_RELEVANT_TESTS_TO_PRINT)
        else:
            print("\nCOMMIT DETAILS: unavailable")

    def inspect_selected_point(self) -> None:
        cand = self.current()
        if cand is None:
            return
        points = cand["interesting_points"]
        if not points:
            print("No interesting points for current mutant.")
            return

        p = points[self.selected_point_idx]
        anchor_commit = p["anchor_commit"]
        mid = cand["mutant_id"]

        print("\n" + "=" * 120)
        print(f"MUTANT: {mid}")
        print(f"POINT : {self.selected_point_idx + 1}/{len(points)}")
        print(f"KIND  : {p['kind']}")
        if p.get("transition"):
            print(f"TRANSITION: {p['transition']}")
        if p.get("from_commit"):
            print(f"FROM  : {p['from_commit']} ({p['from_commit'][:8]})")
        if p.get("to_commit"):
            print(f"TO    : {p['to_commit']} ({p['to_commit'][:8]})")
        if p["kind"] == "always_survived":
            print(f"RANGE : {p['anchor_commit'][:8]} -> {p['anchor_commit_to'][:8]}")

        self.selected_commit = anchor_commit
        self.render()
        self.inspect_commit(anchor_commit)

    def save_current(self, note: str, labels: List[str]) -> None:
        cand = self.current()
        if cand is None:
            return

        mid = cand["mutant_id"]

        if mid in self.saved_mutants:
            print(f"Already saved: {mid}")
            return

        rec: Dict[str, Any] = {
            "project": self.project,
            "bug": self.bug,
            "mutant_id": mid,
            "labels": labels,
            "note": note,
        }

        self.saved_records.append(rec)
        write_saved_records(self.save_file, self.saved_records)

        self.saved_mutants.add(mid)
        print(f"Saved: {mid} labels={labels} note={note!r}")
        print(f"Save file: {self.save_file}")

        self.next_mutant()

    def _close_save_dialog(self) -> None:
        if self.save_dialog_fig is not None:
            try:
                plt.close(self.save_dialog_fig)
            except Exception:
                pass
        self.save_dialog_fig = None
        self.save_dialog_textbox = None
        self.save_dialog_checks = None
        self.save_dialog_candidate_mid = None

    def open_save_dialog(self) -> None:
        cand = self.current()
        if cand is None:
            return

        mid = cand["mutant_id"]
        if mid in self.saved_mutants:
            print(f"Already saved: {mid}")
            return

        self._close_save_dialog()

        self.save_dialog_fig = plt.figure(figsize=(8.8, 5.0))
        self.save_dialog_fig.suptitle(f"Save mutant {mid}", fontsize=12)
        self.save_dialog_candidate_mid = mid

        ax_checks = self.save_dialog_fig.add_axes([0.08, 0.34, 0.58, 0.48])
        ax_text = self.save_dialog_fig.add_axes([0.08, 0.18, 0.84, 0.08])
        ax_save = self.save_dialog_fig.add_axes([0.22, 0.05, 0.22, 0.08])
        ax_cancel = self.save_dialog_fig.add_axes([0.56, 0.05, 0.22, 0.08])

        labels_ui = [label for _key, label in SAVE_REASON_OPTIONS]
        self.save_dialog_checks = CheckButtons(ax_checks, labels_ui, [False] * len(SAVE_REASON_OPTIONS))
        self.save_dialog_textbox = TextBox(ax_text, "Note ", initial="")

        btn_save = Button(ax_save, "Save")
        btn_cancel = Button(ax_cancel, "Cancel")

        info_text = (
            "Select one or more reasons, optionally type a note, then press Enter or click Save."
        )
        self.save_dialog_fig.text(0.08, 0.88, info_text, fontsize=9)

        def do_save(_event=None) -> None:
            current = self.current()
            if current is None:
                self._close_save_dialog()
                return

            if current["mutant_id"] != self.save_dialog_candidate_mid:
                print("Current mutant changed while dialog was open; cancelling dialog save.")
                self._close_save_dialog()
                return

            note = ""
            if self.save_dialog_textbox is not None:
                note = self.save_dialog_textbox.text.rstrip()

            selected_labels: List[str] = []
            if self.save_dialog_checks is not None:
                statuses = list(self.save_dialog_checks.get_status())
                for checked, (key, _ui_label) in zip(statuses, SAVE_REASON_OPTIONS):
                    if checked:
                        selected_labels.append(key)

            self._close_save_dialog()
            self.save_current(note=note, labels=selected_labels)

        def do_cancel(_event=None) -> None:
            self._close_save_dialog()
            print("Cancelled save dialog.")

        btn_save.on_clicked(do_save)
        btn_cancel.on_clicked(do_cancel)

        self.save_dialog_fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: do_save() if event.key == "enter"
            else (do_cancel() if event.key == "escape" else None),
        )

        plt.show(block=False)

    def next_mutant(self) -> None:
        if not self.candidates:
            return
        old_idx = self.idx
        self.idx = min(self.idx + 1, len(self.candidates) - 1)
        self.selected_point_idx = 0
        self.selected_commit = None
        self.render()
        if self.idx != old_idx:
            self.announce_current_mutant()

    def prev_mutant(self) -> None:
        if not self.candidates:
            return
        old_idx = self.idx
        self.idx = max(self.idx - 1, 0)
        self.selected_point_idx = 0
        self.selected_commit = None
        self.render()
        if self.idx != old_idx:
            self.announce_current_mutant()

    def next_point(self) -> None:
        cand = self.current()
        if cand is None or not cand["interesting_points"]:
            return
        self.selected_point_idx = min(self.selected_point_idx + 1, len(cand["interesting_points"]) - 1)
        self.render()

    def prev_point(self) -> None:
        cand = self.current()
        if cand is None or not cand["interesting_points"]:
            return
        self.selected_point_idx = max(self.selected_point_idx - 1, 0)
        self.render()

    def on_key(self, event) -> None:
        if event.key == "right":
            self.next_mutant()
        elif event.key == "left":
            self.prev_mutant()
        elif event.key == "down":
            self.next_point()
        elif event.key == "up":
            self.prev_point()
        elif event.key == "enter":
            self.inspect_selected_point()
        elif event.key == "d":
            self.show_diff_for_selected_commit()
        elif event.key == "p":
            self.show_bug_patch()
        elif event.key == "f":
            self.show_original_failures_on_effective_head()
        elif event.key == "a":
            self.show_patch_and_diff_for_selected_commit()
        elif event.key == "s":
            self.open_save_dialog()
        elif event.key == "x":
            self.next_mutant()
        elif event.key == "q":
            plt.close(self.fig)

    def on_click(self, event) -> None:
        cand = self.current()
        if cand is None or event.inaxes != self.ax or event.xdata is None:
            return

        x = int(round(event.xdata))
        if not (0 <= x < len(self.display_commits)):
            return

        clicked_commit = self.display_commits[x]
        self.selected_commit = clicked_commit

        points = cand["interesting_points"]
        if points:
            point_cols = []
            for i, p in enumerate(points):
                if p["anchor_commit"] in self.display_commits:
                    col = self.display_commits.index(p["anchor_commit"])
                    point_cols.append((abs(col - x), i))
            if point_cols:
                point_cols.sort()
                self.selected_point_idx = point_cols[0][1]

        self.render()
        self.inspect_commit(clicked_commit)

    def run(self) -> None:
        self.render()
        self.announce_current_mutant()
        plt.show()


# ================== MAIN ==================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Visual mutant reviewer: one mutant row at a time, save mutants once. "
            "By default, project/bug and related paths are inferred from --base-dir."
        )
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Example: /home/rooti/shared/LatentMu/data/data_from_hpc/tqdm-8b",
    )
    ap.add_argument("--project", default=None, help="Optional override; usually inferred from base-dir.")
    ap.add_argument("--bug", default=None, help="Optional override; usually inferred from base-dir.")
    ap.add_argument("--save-file", type=Path, default=None, help="Default: <base-dir>/saved_mutants.json")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--mutant-history", type=Path, default=None)
    ap.add_argument("--commit-evolution-file", type=Path, default=None)
    ap.add_argument("--bug-patch-file", type=Path, default=None)
    args = ap.parse_args()

    resolved = derive_default_paths(
        base_dir=args.base_dir,
        project=args.project,
        bug=args.bug,
        save_file=args.save_file,
        mutant_history=args.mutant_history,
        commit_evolution_file=args.commit_evolution_file,
        bug_patch_file=args.bug_patch_file,
    )

    project: str = resolved["project"]
    bug: str = resolved["bug"]
    commits_file: Path = resolved["commits_file"]
    results_dir: Path = resolved["results_dir"]
    mutant_history_path: Optional[Path] = resolved["mutant_history"]
    commit_evolution_file: Optional[Path] = resolved["commit_evolution_file"]
    bug_patch_file: Optional[Path] = resolved["bug_patch_file"]
    save_file: Path = resolved["save_file"]

    print(f"project             : {project}")
    print(f"bug                 : {bug}")
    print(f"base_dir            : {args.base_dir}")
    print(f"commits_file        : {commits_file}")
    print(f"results_dir         : {results_dir}")
    print(f"mutant_history      : {mutant_history_path}")
    print(f"commit_evolution    : {commit_evolution_file}")
    print(f"bug_patch           : {bug_patch_file}")
    print(f"save_file           : {save_file}")

    bug_patch_text = maybe_read_text(bug_patch_file)

    if not commits_file.exists():
        raise SystemExit(f"Missing commits file: {commits_file}")
    if not results_dir.exists():
        raise SystemExit(f"Missing results dir: {results_dir}")

    commits = read_commits_file(commits_file)  # newest -> oldest
    existing_result_paths, missing_commits = resolve_existing_result_paths(commits, results_dir)

    if not existing_result_paths:
        raise SystemExit(f"No results_*.json files found in {results_dir}")

    head = next(c for c in commits if c in existing_result_paths)
    print(f"Using effective head: {head} ({head[:8]})")
    print(f"Effective head path: {existing_result_paths[head]}")

    if missing_commits:
        print(f"Missing {len(missing_commits)} result file(s) in with_lbl; continuing anyway.")
        for c in missing_commits[:10]:
            print(f"  - {results_dir / f'results_{c}.json'}")
        if len(missing_commits) > 10:
            print(f"  ... and {len(missing_commits) - 10} more")

    commit_cache = build_commit_cache(existing_result_paths)
    mutant_ids = collect_mutant_ids(commit_cache)
    if not mutant_ids:
        raise SystemExit(f"No mutants found in any existing result file in {results_dir}")

    mutant_history = maybe_load_json(mutant_history_path)
    commit_evolution = load_commit_evolution_map(commit_evolution_file)

    status_map = build_status_map(mutant_ids, commits, commit_cache)

    if mutant_history is not None:
        print(f"Using mutant history: {mutant_history_path}")
        expected_map = compute_expected_from_history(mutant_ids, commits, mutant_history)
    else:
        print("mutant_history.json not found; falling back to expected-from-presence.")
        expected_map = compute_expected_from_presence(mutant_ids, commits, commit_cache)

    commits_chronological = list(reversed(commits))  # oldest -> newest

    problematic_info = detect_problematic_commits(
        commits_chronological=commits_chronological,
        mutant_ids=mutant_ids,
        expected_map=expected_map,
        status_map=status_map,
    )

    display_commits = compute_display_commits(
        commits_chronological=commits_chronological,
        mutant_ids=mutant_ids,
        expected_map=expected_map,
        status_map=status_map,
    )
    if not display_commits:
        print("All columns were killed/unknown only; falling back to full chronological display.")
        display_commits = commits_chronological
    else:
        print(f"Displaying {len(display_commits)}/{len(commits_chronological)} commits after filtering.")

    saved_records = load_saved_records(save_file)
    saved_mutants = {r.get("mutant_id") for r in saved_records if isinstance(r, dict) and r.get("mutant_id")}
    if saved_mutants:
        print(f"Already saved mutants in output JSON: {len(saved_mutants)}")

    candidates = collect_mutant_candidates(
        mutant_ids=mutant_ids,
        commits_chronological=commits_chronological,
        expected_map=expected_map,
        status_map=status_map,
        problematic_info=problematic_info,
        saved_mutants=saved_mutants,
    )

    print(f"Candidate mutants to review: {len(candidates)}")
    if bug_patch_text:
        print(f"Loaded bug patch from: {bug_patch_file}")
    else:
        print(f"Bug patch not found: {bug_patch_file}")

    reviewer = MutantReviewer(
        project=project,
        bug=bug,
        results_dir=results_dir,
        commits_chronological=commits_chronological,
        display_commits=display_commits,
        candidates=candidates,
        problematic_info=problematic_info,
        commit_cache=commit_cache,
        commit_evolution=commit_evolution,
        mutant_history=mutant_history,
        repo_root=args.repo_root,
        save_file=save_file,
        already_saved=saved_mutants,
        saved_records=saved_records,
        bug_patch_text=bug_patch_text,
        effective_head_commit=head,
    )
    reviewer.run()


if __name__ == "__main__":
    main()