from __future__ import annotations

import argparse
import json
import ast
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from git import Repo
from gumtree_mutant_mapper import map_mutant  # your GumTree-based mapper


# --------------------------------------------------------------------
# Helper functions for locations / pretty-printing
# --------------------------------------------------------------------

def offset_to_line_col(src: str, offset: int) -> tuple[int, int]:
    """
    Convert a 0-based character offset into (line_number, column_number),
    both 1-based. Lines are split on '\n'.
    """
    if offset < 0:
        offset = 0
    if offset > len(src):
        offset = len(src)

    line = 1
    col = 1
    i = 0
    while i < offset:
        if src[i] == '\n':
            line += 1
            col = 1
        else:
            col += 1
        i += 1
    return line, col


def get_line(src: str, line_no: int) -> str:
    """Return line_no (1-based) from src, without trailing newline; or '' if out of range."""
    lines = src.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def print_mutation_view(
    label: str,
    src: str,
    start: int,
    end: int,
    path: str,
    commit_hex: str | None,
    context_lines: int = 2,
):
    """
    Pretty-print the mutation location in `src`.
    """
    print(f"\n--- {label} ---")
    print(f"file: {path}")
    if commit_hex is not None:
        print(f"commit: {commit_hex}")
    print(f"coords: start={start}, end={end}")

    if start is None or end is None:
        print("(no coordinates)")
        print(f"--- end of {label} ---\n")
        return

    start = max(0, min(start, len(src)))
    end = max(start, min(end, len(src)))

    start_line, start_col = offset_to_line_col(src, start)
    line_no = start_line

    line_start = max(1, line_no - context_lines)
    line_end = line_no + context_lines

    lines = src.splitlines()
    line_end = min(line_end, len(lines))

    for ln in range(line_start, line_end + 1):
        line_text = get_line(src, ln)
        print(f"{ln:5d} | {line_text}")
        if ln == line_no:
            underline_len = max(1, min(len(line_text) - (start_col - 1), (end - start)))
            if underline_len <= 0:
                underline_len = 1
            prefix_spaces = " " * (start_col - 1)
            carets = "^" * underline_len
            print(f"      | {prefix_spaces}{carets} MUTATED REGION")

    print(f"--- end of {label} ---\n")


def apply_mutant(src: str, start: int, end: int, mutant_value: str) -> str:
    """Return mutated src by replacing [start:end] with mutant_value."""
    return src[:start] + mutant_value + src[end:]


def is_parseable_python(src: str) -> bool:
    """Return True if src parses as Python, False otherwise."""
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def deactivate_mutant(state: dict, reason: str | None):
    """Mark a mutant as inactive and store a (simple) reason if provided."""
    state["active"] = False
    state["death_reason"] = reason


# --------------------------------------------------------------------
# Git helper functions
# --------------------------------------------------------------------

def get_blob(commit, path: str):
    """Return the blob for `path` in `commit`, or None if it doesn't exist."""
    try:
        return commit.tree / path
    except KeyError:
        return None


def get_file_content(commit, path: str) -> str | None:
    """Return file content in `commit` as a string, or None if it doesn't exist."""
    blob = get_blob(commit, path)
    if blob is None:
        return None
    data = blob.data_stream.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


# --------------------------------------------------------------------
# CLI argument parsing
# --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track Python mutants backwards through Git history."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the Git repository root.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="HEAD",
        help="Branch or commit to start from (default: HEAD).",
    )

    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--days",
        type=int,
        default=365,
        help="Look back this many days from the branch tip (default: 365).",
    )
    cutoff_group.add_argument(
        "--since",
        type=str,
        help=(
            "Only consider commits after this ISO date/time "
            "(e.g. '2023-01-01' or '2023-01-01T12:00:00')."
        ),
    )

    parser.add_argument(
        "--mutant-json",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSON files containing mutants.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mutant_history.json",
        help="Output JSON file for mutant history (default: mutant_history.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mapping output from GumTree mapper.",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="Number of parallel jobs (processes) to use (default: 1).",
    )

    return parser.parse_args()


# --------------------------------------------------------------------
# Batching + worker
# --------------------------------------------------------------------

def partition_even(items: List[Dict[str, Any]], njobs: int) -> List[List[Dict[str, Any]]]:
    """
    Split items into njobs batches with sizes differing by at most 1.
    """
    njobs = max(1, int(njobs))
    n = len(items)
    if njobs == 1 or n == 0:
        return [items]

    base = n // njobs
    rem = n % njobs
    batches: List[List[Dict[str, Any]]] = []
    start = 0
    for i in range(njobs):
        size = base + (1 if i < rem else 0)
        end = start + size
        if size > 0:
            batches.append(items[start:end])
        start = end
    return batches


def track_mutants_batch(
    *,
    batch_mutants: List[Dict[str, Any]],
    repo_path_str: str,
    branch: str,
    cutoff_iso: str,
    verbose: bool,
) -> Dict[str, dict]:
    """
    Worker: tracks only the mutants in batch_mutants, returns mutant_states dict.
    JSON-serializable inputs only (for ProcessPool).
    """
    repo_path = Path(repo_path_str)
    repo = Repo(repo_path)

    head_commit = repo.commit(branch)
    cutoff = datetime.fromisoformat(cutoff_iso)

    # Per-mutant tracking state
    mutant_states: dict[str, dict] = {}

    for m in batch_mutants:
        local_id = m["mutant_id"]
        path = m["file"]
        uid = f"{path}#{local_id}"

        src_head = get_file_content(head_commit, path)
        if src_head is None:
            mutant_states[uid] = {
                "info": m,
                "file": path,
                "active": False,
                "history": [],
                "last_commit": None,
                "current_start": None,
                "current_end": None,
                "death_reason": None,
            }
            continue

        s = m["mutation_start_pos"]
        e = m["mutation_end_pos"]

        mutant_states[uid] = {
            "info": m,
            "file": path,
            "active": True,
            "current_start": s,
            "current_end": e,
            "last_commit": head_commit.hexsha,
            "history": [
                {
                    "commit": head_commit.hexsha,
                    "date": head_commit.committed_datetime.isoformat(),
                    "start_pos": s,
                    "end_pos": e,
                }
            ],
            "death_reason": None,
        }

    # Initial pruning: drop mutants that are already unparsable at HEAD
    for uid, state in mutant_states.items():
        if not state["active"]:
            continue

        info = state["info"]
        path = state["file"]
        src_head = get_file_content(head_commit, path)
        if src_head is None:
            deactivate_mutant(state, reason=None)
            continue

        s = info["mutation_start_pos"]
        e = info["mutation_end_pos"]
        mutated_src_head = apply_mutant(src_head, s, e, info["mutant_value"])

        if not is_parseable_python(mutated_src_head):
            deactivate_mutant(state, reason="unparseable")

    # Walk history: HEAD → earlier commits
    previous = head_commit

    for i, commit in enumerate(repo.iter_commits(rev=head_commit.hexsha)):
        if i == 0:
            continue

        if commit.committed_datetime < cutoff:
            break

        active_uids = [uid for uid, st in mutant_states.items() if st["active"]]
        if not active_uids:
            break

        older = commit
        newer = previous

        # Precompute which files changed and cache contents
        active_files = {mutant_states[uid]["file"] for uid in active_uids}

        file_changed: dict[str, bool] = {}
        src_newer_cache: dict[str, str | None] = {}
        src_older_cache: dict[str, str | None] = {}

        for path in active_files:
            prev_blob = get_blob(newer, path)
            older_blob = get_blob(older, path)

            if prev_blob is None or older_blob is None:
                file_changed[path] = True
                src_newer_cache[path] = get_file_content(newer, path)
                src_older_cache[path] = get_file_content(older, path)
                continue

            file_changed[path] = (prev_blob.hexsha != older_blob.hexsha)
            src_newer_cache[path] = get_file_content(newer, path)
            src_older_cache[path] = get_file_content(older, path)

        for uid in active_uids:
            state = mutant_states[uid]
            path = state["file"]
            info = state["info"]
            local_id = info["mutant_id"]

            src_newer = src_newer_cache.get(path)
            src_older = src_older_cache.get(path)

            if src_newer is None or src_older is None:
                deactivate_mutant(state, reason=None)
                continue

            if not file_changed[path]:
                state["last_commit"] = older.hexsha
                state["history"].append(
                    {
                        "commit": older.hexsha,
                        "date": older.committed_datetime.isoformat(),
                        "start_pos": state["current_start"],
                        "end_pos": state["current_end"],
                    }
                )
                continue

            mu_step = {
                "mutant_id": local_id,
                "file": path,
                "mutation_start_pos": state["current_start"],
                "mutation_end_pos": state["current_end"],
                "mutant_value": info["mutant_value"],
            }

            mapped_mutant, new_start_pos, new_end_pos = map_mutant(
                src_from=src_newer,
                src_to=src_older,
                mu_info_from=mu_step,
                lang=".py",
                verbose=verbose,
            )

            if mapped_mutant is None:
                deactivate_mutant(state, reason="no_mapping")
                continue

            mutated_src_older = apply_mutant(
                src_older,
                new_start_pos,
                new_end_pos,
                info["mutant_value"],
            )

            if not is_parseable_python(mutated_src_older):
                deactivate_mutant(state, reason="unparseable")
                continue

            state["current_start"] = new_start_pos
            state["current_end"] = new_end_pos
            state["last_commit"] = older.hexsha
            state["history"].append(
                {
                    "commit": older.hexsha,
                    "date": older.committed_datetime.isoformat(),
                    "start_pos": new_start_pos,
                    "end_pos": new_end_pos,
                }
            )

        previous = older

    return mutant_states


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    repo_path = Path(args.repo)
    repo = Repo(repo_path)

    head_commit = repo.commit(args.branch)
    head_time = head_commit.committed_datetime

    if args.since:
        cutoff = datetime.fromisoformat(args.since)
    else:
        cutoff = head_time - timedelta(days=args.days)

    print("HEAD:", head_commit.hexsha, head_time)
    print("Cutoff:", cutoff)
    print("-" * 80)

    # Load mutants
    all_mutants: list[Dict[str, Any]] = []
    for json_path in args.mutant_json:
        p = Path(json_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        all_mutants.extend(data)

    print(f"Loaded {len(all_mutants)} mutants from {len(args.mutant_json)} JSON files.")

    njobs = max(1, int(args.njobs))
    batches = partition_even(all_mutants, njobs)
    if njobs > 1:
        print(f"Partitioned into {len(batches)} batches for --njobs {njobs}:",
              [len(b) for b in batches])

    # Run (parallel if njobs > 1)
    mutant_states: dict[str, dict] = {}

    if njobs == 1 or len(batches) == 1:
        mutant_states = track_mutants_batch(
            batch_mutants=batches[0],
            repo_path_str=str(repo_path),
            branch=args.branch,
            cutoff_iso=cutoff.isoformat(),
            verbose=args.verbose,
        )
    else:
        with ProcessPoolExecutor(max_workers=njobs) as ex:
            futures = []
            for bi, batch in enumerate(batches):
                futures.append(
                    ex.submit(
                        track_mutants_batch,
                        batch_mutants=batch,
                        repo_path_str=str(repo_path),
                        branch=args.branch,
                        cutoff_iso=cutoff.isoformat(),
                        verbose=args.verbose,
                    )
                )

            for fut in as_completed(futures):
                batch_states = fut.result()
                # Merge: uid keys are globally unique (file#mutant_id)
                for uid, st in batch_states.items():
                    if uid in mutant_states:
                        raise RuntimeError(f"Duplicate uid during merge: {uid}")
                    mutant_states[uid] = st

    # ----------------------------------------------------------------
    # Summary / saving results (same as before)
    # ----------------------------------------------------------------
    tracked = sum(1 for s in mutant_states.values() if s["history"])
    print(f"\nTracked {tracked} mutants with at least one commit.")

    reason_counts: dict[str, int] = {}
    for uid, state in mutant_states.items():
        reason = state.get("death_reason")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("Death reasons (only 'unparseable' and 'no_mapping' are tracked):")
    for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {reason}: {count}")

    for uid, state in mutant_states.items():
        h = state["history"]
        if not h:
            continue
        first = h[0]
        last = h[-1]
        local_id = state["info"]["mutant_id"]
        path = state["file"]
        death_reason = state.get("death_reason")

        line = (
            f"Mutant {uid} (file={path}, local_id={local_id}) tracked from "
            f"{first['commit']} @ {first['date']} "
            f"back to {last['commit']} @ {last['date']} "
            f"({len(h)} commits)."
        )
        if death_reason:
            line += f" Stopped because: {death_reason}."

        print(line)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mutant_states, f, indent=2)

    print(f"\nSaved mutant history to {output_path}")


if __name__ == "__main__":
    main()
