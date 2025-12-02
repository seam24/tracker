from __future__ import annotations

import argparse
import json
import ast
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

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
    Pretty-print the mutation location in `src`:

    - file + commit
    - coords
    - a few lines around the mutation
    - underline the mutated region with ^^^
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

    # Normalise
    start = max(0, min(start, len(src)))
    end = max(start, min(end, len(src)))

    start_line, start_col = offset_to_line_col(src, start)

    # We highlight only the first line for simplicity.
    line_no = start_line

    # Context range
    line_start = max(1, line_no - context_lines)
    line_end = line_no + context_lines

    lines = src.splitlines()
    line_end = min(line_end, len(lines))

    # Print context lines
    for ln in range(line_start, line_end + 1):
        line_text = get_line(src, ln)
        print(f"{ln:5d} | {line_text}")
        if ln == line_no:
            # Build underline: spaces up to start_col-1, then carets for (end-start) chars or to end of line
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
    return data  # already str


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

    # Cutoff selection: either relative (--days) or absolute (--since)
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
        help="One or more JSON files containing mutants (same format as mu_utils.json, etc.).",
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

    return parser.parse_args()


# --------------------------------------------------------------------
# Core logic wrapped in a function using args
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

    # ----------------------------------------------------------------
    # Load mutants (assumed to be for the chosen HEAD/branch)
    # ----------------------------------------------------------------

    all_mutants: list[Dict[str, Any]] = []
    for json_path in args.mutant_json:
        p = Path(json_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        all_mutants.extend(data)

    print(f"Loaded {len(all_mutants)} mutants from {len(args.mutant_json)} JSON files.")

    # ----------------------------------------------------------------
    # Per-mutant tracking state
    # ----------------------------------------------------------------

    mutant_states: dict[str, dict] = {}  # uid -> state

    for m in all_mutants:
        local_id = m["mutant_id"]   # id within the file
        path = m["file"]
        uid = f"{path}#{local_id}"  # global unique id for this mutant

        # Baseline content at HEAD / branch tip
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

    print(f"Initialized tracking state for {len(mutant_states)} mutants.")

    # ----------------------------------------------------------------
    # Initial pruning: drop mutants that are already unparsable at HEAD
    # ----------------------------------------------------------------

    initial_dropped = 0

    for uid, state in mutant_states.items():
        if not state["active"]:
            continue

        info = state["info"]
        path = state["file"]
        src_head = get_file_content(head_commit, path)
        if src_head is None:
            deactivate_mutant(state, reason=None)
            initial_dropped += 1
            continue

        s = info["mutation_start_pos"]
        e = info["mutation_end_pos"]
        mutated_src_head = apply_mutant(src_head, s, e, info["mutant_value"])

        if not is_parseable_python(mutated_src_head):
            deactivate_mutant(state, reason="unparseable")
            initial_dropped += 1

    print(f"Initial pruning: dropped {initial_dropped} mutants that are unparsable at HEAD.")

    # ----------------------------------------------------------------
    # Walk history: HEAD (baseline) → earlier commits
    # ----------------------------------------------------------------

    previous = head_commit  # this is the 'newer' side in each pair

    for i, commit in enumerate(repo.iter_commits(rev=head_commit.hexsha)):  # newest → oldest
        if i == 0:
            # first commit produced by iter_commits is HEAD itself
            continue

        if commit.committed_datetime < cutoff:
            break

        # determine which mutants are still active before doing any work
        active_uids = [uid for uid, st in mutant_states.items() if st["active"]]
        if not active_uids:
            print("\nNo more active mutants to propagate; stopping history walk.")
            break

        older = commit      # older commit
        newer = previous    # newer commit

        print("\n===========================================================")
        print(f"Mapping from newer {newer.hexsha} -> older {older.hexsha}")
        print(f"Active mutants this step: {len(active_uids)}")
        print("===========================================================")

        # --- precompute which files changed and cache contents ---
        active_files = {mutant_states[uid]["file"] for uid in active_uids}

        file_changed: dict[str, bool] = {}
        src_newer_cache: dict[str, str | None] = {}
        src_older_cache: dict[str, str | None] = {}

        for path in active_files:
            prev_blob = get_blob(newer, path)
            older_blob = get_blob(older, path)

            if prev_blob is None or older_blob is None:
                # treat as "changed" so tracking can fail appropriately
                file_changed[path] = True
                src_newer_cache[path] = get_file_content(newer, path)
                src_older_cache[path] = get_file_content(older, path)
                continue

            if prev_blob.hexsha == older_blob.hexsha:
                file_changed[path] = False
            else:
                file_changed[path] = True

            # cache content in both cases
            src_newer_cache[path] = get_file_content(newer, path)
            src_older_cache[path] = get_file_content(older, path)

        # For each mutant that is still active, try to map it one step back
        for uid in active_uids:
            state = mutant_states[uid]
            path = state["file"]
            info = state["info"]
            local_id = info["mutant_id"]

            src_newer = src_newer_cache.get(path)
            src_older = src_older_cache.get(path)

            # If file is missing in either commit: stop tracking this mutant (no reason tracked)
            if src_newer is None or src_older is None:
                deactivate_mutant(state, reason=None)
                continue

            if not file_changed[path]:
                # file did not change: mutant trivially propagates with same coords
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

            # --- file changed: need GumTree-based mapping ---

            mu_step = {
                "mutant_id": local_id,  # keep original id for logging
                "file": path,
                "mutation_start_pos": state["current_start"],
                "mutation_end_pos": state["current_end"],
                "mutant_value": info["mutant_value"],
            }

            # We want to map from newer -> older
            mapped_mutant, new_start_pos, new_end_pos = map_mutant(
                src_from=src_newer,
                src_to=src_older,
                mu_info_from=mu_step,
                lang=".py",
                verbose=args.verbose,
            )

            if mapped_mutant is None:
                # cannot propagate further back -> stop tracking
                deactivate_mutant(state, reason="no_mapping")
                continue

            # --- SANITY CHECK: reconstruct and parse on older version ---
            mutated_src_older = apply_mutant(
                src_older,
                new_start_pos,
                new_end_pos,
                info["mutant_value"],
            )

            if not is_parseable_python(mutated_src_older):
                # Debug block you can uncomment if needed:
                # print(
                #     f"[WARN] Mutant {uid} became unparsable at commit {older.hexsha}. "
                #     f"start={new_start_pos}, end={new_end_pos}"
                # )
                # print(info)
                # print_mutation_view(
                #     label="NEWER original (before mapping)",
                #     src=src_newer,
                #     start=state["current_start"],
                #     end=state["current_end"],
                #     path=path,
                #     commit_hex=newer.hexsha,
                #     context_lines=2,
                # )
                # print_mutation_view(
                #     label="OLDER original (before applying mutant_value)",
                #     src=src_older,
                #     start=new_start_pos,
                #     end=new_end_pos,
                #     path=path,
                #     commit_hex=older.hexsha,
                #     context_lines=2,
                # )
                # mutated_start = new_start_pos
                # mutated_end = new_start_pos + len(info["mutant_value"])
                # print_mutation_view(
                #     label="OLDER mutated (after applying mutant_value)",
                #     src=mutated_src_older,
                #     start=mutated_start,
                #     end=mutated_end,
                #     path=path,
                #     commit_hex=older.hexsha,
                #     context_lines=2,
                # )
                deactivate_mutant(state, reason="unparseable")
                continue

            # successfully mapped to `older`
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

        # Move the window one step back
        previous = older

    # ----------------------------------------------------------------
    # Summary / saving results
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
