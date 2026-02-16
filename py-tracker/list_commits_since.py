#!/usr/bin/env python3
"""
List and filter commits along the mainline (first-parent) history of a Git repository.

------------------------------------------------------------

Purpose:
This script is used to obtain a bounded list of commits for downstream analysis.
It provides a simple filtering mechanism based on filenames and glob patterns,
assuming that relevant files (e.g., "fish.py") exist and are consistently named
across the time window of interest.

------------------------------------------------------------

The script walks commits from HEAD backwards following the first-parent chain and
prints those that either:
  - are forced (HEAD and the boundary commit), or
  - modify at least one file whose basename matches:
        * an exact name in --files, or
        * a glob pattern in --patterns.

Changed files are determined per commit using a parent → commit diff.

------------------------------------------------------------
Boundary selection (exactly one required):

--days N
    Stop when reaching commits with date ≤ (HEAD_date − N days).
    The boundary is the first mainline commit at or before this cutoff.

--start <hash>
    Define the stopping point using a commit hash:

    • If the start commit lies on the first-parent chain:
          stop exactly at that commit (inclusive).

    • If the start commit is not on the first-parent chain (e.g., merged branch):
          use the start commit’s timestamp as a cutoff and stop at the closest
          mainline commit whose date is not older than the start date.

------------------------------------------------------------

Matching is performed on basenames only (directory paths are ignored).
All timestamps are normalized to UTC.
"""
import argparse
import os
import fnmatch
from datetime import timedelta, timezone
from git import Repo


def get_args():
    """Simply specify args"""
    p = argparse.ArgumentParser()
    p.add_argument("repo", help="Path to the git repository root")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--start",help="Starting commit hash (inclusive if on mainline; otherwise date-based cutoff)")
    g.add_argument("--days", type=int, help="Number of days before HEAD commit date")

    p.add_argument("--files",nargs="*",default=[],help="Basenames to keep commits for (exact match). Example: --files fish.py test_fish.py")
    p.add_argument("--patterns",nargs="*",default=[],help='Glob patterns (match basename). Example: --patterns "test_*.py" "*_test.py" "*.py"')

    p.add_argument("--verbose", action="store_true", help="Print hash + date + summary for kept commits")
    p.add_argument("--debug", action="store_true", help="Print per-commit diagnostics (changed files, decision)")
    return p.parse_args()


def normalize_dt_utc(dt):
    """Normalize datetime"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_head_info(repo):
    """Simply extract HEAD information"""
    head = repo.head.commit
    dt = normalize_dt_utc(head.committed_datetime)
    return head, head.hexsha, dt


def changed_paths_in_commit(repo: Repo, commit):
    """Files changed by `commit` along the first-parent chain"""
    if not commit.parents:
        return list(commit.stats.files.keys())
    parent = commit.parents[0]
    out = repo.git.diff("--name-only", parent.hexsha, commit.hexsha)
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_interesting_path(path, files_set, patterns):
    """Match by basename: exact in files_set OR any glob in patterns"""
    if not files_set and not patterns:
        return True

    base = os.path.basename(path)

    if files_set and base in files_set:
        return True

    return any(fnmatch.fnmatch(base, pat) for pat in patterns)


def commit_is_interesting(repo, commit, files_set, patterns):
    """Decide whether the commit should be kept or discarded"""
    if not files_set and not patterns:
        return True

    for p in changed_paths_in_commit(repo, commit):
        if is_interesting_path(p, files_set, patterns):
            return True
    return False


def boundary_commit_by_days(repo, head_commit, days: int):
    """Decide where to stop when user provides --days"""
    if days < 0:
        raise SystemExit("ERROR: --days must be >= 0")

    head_dt = normalize_dt_utc(head_commit.committed_datetime)
    cutoff = head_dt - timedelta(days=days)

    last = head_commit
    for c in repo.iter_commits(rev=head_commit.hexsha, first_parent=True):
        last = c
        if normalize_dt_utc(c.committed_datetime) <= cutoff:
            return c, cutoff
    return last, cutoff


def boundary_commit_by_start(repo, head_commit, start_commit):
    """Decide where to stop when user provides --start"""
    start_dt = normalize_dt_utc(start_commit.committed_datetime)
    head_dt = normalize_dt_utc(head_commit.committed_datetime)

    if start_dt > head_dt:
        raise SystemExit(
            f"ERROR: --start commit is newer than HEAD.\n"
            f"HEAD : {head_commit.hexsha} @ {head_dt.isoformat(timespec='seconds')}\n"
            f"START: {start_commit.hexsha} @ {start_dt.isoformat(timespec='seconds')}"
        )

    prev = None
    for c in repo.iter_commits(rev=head_commit.hexsha, first_parent=True):
        c_dt = normalize_dt_utc(c.committed_datetime)

        # Exact match on mainline (inclusive stop)
        if c.hexsha == start_commit.hexsha:
            return c, start_dt, "exact"

        # Once we cross to <= start_dt, choose boundary:
        # - if equal, choose current c (exact timestamp match)
        # - if strictly older, choose prev (closest newer/equal), else c if no prev
        if c_dt <= start_dt:
            if c_dt == start_dt:
                return c, start_dt, "date"
            return (prev or c), start_dt, "date"

        prev = c

    # It is not expected to happen, but fallback to last commit seen
    return prev or head_commit, start_dt, "date"


def walk_and_print(repo,head_commit,*,boundary_commit=None,boundary_label=None,cutoff_dt=None,files_set=None,patterns=None,debug=False,verbose=False):
    files_set = files_set or set()
    patterns = patterns or []

    must_include = {head_commit.hexsha}
    if boundary_commit is not None:
        must_include.add(boundary_commit.hexsha)

    for commit in repo.iter_commits(rev=head_commit.hexsha, first_parent=True):
        dt = normalize_dt_utc(commit.committed_datetime)

        if commit.hexsha in must_include:
            keep = True
            reason = f"forced(head/{boundary_label})" if boundary_label else "forced(head/boundary)"
        else:
            keep = commit_is_interesting(repo, commit, files_set, patterns)
            reason = "interesting" if keep else "discarded"

        if debug:
            paths = changed_paths_in_commit(repo, commit)
            print("=" * 80)
            print("commit :", commit.hexsha)
            print("date   :", dt.isoformat(timespec="seconds"))
            print("keep?  :", "YES" if keep else "no", f"({reason})")
            if files_set:
                print("files   :", sorted(files_set))
            if patterns:
                print("patterns:", patterns)
            if cutoff_dt:
                print("cutoff (UTC):", cutoff_dt.isoformat(timespec="seconds"))
            if boundary_commit:
                print(f"boundary ({boundary_label}):", boundary_commit.hexsha)
            print("changed files:")
            for p in paths:
                mark = " <==" if is_interesting_path(p, files_set, patterns) else ""
                print("  ", p, mark)

        if keep:
            if verbose:
                print(f"{commit.hexsha} {dt.isoformat(timespec='seconds')} {commit.summary}")
            else:
                print(commit.hexsha)

        # Termination: stop after printing boundary commit (inclusive)
        if boundary_commit is not None and commit.hexsha == boundary_commit.hexsha:
            break
        if cutoff_dt is not None and dt <= cutoff_dt:
            break


def main():
    args = get_args()
    repo = Repo(args.repo)

    head_commit, head_hash, head_dt = get_head_info(repo)

    boundary_commit = None
    boundary_label = None
    cutoff_dt = None

    if args.days is not None:
        boundary_commit, cutoff_dt = boundary_commit_by_days(repo, head_commit, args.days)
        boundary_label = "days"
    else:
        start_commit = repo.commit(args.start)
        boundary_commit, _start_dt, mode = boundary_commit_by_start(repo, head_commit, start_commit)
        boundary_label = "start-exact" if mode == "exact" else "start-date"

    if args.verbose or args.debug:
        print("HEAD:", head_hash)
        print("HEAD date (UTC):", head_dt.isoformat(timespec="seconds"))

    walk_and_print(repo,head_commit,boundary_commit=boundary_commit,boundary_label=boundary_label,cutoff_dt=cutoff_dt,files_set=set(args.files),patterns=args.patterns,debug=args.debug,verbose=args.verbose)


if __name__ == "__main__":
    main()
