#!/usr/bin/env python3
"""
Prepare two checkouts (fixed + buggy) for a repo given a fix commit, and optionally
copy regression tests (or any files) from the fix commit into the buggy checkout.

Core idea:
- fixed checkout: fix commit
- buggy checkout: parent of fix commit (fix^)
- buggy checkout then "imports" chosen files from fix commit:
    git checkout <fix_commit> -- <paths...>

This yields: buggy code + fixed tests.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run(cmd: List[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    # Print commands for transparency/debugging
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=False)


def run_capture(cmd: List[str], cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
    return cp.stdout.strip()


def ensure_git() -> None:
    try:
        run_capture(["git", "--version"])
    except Exception as e:
        raise SystemExit(f"ERROR: git not available: {e}")


def clone_if_needed(repo_url: str, dest: Path) -> None:
    if dest.exists():
        # assume already cloned
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(dest)])


def resolve_parent_commit(repo_dir: Path, fix_commit: str) -> str:
    # parent = fix^
    return run_capture(["git", "rev-parse", f"{fix_commit}^"], cwd=repo_dir)


def ensure_commit_exists(repo_dir: Path, commit: str) -> None:
    run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_dir)


def list_changed_files(repo_dir: Path, commit: str) -> List[str]:
    # files changed in the fix commit
    out = run_capture(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit], cwd=repo_dir)
    return [line.strip() for line in out.splitlines() if line.strip()]


def guess_test_files(changed_files: List[str]) -> List[str]:
    """
    Heuristic: keep paths that look like tests.
    Adjust if your project uses different conventions.
    """
    testy = []
    for p in changed_files:
        norm = p.replace("\\", "/")
        base = os.path.basename(norm)
        if (
            norm.startswith("tests/")
            or "/tests/" in norm
            or base.startswith("test_")
            or base.endswith("_test.py")
            or base.endswith("test.py")
        ):
            testy.append(p)
    return testy


def remove_worktree(path: Path) -> None:
    if not path.exists():
        return
    # Best effort cleanup:
    # - git worktree remove if possible
    # - else rm -rf
    try:
        run(["git", "worktree", "remove", "--force", str(path)])
    except Exception:
        shutil.rmtree(path, ignore_errors=True)


def add_worktree(repo_dir: Path, path: Path, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # detach at commit to avoid branch concerns
    run(["git", "worktree", "add", "--detach", str(path), commit], cwd=repo_dir)


def import_files_from_commit(worktree_dir: Path, source_commit: str, paths: List[str]) -> None:
    if not paths:
        return
    # In the buggy worktree, checkout specific files from the fix commit
    run(["git", "checkout", source_commit, "--", *paths], cwd=worktree_dir)


def main() -> None:
    ensure_git()

    ap = argparse.ArgumentParser(
        description="Create fixed + buggy checkouts from a repo and a fix commit, with buggy using tests from fixed."
    )
    ap.add_argument("--repo", required=True, help="Git repo URL (or local path). Example: https://github.com/...git")
    ap.add_argument("--fix", required=True, help="Fix commit SHA (the commit that contains the fix + regression tests)")
    ap.add_argument(
        "--out",
        default=".",
        help="Output directory where worktrees will be created (default: current dir)",
    )
    ap.add_argument(
        "--name",
        default="case",
        help="Prefix name for created dirs: <name>-repo, <name>-fixed, <name>-buggy (default: case)",
    )
    ap.add_argument(
        "--paths-from-fix",
        default="",
        help=(
            "Comma-separated list of file paths to import from fix commit into buggy checkout. "
            "Example: tests/test_prompt.py,cookiecutter/prompt.py "
            "(If omitted, tries to auto-detect test files changed in the fix commit.)"
        ),
    )
    ap.add_argument(
        "--import-mode",
        choices=["tests-only", "all-changed", "explicit"],
        default="tests-only",
        help=(
            "What to import from the fix commit into buggy worktree:\n"
            "  tests-only  -> import only changed files that look like tests (default)\n"
            "  all-changed -> import all files changed in the fix commit\n"
            "  explicit    -> import exactly --paths-from-fix\n"
        ),
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="If set, remove any existing worktrees/output dirs for this --name before creating new ones.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    base_repo_dir = out_dir / f"{args.name}-repo"
    fixed_dir = out_dir / f"{args.name}-fixed"
    buggy_dir = out_dir / f"{args.name}-buggy"

    if args.clean:
        # Remove worktrees first (git-managed), then repo dir optionally
        # Note: repo dir might be shared; we only remove worktrees here.
        remove_worktree(fixed_dir)
        remove_worktree(buggy_dir)
        # Do not auto-delete base_repo_dir unless you want it; keep clone cache.

    # Clone if needed (or treat args.repo as local repo path)
    if Path(args.repo).exists():
        # local path
        base_repo_dir = Path(args.repo).resolve()
        print(f"Using existing local repo: {base_repo_dir}")
    else:
        clone_if_needed(args.repo, base_repo_dir)

    ensure_commit_exists(base_repo_dir, args.fix)
    parent = resolve_parent_commit(base_repo_dir, args.fix)
    ensure_commit_exists(base_repo_dir, parent)

    # Compute import paths
    changed = list_changed_files(base_repo_dir, args.fix)
    explicit_paths = [p.strip() for p in args.paths_from_fix.split(",") if p.strip()]

    if args.import_mode == "explicit":
        import_paths = explicit_paths
        if not import_paths:
            raise SystemExit("ERROR: --import-mode explicit requires --paths-from-fix to be non-empty")
    elif args.import_mode == "all-changed":
        import_paths = changed
    else:
        # tests-only
        import_paths = guess_test_files(changed)

    # If worktrees already exist, remove them (otherwise git worktree add fails)
    remove_worktree(fixed_dir)
    remove_worktree(buggy_dir)

    # Create fixed and buggy worktrees
    add_worktree(base_repo_dir, fixed_dir, args.fix)
    add_worktree(base_repo_dir, buggy_dir, parent)

    # Import desired files from fix commit into buggy worktree
    if import_paths:
        print("\nImporting these paths from FIX into BUGGY checkout:")
        for p in import_paths:
            print(" -", p)
        import_files_from_commit(buggy_dir, args.fix, import_paths)
    else:
        print("\nNo files selected for import from fix commit into buggy checkout.")

    print("\nDone.")
    print(f"Fixed checkout: {fixed_dir}")
    print(f"Buggy checkout (with selected files from fix): {buggy_dir}")
    print(f"Buggy base commit: {parent}")
    print("\nTip: run tests inside either directory, e.g.:")
    print(f"  cd {buggy_dir} && pytest -q")
    print(f"  cd {fixed_dir} && pytest -q")


if __name__ == "__main__":
    main()
