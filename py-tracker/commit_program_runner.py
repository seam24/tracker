#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from git import Repo, Commit


def run_allow_fail(
    cmd: list[str],
    cwd: str | Path | None = None,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """
    Run a command, but do NOT raise if it fails.
    If log_file is given, capture stdout+stderr into that file.
    If env is given, use that environment (e.g., to override HOME).
    Return the exit code.
    """
    print(f"+ (cwd={cwd}) {' '.join(cmd)}")

    if cwd is not None:
        cwd = str(cwd)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=env)

    if proc.returncode != 0:
        print(f"!! Command exited with code {proc.returncode}, continuing.")
    return proc.returncode


def get_commits_since(repo: Repo, days: int) -> list[Commit]:
    """
    Return list of Commit objects from HEAD in the last `days` days,
    relative to HEAD's commit date (newest first).
    """
    head_commit: Commit = repo.head.commit
    head_time = head_commit.committed_datetime

    cutoff = head_time - dt.timedelta(days=days)
    print("HEAD:", head_commit.hexsha, head_time)
    print("Cutoff:", cutoff)
    print("-" * 80)

    commits: list[Commit] = []
    for commit in repo.iter_commits(rev=head_commit.hexsha):  # newest → oldest
        if commit.committed_datetime < cutoff:
            break
        commits.append(commit)

    return commits


def compute_partition_indices(total: int, njobs: int, jid: int) -> tuple[int, int]:
    """
    Compute [start, end) indices for job `jid` out of `njobs` total, partitioning
    `total` elements contiguously and as evenly as possible.
    """
    if njobs <= 0:
        raise SystemExit("njobs must be >= 1")
    if jid < 0 or jid >= njobs:
        raise SystemExit(f"jid must be in [0,{njobs-1}]")

    base = total // njobs
    rem = total % njobs

    if jid < rem:
        start = jid * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (jid - rem) * base
        end = start + base

    return start, end


def checkout_commit_with_archive(repo_root: Path, commit_hex: str, work_dir: Path) -> None:
    """
    Populate work_dir with the tree of commit_hex using `git archive`.

    IMPORTANT:
    ----------
    This method is chosen **because it is 100% safe for parallel use**.

    - `git worktree` modifies `.git/worktrees/*` and will race/corrupt
      if multiple processes run `worktree add/remove` at the same time.

    - `shutil.copytree(repo_root, ...)` is also unsafe in parallel because:
        * existing .git/worktrees/* may disappear during remove operations
        * copying a repo while another job removes a worktree can
          produce errors like “missing but already registered worktree”.

    `git archive` avoids ALL `.git` structure and writes only file snapshots.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # git -C repo_root archive <commit> | tar -xC work_dir
    p1 = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", commit_hex],
        stdout=subprocess.PIPE,
    )
    try:
        p2 = subprocess.Popen(
            ["tar", "-xC", str(work_dir)],
            stdin=p1.stdout,
        )
        p1.stdout.close()
        rc2 = p2.wait()
        rc1 = p1.wait()
    finally:
        if p1.poll() is None:
            p1.wait()

    if rc1 != 0 or rc2 != 0:
        raise RuntimeError(
            f"checkout_commit_with_archive failed for {commit_hex}: "
            f"git rc={rc1}, tar rc={rc2}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pytest for each commit in the last N days (relative to HEAD)."
    )
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Number of days back from HEAD's commit date to include commits.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="Path to the root of the git repository.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help=(
            "Optional name of the conda environment to use. "
            "If provided, commands are run as "
            "'conda run -n <env> ...'. If omitted, the script "
            "invokes 'pytest'/'pip' directly in the current environment."
        ),
    )

    # Running pip install/uninstall concurrently from multiple processes
    # can corrupt the environment (conda or otherwise).
    # Therefore: if --pkg-name is provided, only sequential mode is allowed.
    parser.add_argument(
        "--pkg-name",
        type=str,
        default=None,
        help=(
            "Python package name to install -e ./uninstall with pip for each commit "
            "(e.g. 'PySnooper'). If provided, ONLY sequential mode is allowed. "
            "If omitted, no pip operations are performed and parallel mode via "
            "--cores/--jid is allowed."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory where pytest results will be stored as pres_<commit>.json",
    )
    # Parallelism controls
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Total number of parallel jobs/cores. Default: 1.",
    )
    parser.add_argument(
        "--jid",
        type=int,
        default=0,
        help="Job id for this process: 0 <= jid < cores. Default: 0.",
    )

    args = parser.parse_args()

    if args.cores <= 0:
        raise SystemExit("--cores must be >= 1")
    if not (0 <= args.jid < args.cores):
        raise SystemExit("--jid must satisfy 0 <= jid < cores")

    # If pkg_name is provided, force sequential mode (no parallel sharding)
    if args.pkg_name is not None and (args.cores != 1 or args.jid != 0):
        raise SystemExit(
            "Parallel mode (cores/jid) is NOT supported when --pkg-name is provided.\n"
            "Remove --pkg-name for parallel runs, or use --cores 1 --jid 0 for "
            "sequential mode with pip."
        )

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / ".git").exists():
        raise SystemExit(f"{repo_root} does not look like a git repo (no .git found).")

    repo = Repo(repo_root)

    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"+ Results will be saved under: {results_root}")
    print(
        f"+ cores={args.cores}, jid={args.jid}, "
        f"pkg_name={args.pkg_name}, conda_env={args.conda_env}"
    )

    commits = get_commits_since(repo, args.days)
    if not commits:
        print(f"No commits in the last {args.days} days (relative to HEAD). Exiting.")
        return

    total = len(commits)
    print(f"+ Found {total} commits in window.")
    print(f"+ First commit (should be HEAD): {commits[0].hexsha}")

    # Decide which commits this process will handle
    if args.pkg_name is None and args.cores > 1:
        # Parallel mode: use contiguous partition [start,end)
        start, end = compute_partition_indices(total, args.cores, args.jid)
        selected_indices = list(range(start, end))
        print(f"+ Partition for jid={args.jid}: start={start}, end={end} (total={total})")
    else:
        # Sequential mode (or cores=1 without pkg): process all commits
        selected_indices = list(range(total))

    if not selected_indices:
        print("+ Nothing to do for this job. Exiting.")
        return

    shard_commits = [commits[i] for i in selected_indices]
    print(f"+ This job will handle {len(shard_commits)} commits:")
    print(f"  first={shard_commits[0].hexsha}, last={shard_commits[-1].hexsha}")

    tmp_root = Path(tempfile.mkdtemp(prefix=f"git-pytest-commits-j{args.jid}-"))
    work_dir = tmp_root / "work"
    print(f"+ Using temp root directory: {tmp_root}")

    # Per-job HOME to isolate cookiecutter replay dir and similar state
    home_dir = tmp_root / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["HOME"] = str(home_dir)
    print(f"+ Using isolated HOME for this job: {home_dir}")

    parallel_mode = (args.pkg_name is None and args.cores > 1)

    # Helper to optionally prepend "conda run -n <env>".
    def wrap_cmd(inner: list[str]) -> list[str]:
        if args.conda_env:
            return ["conda", "run", "-n", args.conda_env] + inner
        return inner

    try:
        # Safety uninstall before loop if pkg_name is provided (sequential pip mode)
        if args.pkg_name:
            print(f"+ Safety: uninstalling {args.pkg_name} BEFORE loop")
            run_allow_fail(
                wrap_cmd(["pip", "uninstall", "-y", args.pkg_name]),
                env=env_base,
            )

        for global_idx in selected_indices:
            commit = commits[global_idx]
            print("\n" + "=" * 80)
            print(f"[global_idx={global_idx}] Processing commit: {commit.hexsha}")
            print("=" * 80)

            is_head_global = (global_idx == 0)

            # Ensure work_dir is empty before each iteration
            if work_dir.exists():
                print(f"+ Removing previous work dir {work_dir}")
                shutil.rmtree(work_dir)

            if parallel_mode:
                # PARALLEL MODE (no git worktree)
                if is_head_global:
                    # Only one job will see global_idx == 0 → safe
                    print(
                        "+ Parallel HEAD iteration → copying current repo_root "
                        "(incl. uncommitted) into work_dir."
                    )
                    shutil.copytree(repo_root, work_dir)
                else:
                    print("+ Parallel mode → using git archive to populate work_dir.")
                    checkout_commit_with_archive(repo_root, commit.hexsha, work_dir)
            else:
                # SEQUENTIAL MODE (old semantics with worktree)
                if is_head_global:
                    print(
                        "+ HEAD iteration → copying current repo_root "
                        "(incl. uncommitted) into work_dir."
                    )
                    shutil.copytree(repo_root, work_dir)
                else:
                    print("+ Using git worktree add for this commit (via GitPython).")
                    repo.git.worktree("add", str(work_dir), commit.hexsha)

            # Files for this commit
            result_file = results_root / f"pres_{commit.hexsha}.json"
            log_file = results_root / f"log_{commit.hexsha}.txt"

            # If pkg_name is provided → do per-commit pip install/uninstall (sequential only)
            if args.pkg_name is not None:
                pip_install_cmd = wrap_cmd(["pip", "install", "-e", "."])
                print("+ Running pip install -e .")
                run_allow_fail(pip_install_cmd, cwd=work_dir, env=env_base)

            # Run pytest, logging output (always)
            pytest_cmd = wrap_cmd(
                [
                    "pytest",
                    "--mode",
                    "program",
                    "--program-results",
                    str(result_file),
                ]
            )

            exit_code = run_allow_fail(
                pytest_cmd,
                cwd=work_dir,
                log_file=log_file,
                env=env_base,
            )
            print(f"+ pytest exit code for {commit.hexsha}: {exit_code}")
            print(f"+ pytest log for {commit.hexsha} saved to: {log_file}")

            if args.pkg_name is not None:
                pip_uninstall_cmd = wrap_cmd(
                    ["pip", "uninstall", "-y", args.pkg_name]
                )
                print(f"+ Uninstalling package {args.pkg_name} from environment")
                run_allow_fail(pip_uninstall_cmd, cwd=work_dir, env=env_base)

            if not parallel_mode and not is_head_global:
                # Only sequential mode uses worktrees
                print("+ Cleaning up worktree for this commit.")
                try:
                    repo.git.worktree("remove", str(work_dir), "--force")
                except Exception as e:
                    print(f"!! Failed to remove worktree {work_dir}: {e}")

    finally:
        # Safety uninstall after loop if pkg_name is provided
        if args.pkg_name:
            print(f"+ Safety: uninstalling {args.pkg_name} AFTER loop")
            run_allow_fail(
                wrap_cmd(["pip", "uninstall", "-y", args.pkg_name]),
                env=env_base,
            )

        print(f"\n+ Cleaning up temp root directory: {tmp_root}")
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nAll done for jid={args.jid}. Results stored in: {results_root}")


if __name__ == "__main__":
    main()
