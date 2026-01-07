#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# ============================================================
# Utility functions
# ============================================================

def compute_partition_indices(total, njobs, jid):
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


def run_allow_fail(cmd, cwd=None, log_file=None, env=None) -> int:
    print(f"+ (cwd={cwd}) {' '.join(cmd)}")
    if cwd is not None:
        cwd = str(cwd)

    if log_file:
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
        print(f"!! Command exited with {proc.returncode}")
    return proc.returncode


def checkout_commit_with_archive(repo_root: Path, commit_hex: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", commit_hex],
        stdout=subprocess.PIPE,
    )
    try:
        p2 = subprocess.Popen(["tar", "-xC", str(dest)], stdin=p1.stdout)
        p1.stdout.close()
        rc2 = p2.wait()
        rc1 = p1.wait()
    finally:
        if p1.poll() is None:
            p1.wait()
    if rc1 != 0 or rc2 != 0:
        raise RuntimeError(f"git archive failed for commit {commit_hex}")


def reconstruct_mutant_code(original: str, mu: dict) -> str:
    return original[: mu["start_pos"]] + mu["mutant_value"] + original[mu["end_pos"] :]


def sanitize_mutant_key(k: str) -> str:
    return k.replace("/", "_").replace("#", "_").replace(".", "_")


def get_mutants(commit: str, mutant_history_path: Path):
    with mutant_history_path.open("r") as f:
        mhist = json.load(f)

    chosen = []
    for key, entry in mhist.items():
        info = entry["info"]
        history = entry["history"]

        start = end = None
        for h in history:
            if h["commit"] == commit:
                start = h["start_pos"]
                end = h["end_pos"]
                break

        if start is None:
            continue

        chosen.append({
            "mutant_key": key,
            "mutant_id": info["mutant_id"],
            "file": info["file"],
            "module": info["module"],
            "op_type": info["op_type"],
            "scope": info["scope"],
            "start_pos": start,
            "end_pos": end,
            "mutant_value": info["mutant_value"],
            "commit": commit,
        })
    return chosen


# ============================================================
# Base runner
# ============================================================

class BaseCommitRunner:
    def __init__(
        self,
        repo_root: Path,
        commit: str,
        conda_env: str,
        pkg_name: str | None,
        results_root: Path,
        use_head: bool,
        sandbox_mode: str,
        test_paths: list[str] | None,
        ignore_paths: list[str] | None,
    ):
        self.repo_root = repo_root
        self.commit = commit
        self.conda_env = conda_env
        self.pkg_name = pkg_name
        self.results_root = results_root
        self.use_head = use_head
        self.sandbox_mode = sandbox_mode
        self.test_paths = list(test_paths) if test_paths else []
        self.ignore_paths = list(ignore_paths) if ignore_paths else []

        # Sandbox
        self.tmp_root = Path(tempfile.mkdtemp(prefix="sandbox-"))
        self.work_dir = self.tmp_root / "work"
        self.home_dir = self.tmp_root / "home"

        # Extra cache dirs inside sandbox
        self.cache_dir = self.tmp_root / "cache"
        self.pycache_dir = self.tmp_root / "pycache"
        self.pytest_basetemp_root = self.tmp_root / "pytest_basetemp"

        self.env = os.environ.copy()

    def prepare_environment(self):
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pycache_dir.mkdir(parents=True, exist_ok=True)
        self.pytest_basetemp_root.mkdir(parents=True, exist_ok=True)

        # Isolate HOME (already good)
        self.env["HOME"] = str(self.home_dir)

        # Keep common caches inside sandbox
        self.env["XDG_CACHE_HOME"] = str(self.cache_dir)

        # Make python bytecode deterministic / isolated
        # If you prefer allowing pyc, remove PYTHONDONTWRITEBYTECODE
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPYCACHEPREFIX"] = str(self.pycache_dir)

        print(f"+ Isolated HOME: {self.home_dir}")
        print(f"+ Isolated XDG_CACHE_HOME: {self.cache_dir}")
        print(f"+ Isolated PYTHONPYCACHEPREFIX: {self.pycache_dir}")

    def uninstall_package(self):
        if not self.pkg_name:
            return
        run_allow_fail(
            ["conda", "run", "-n", self.conda_env, "python", "-m", "pip", "uninstall", "-y", self.pkg_name],
            env=self.env,
        )

    def checkout(self):
        if self.use_head:
            shutil.copytree(self.repo_root, self.work_dir)
            return

        if self.sandbox_mode == "archive":
            checkout_commit_with_archive(self.repo_root, self.commit, self.work_dir)
        elif self.sandbox_mode == "copy":
            shutil.copytree(self.repo_root, self.work_dir)
        elif self.sandbox_mode == "symlink":
            shutil.copytree(self.repo_root, self.work_dir, copy_function=os.symlink, dirs_exist_ok=True)
        else:
            raise ValueError(f"Unknown sandbox_mode: {self.sandbox_mode}")

    def install_package(self):
        if not self.pkg_name:
            return
        run_allow_fail(
            ["conda", "run", "-n", self.conda_env, "python", "-m", "pip", "install", "-e", "."],
            cwd=self.work_dir,
            env=self.env,
        )

    # ----------------------------
    # Cache nuking (best effort)
    # ----------------------------
    def _rm_tree(self, p: Path):
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except Exception:
            pass

    def clear_workdir_caches(self):
        """
        Clear caches inside the checked-out work tree.
        This is important because you run pytest repeatedly in the SAME work_dir
        (especially in mutant mode).
        """
        if not self.work_dir.exists():
            return

        # pytest cache
        self._rm_tree(self.work_dir / ".pytest_cache")

        # python caches inside work tree (if any were created)
        for d in self.work_dir.rglob("__pycache__"):
            self._rm_tree(d)

        # stray .pyc files
        for f in self.work_dir.rglob("*.pyc"):
            try:
                f.unlink()
            except Exception:
                pass

    def clear_global_sandbox_caches(self):
        """
        Clear caches under sandbox dirs (not strictly necessary each time,
        but useful if some tool writes there).
        """
        self._rm_tree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._rm_tree(self.pycache_dir)
        self.pycache_dir.mkdir(parents=True, exist_ok=True)

        # basetemp root; recreate
        self._rm_tree(self.pytest_basetemp_root)
        self.pytest_basetemp_root.mkdir(parents=True, exist_ok=True)

    def conda_pytest(self, args, cwd=None, log_file=None, run_tag: str = "run"):
        """
        Run pytest with aggressive cache-clearing and unique basetemp.
        """
        # Clear caches that might affect outcomes between runs
        self.clear_workdir_caches()

        # Unique basetemp for this specific pytest invocation
        basetemp = self.pytest_basetemp_root / run_tag
        if basetemp.exists():
            shutil.rmtree(basetemp, ignore_errors=True)
        basetemp.mkdir(parents=True, exist_ok=True)

        ignore_args = []
        for p in self.ignore_paths:
            ignore_args += ["--ignore", p]

        # Force pytest to clear its own cache as well
        forced = ["--cache-clear", "--basetemp", str(basetemp)]

        cmd = ["conda", "run", "-n", self.conda_env, "python", "-m", "pytest"] + forced + ignore_args + self.test_paths + args
        return run_allow_fail(cmd, cwd=cwd, env=self.env, log_file=log_file)

    def cleanup(self):
        if self.pkg_name:
            self.uninstall_package()
        shutil.rmtree(self.tmp_root, ignore_errors=True)


# ============================================================
# ProgramCommitRunner
# ============================================================

class ProgramCommitRunner(BaseCommitRunner):
    def run_program_tests(self):
        out_json = self.results_root / f"pres_{self.commit}.json"
        out_log = self.results_root / f"log_program_{self.commit}.txt"
        self.program_results_path = out_json

        args = ["--mode", "program", "--program-results", str(out_json)]
        self.conda_pytest(args, cwd=self.work_dir, log_file=out_log, run_tag=f"program-{self.commit}")

    def run(self):
        try:
            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()

            # optional: also clear sandbox-level caches once before the run
            self.clear_global_sandbox_caches()

            self.run_program_tests()
        finally:
            self.cleanup()


# ============================================================
# MutantCommitRunner
# ============================================================

class MutantCommitRunner(BaseCommitRunner):
    def __init__(self, *args, mutants, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutants = mutants
        self.original_cache = {}

        # to be set externally:
        # self.program_results_path

        # pytest plugin options (from CLI)
        self.stop_on_kill = False
        self.timeout_min = None
        self.timeout_multiplier = None
        self.disable_incompetent = False

    def preload_originals(self):
        for mu in self.mutants:
            rel = Path(mu["file"])
            abs_path = self.work_dir / rel
            if rel.as_posix() not in self.original_cache:
                self.original_cache[rel.as_posix()] = abs_path.read_text()

    def apply_mutant(self, mu):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        original = self.original_cache[rel.as_posix()]
        mutated = reconstruct_mutant_code(original, mu)
        if abs_path.is_symlink():
            abs_path.unlink()
        abs_path.write_text(mutated)

    def restore(self, mu):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        original = self.original_cache[rel.as_posix()]
        if abs_path.is_symlink():
            abs_path.unlink()
        abs_path.write_text(original)

    def run_single_mutant(self, mu):
        self.apply_mutant(mu)

        inner = self.work_dir / ".inner_mutant.json"
        if inner.exists():
            inner.unlink()

        safe = sanitize_mutant_key(mu["mutant_key"])
        log = self.results_root / f"log_mut_{safe}_{self.commit}.txt"
        out = self.results_root / f"mutant_{safe}_{self.commit}.json"

        args = [
            "--mode", "mutant",
            "--program-results", str(self.program_results_path),
            "--mutant-results", str(inner),
        ]
        if self.stop_on_kill:
            args.append("--stop-on-kill")
        if self.timeout_min:
            args += ["--timeout-min", str(self.timeout_min)]
        if self.timeout_multiplier:
            args += ["--timeout-multiplier", str(self.timeout_multiplier)]
        if self.disable_incompetent:
            args.append("--disable-incompetent")

        # Ensure *this* mutant run doesn't reuse previous caches
        run_tag = f"mut-{safe}"

        rc = self.conda_pytest(args, cwd=self.work_dir, log_file=log, run_tag=run_tag)

        wrapped = dict(mu)
        wrapped["pytest_exit_code"] = rc
        wrapped["test_outcomes"] = None

        if inner.exists():
            wrapped["test_outcomes"] = json.loads(inner.read_text())

        out.write_text(json.dumps(wrapped, indent=2))

        self.restore(mu)

        # Clear caches again after restore (best effort)
        self.clear_workdir_caches()

    def run(self):
        try:
            if not hasattr(self, "program_results_path"):
                raise RuntimeError("Mutant runner requires program_results_path")

            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()

            # optional: clear sandbox-level caches once before starting loop
            self.clear_global_sandbox_caches()

            self.preload_originals()

            for mu in self.mutants:
                self.run_single_mutant(mu)

        finally:
            self.cleanup()


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--repo", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--conda-env", required=True)
    p.add_argument("--pkg-name")
    p.add_argument("--results", required=True)

    p.add_argument("--run", choices=["program", "mutant", "both"], default="both")

    p.add_argument("--tests", nargs="*", default=[])

    # Match pytest's flag name; allow repeated use:
    #   --ignore path1 --ignore path2 ...
    p.add_argument("--ignore", action="append", default=[],
                   help="Paths to ignore (passed through to pytest as --ignore)")

    p.add_argument("--sandbox-mode", choices=["archive", "copy", "symlink"], default="archive")
    p.add_argument("--use-head", action="store_true")

    p.add_argument("--mutant-history")
    p.add_argument("--mutant-limit", type=int)

    # pytest-mutation plugin options
    p.add_argument("--stop-on-kill", action="store_true")
    p.add_argument("--timeout-min", type=float)
    p.add_argument("--timeout-multiplier", type=float)
    p.add_argument("--disable-incompetent", action="store_true")

    p.add_argument("--njobs", type=int, default=1, help="Parallel jobs for mutant mode")
    p.add_argument("--jid", type=int, default=0, help="Job index for mutant mode")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    repo_root = Path(args.repo).resolve()
    results_dir = Path(args.results).resolve()
    commit = args.commit

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load mutants if needed
    mutants = []
    if args.run in ("mutant", "both"):

        if args.njobs < 1:
            raise SystemExit("--njobs must be >= 1")

        if args.jid < 0:
            raise SystemExit("--jid must be >= 0")

        if args.jid >= args.njobs:
            raise SystemExit(f"--jid must be < --njobs (got jid={args.jid}, njobs={args.njobs})")

        if not args.mutant_history:
            raise SystemExit("--mutant-history required for mutant mode")

        mutants = get_mutants(commit, Path(args.mutant_history))
        if args.mutant_limit:
            mutants = mutants[: args.mutant_limit]

    # -------------------------------
    # Run PROGRAM mode
    # -------------------------------
    if args.run in ("program", "both"):
        prog = ProgramCommitRunner(
            repo_root,
            commit,
            args.conda_env,
            args.pkg_name,
            results_dir,
            use_head=args.use_head,
            sandbox_mode=args.sandbox_mode,
            test_paths=args.tests,
            ignore_paths=args.ignore,
        )
        prog.run()

    # -------------------------------
    # Run MUTANT mode
    # -------------------------------
    if args.run in ("mutant", "both"):
        njobs = max(1, args.njobs)
        jid = args.jid

        total = len(mutants)
        start, end = compute_partition_indices(total, njobs, jid)
        job_mutants = mutants[start:end]

        print(f"+ Mutant partition: total={total}, njobs={njobs}, jid={jid}, this_job={len(job_mutants)}")

        mut = MutantCommitRunner(
            repo_root,
            commit,
            args.conda_env,
            args.pkg_name,
            results_dir,
            mutants=job_mutants,
            use_head=args.use_head,
            sandbox_mode=args.sandbox_mode,
            test_paths=args.tests,
            ignore_paths=args.ignore,
        )

        # Use existing program result file (from this run or previous)
        if args.run == "both":
            mut.program_results_path = prog.program_results_path
        else:
            mut.program_results_path = results_dir / f"pres_{commit}.json"

        # Pass pytest plugin flags
        mut.stop_on_kill = args.stop_on_kill
        mut.timeout_min = args.timeout_min
        mut.timeout_multiplier = args.timeout_multiplier
        mut.disable_incompetent = args.disable_incompetent

        mut.run()

