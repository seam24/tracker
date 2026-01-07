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
    """
    Split [0, total) into njobs contiguous partitions, return (start, end)
    for partition jid.

    Example:
      total=10, njobs=3 partitions sizes: 4,3,3
      jid=0 => [0,4)
      jid=1 => [4,7)
      jid=2 => [7,10)
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


def run_allow_fail(cmd, cwd=None, log_file: Path | None = None, env=None) -> int:
    """
    Run a command and return its exit code, without raising.

    If log_file is provided, redirect stdout+stderr to that file.
    """
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


def run_or_die(cmd, cwd=None, log_file: Path | None = None, env=None, what: str = "command"):
    """
    Run a command and raise RuntimeError if it fails.
    """
    rc = run_allow_fail(cmd, cwd=cwd, log_file=log_file, env=env)
    if rc != 0:
        raise RuntimeError(f"{what} failed with exit code {rc}: {' '.join(cmd)}")
    return rc


def checkout_commit_with_archive(repo_root: Path, commit_hex: str, dest: Path):
    """
    Fast, clean checkout: uses `git archive <commit> | tar -xC <dest>`
    so we avoid mutating repo state, and avoid .git directory in the sandbox.
    """
    dest.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", commit_hex],
        stdout=subprocess.PIPE,
    )
    try:
        p2 = subprocess.Popen(["tar", "-xC", str(dest)], stdin=p1.stdout)
        assert p1.stdout is not None
        p1.stdout.close()
        rc2 = p2.wait()
        rc1 = p1.wait()
    finally:
        if p1.poll() is None:
            p1.wait()
    if rc1 != 0 or rc2 != 0:
        raise RuntimeError(f"git archive failed for commit {commit_hex}")


def reconstruct_mutant_code(original: str, mu: dict) -> str:
    """
    Apply a mutant patch using byte positions stored in mutant_history.json.
    """
    return original[: mu["start_pos"]] + mu["mutant_value"] + original[mu["end_pos"] :]


def sanitize_mutant_key(k: str) -> str:
    """
    Turn a mutant key into a filename-safe token.
    """
    return k.replace("/", "_").replace("#", "_").replace(".", "_")


def get_mutants(commit: str, mutant_history_path: Path):
    """
    Extract the set of mutants that exist at a given commit, with their
    start/end positions updated for that commit.
    """
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
    """
    Base class for both program-mode and mutant-mode runs.

    Key properties:
      - Creates a private sandbox (TMPDIR-aware)
      - Checks out the commit into sandbox/work
      - Forces caches into sandbox paths (HOME, XDG_CACHE_HOME, pycache, pytest basetemp)
      - Optionally installs the package editable via pip
    """
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

        # Environment for subprocess calls
        self.env = os.environ.copy()

        # Choose a stable temp base (prefer TMPDIR/SLURM_TMPDIR if set)
        base_tmp = (
            self.env.get("TMPDIR")
            or self.env.get("SLURM_TMPDIR")
            or self.env.get("TEMP")
            or self.env.get("TMP")
            or "/tmp"
        )
        base_tmp_path = Path(base_tmp)
        base_tmp_path.mkdir(parents=True, exist_ok=True)

        # Sandbox roots (make sure they live under stable base_tmp)
        self.tmp_root = Path(tempfile.mkdtemp(prefix="sandbox-", dir=str(base_tmp_path)))
        self.work_dir = self.tmp_root / "work"
        self.home_dir = self.tmp_root / "home"

        # Extra cache dirs inside sandbox
        self.cache_dir = self.tmp_root / "cache"
        self.pycache_dir = self.tmp_root / "pycache"
        self.pytest_basetemp_root = self.tmp_root / "pytest_basetemp"

        # Where we store install logs for debugging
        self.install_log = self.results_root / f"log_install_{self.commit}.txt"
        self.uninstall_log = self.results_root / f"log_uninstall_{self.commit}.txt"
        self.pip_debug_log = self.results_root / f"log_pip_debug_{self.commit}.txt"

    def prepare_environment(self):
        """
        Make sandbox directories and force cache paths to the sandbox.
        Also push the work tree onto PYTHONPATH so `import sanic` can work
        from source even if pip install fails.
        """
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pycache_dir.mkdir(parents=True, exist_ok=True)
        self.pytest_basetemp_root.mkdir(parents=True, exist_ok=True)

        # Isolate HOME
        self.env["HOME"] = str(self.home_dir)
        self.env["XDG_CACHE_HOME"] = str(self.cache_dir)

        # Make python bytecode deterministic / isolated
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPYCACHEPREFIX"] = str(self.pycache_dir)

        # Ensure subprocesses also use this sandbox as temp (important!)
        self.env["TMPDIR"] = str(self.tmp_root)
        self.env["TEMP"] = str(self.tmp_root)
        self.env["TMP"] = str(self.tmp_root)

        print(f"+ Sandbox root: {self.tmp_root}")
        print(f"+ Work dir: {self.work_dir}")
        print(f"+ Isolated HOME: {self.home_dir}")
        print(f"+ Isolated XDG_CACHE_HOME: {self.cache_dir}")
        print(f"+ Isolated PYTHONPYCACHEPREFIX: {self.pycache_dir}")

    def _conda_python(self):
        return ["conda", "run", "-n", self.conda_env, "python"]

    def _pip(self):
        return ["conda", "run", "-n", self.conda_env, "python", "-m", "pip"]

    def pip_debug_snapshot(self):
        """
        Dump basic pip state into a debug log, to help diagnose broken envs.
        """
        with self.pip_debug_log.open("w", encoding="utf-8") as f:
            for cmd in (
                self._conda_python() + ["-V"],
                self._pip() + ["--version"],
                self._pip() + ["list"],
                self._pip() + ["check"],
            ):
                f.write(f"\n\n$ {' '.join(cmd)}\n")
                try:
                    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=self.env)
                    f.write(f"[exit={proc.returncode}]\n")
                except Exception as e:
                    f.write(f"[exception] {e}\n")

    def nuke_editable_artifacts(self):
        """
        IMPORTANT FIX for your error:
        pip editable installs drop *.egg-link and often a .pth entry.
        If a previous editable install pointed at /tmp/sandbox-XXXX/work and that sandbox
        is gone, pip can error with:
          OSError: [Errno 2] No such file or directory: '/tmp/sandbox-.../work'

        This removes:
          - <pkg>.egg-link
          - easy-install.pth lines mentioning the package
          - any stray *egg-info for the package name (best-effort)
        """
        if not self.pkg_name:
            return

        code = r"""
import os, site, glob, pathlib, re

pkg = os.environ.get("PKG_NAME", "").strip()
if not pkg:
    raise SystemExit("PKG_NAME not set")

paths = []
for p in (site.getsitepackages() + [site.getusersitepackages()]):
    if p and os.path.isdir(p):
        paths.append(p)

print("site-packages paths:")
for p in paths:
    print(" -", p)

def rm(path):
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print("WARN rm failed:", path, e)

# Remove egg-link files
for sp in paths:
    for cand in glob.glob(os.path.join(sp, f"{pkg}.egg-link")):
        print("DELETE egg-link:", cand)
        rm(cand)

# Clean easy-install.pth if present
for sp in paths:
    pth = os.path.join(sp, "easy-install.pth")
    if os.path.isfile(pth):
        txt = pathlib.Path(pth).read_text(errors="ignore").splitlines()
        kept = []
        removed = 0
        for line in txt:
            if re.search(rf"\\b{re.escape(pkg)}\\b", line):
                removed += 1
                print("REMOVE from easy-install.pth:", line)
            else:
                kept.append(line)
        if removed:
            pathlib.Path(pth).write_text("\n".join(kept) + ("\n" if kept else ""))
            print("UPDATED easy-install.pth:", pth)

# Remove obvious egg-info directories/files
for sp in paths:
    for pat in (f"{pkg}*.egg-info", f"{pkg.replace('-','_')}*.egg-info"):
        for cand in glob.glob(os.path.join(sp, pat)):
            print("DELETE egg-info:", cand)
            rm(cand)
"""
        self.install_log.parent.mkdir(parents=True, exist_ok=True)
        with self.install_log.open("a", encoding="utf-8") as f:
            f.write("\n\n=== NUKE editable artifacts ===\n")

        env2 = dict(self.env)
        env2["PKG_NAME"] = self.pkg_name
        run_allow_fail(self._conda_python() + ["-c", code], env=env2, log_file=self.install_log)

    def uninstall_package(self):
        """
        Best-effort uninstall of pkg_name from the conda env.
        Not fatal if it fails.
        """
        if not self.pkg_name:
            return
        run_allow_fail(
            self._pip() + ["uninstall", "-y", self.pkg_name],
            env=self.env,
            log_file=self.uninstall_log,
        )
        # Extra: remove stale editable artifacts even if uninstall failed
        self.nuke_editable_artifacts()

    def checkout(self):
        """
        Place the commit content into sandbox/work.
        """
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

    # ----------------------------
    # httpx repair helpers
    # ----------------------------

    def repair_broken_httpx(self):
        """
        Best-effort repair for corrupted httpx installs.
        """
        code = r"""
import site, sys, shutil, glob, os
paths = site.getsitepackages() + [site.getusersitepackages()]
seen = set()
targets = []
for sp in paths:
    if not sp or not os.path.isdir(sp):
        continue
    for pat in ("httpx", "httpx-*", "*ttpx*", "httpx*.dist-info", "httpx*.egg-info"):
        for p in glob.glob(os.path.join(sp, pat)):
            if p not in seen:
                seen.add(p)
                targets.append(p)

print("site-packages candidates:")
for sp in paths:
    print(" -", sp)

print("delete targets:")
for t in targets:
    print(" -", t)

for t in targets:
    try:
        if os.path.isdir(t) and not os.path.islink(t):
            shutil.rmtree(t, ignore_errors=True)
        else:
            try:
                os.unlink(t)
            except FileNotFoundError:
                pass
    except Exception as e:
        print("WARN could not delete", t, ":", e)
"""
        self.install_log.parent.mkdir(parents=True, exist_ok=True)
        with self.install_log.open("a", encoding="utf-8") as f:
            f.write("\n\n=== REPAIR httpx: delete broken files ===\n")
        run_allow_fail(self._conda_python() + ["-c", code], env=self.env, log_file=self.install_log)

        with self.install_log.open("a", encoding="utf-8") as f:
            f.write("\n\n=== REPAIR httpx: reinstall httpx==0.11.1 ===\n")
        run_allow_fail(
            self._pip() + ["install", "--no-cache-dir", "--force-reinstall", "httpx==0.11.1"],
            env=self.env,
            log_file=self.install_log,
        )

    def install_package(self):
        """
        Install the repository in editable mode (fatal if it fails).
        """
        if not self.pkg_name:
            return

        self.pip_debug_snapshot()

        # remove any stale egg-link / pth before install (critical)
        self.nuke_editable_artifacts()

        # best-effort httpx repair
        self.repair_broken_httpx()

        # Ensure PYTHONPATH includes work_dir so importing from source is possible.
        prev_pp = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = f"{self.work_dir}{os.pathsep}{prev_pp}" if prev_pp else str(self.work_dir)

        # Sanity check: work_dir must exist before pip touches it
        if not self.work_dir.exists():
            raise RuntimeError(f"work_dir does not exist before install: {self.work_dir}")

        # Install editable, avoid build isolation to reduce temp path weirdness
        run_or_die(
            self._pip() + ["install", "-e", ".", "--no-build-isolation"],
            cwd=self.work_dir,
            env=self.env,
            log_file=self.install_log,
            what="pip install -e .",
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
        if not self.work_dir.exists():
            return
        self._rm_tree(self.work_dir / ".pytest_cache")
        for d in self.work_dir.rglob("__pycache__"):
            self._rm_tree(d)
        for f in self.work_dir.rglob("*.pyc"):
            try:
                f.unlink()
            except Exception:
                pass

    def clear_global_sandbox_caches(self):
        self._rm_tree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._rm_tree(self.pycache_dir)
        self.pycache_dir.mkdir(parents=True, exist_ok=True)

        self._rm_tree(self.pytest_basetemp_root)
        self.pytest_basetemp_root.mkdir(parents=True, exist_ok=True)

    def conda_pytest(self, args, cwd=None, log_file=None, run_tag: str = "run"):
        self.clear_workdir_caches()

        basetemp = self.pytest_basetemp_root / run_tag
        if basetemp.exists():
            shutil.rmtree(basetemp, ignore_errors=True)
        basetemp.mkdir(parents=True, exist_ok=True)

        ignore_args = []
        for p in self.ignore_paths:
            ignore_args += ["--ignore", p]

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

        run_tag = f"mut-{safe}"
        rc = self.conda_pytest(args, cwd=self.work_dir, log_file=log, run_tag=run_tag)

        wrapped = dict(mu)
        wrapped["pytest_exit_code"] = rc
        wrapped["test_outcomes"] = None
        if inner.exists():
            wrapped["test_outcomes"] = json.loads(inner.read_text())

        out.write_text(json.dumps(wrapped, indent=2))

        self.restore(mu)
        self.clear_workdir_caches()

    def run(self):
        try:
            if not hasattr(self, "program_results_path"):
                raise RuntimeError("Mutant runner requires program_results_path")

            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()
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

    # PROGRAM
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

    # MUTANT
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

        if args.run == "both":
            mut.program_results_path = prog.program_results_path
        else:
            mut.program_results_path = results_dir / f"pres_{commit}.json"

        mut.stop_on_kill = args.stop_on_kill
        mut.timeout_min = args.timeout_min
        mut.timeout_multiplier = args.timeout_multiplier
        mut.disable_incompetent = args.disable_incompetent

        mut.run()
