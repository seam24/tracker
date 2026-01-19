#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# Utility functions
# ============================================================

def compute_partition_indices(total: int, njobs: int, jid: int) -> tuple[int, int]:
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


def run_allow_fail(cmd, cwd=None, env=None, log_file: Path | None = None) -> int:
    print(f"+ (cwd={cwd}) {' '.join(cmd)}")
    if cwd is not None:
        cwd = str(cwd)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=env)

    if proc.returncode != 0:
        print(f"!! Command exited with {proc.returncode}")
    return proc.returncode


def run_or_die(cmd, cwd=None, env=None, what="command"):
    rc = run_allow_fail(cmd, cwd=cwd, env=env)
    if rc != 0:
        raise RuntimeError(f"{what} failed with exit code {rc}: {' '.join(cmd)}")
    return rc


def checkout_commit_with_archive(repo_root: Path, commit_hex: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen(["git", "-C", str(repo_root), "archive", commit_hex], stdout=subprocess.PIPE)
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
    return original[: mu["start_pos"]] + mu["mutant_value"] + original[mu["end_pos"] :]


def sanitize_mutant_key(k: str) -> str:
    return k.replace("/", "_").replace("#", "_").replace(".", "_")


def get_mutants(commit: str, mutant_history_path: Path) -> list[dict]:
    with mutant_history_path.open("r", encoding="utf-8") as f:
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

        chosen.append(
            {
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
            }
        )
    return chosen


# ============================================================
# JUnit XML metadata helpers (single file)
# ============================================================

def inject_properties(junit_path: Path, props: dict[str, str]) -> None:
    if not junit_path.exists():
        return

    tree = ET.parse(junit_path)
    root = tree.getroot()

    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))
        if not suites:
            suites = list(root.findall(".//testsuite"))

    for suite in suites:
        properties_elem = suite.find("properties")
        if properties_elem is None:
            properties_elem = ET.Element("properties")
            suite.insert(0, properties_elem)

        existing = {}
        for p in properties_elem.findall("property"):
            name = p.get("name")
            if name:
                existing[name] = p

        for k, v in props.items():
            if k in existing:
                existing[k].set("value", v)
            else:
                p = ET.Element("property")
                p.set("name", k)
                p.set("value", v)
                properties_elem.append(p)

    tree.write(junit_path, encoding="utf-8", xml_declaration=True)


def read_property_from_junit(junit_path: Path, prop_name: str) -> str | None:
    if not junit_path.exists():
        return None
    try:
        root = ET.parse(junit_path).getroot()
    except Exception:
        return None

    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))
        if not suites:
            suites = list(root.findall(".//testsuite"))

    for suite in suites:
        props = suite.find("properties")
        if props is None:
            continue
        for p in props.findall("property"):
            if p.get("name") == prop_name:
                return p.get("value")
    return None


# ============================================================
# Base runner
# ============================================================

class BaseCommitRunner:
    """
    Shared sandbox + checkout + optional editable install.

    Kept:
      - isolated pytest basetemp inside sandbox (safe, prevents rare tmp collisions)

    Dropped:
      - httpx repair
      - pip debug snapshots
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

        self.env = os.environ.copy()

        base_tmp = (
            self.env.get("TMPDIR")
            or self.env.get("SLURM_TMPDIR")
            or self.env.get("TEMP")
            or self.env.get("TMP")
            or "/tmp"
        )
        base_tmp_path = Path(base_tmp)
        base_tmp_path.mkdir(parents=True, exist_ok=True)

        self.tmp_root = Path(tempfile.mkdtemp(prefix="sandbox-", dir=str(base_tmp_path)))
        self.work_dir = self.tmp_root / "work"
        self.home_dir = self.tmp_root / "home"
        self.cache_dir = self.tmp_root / "cache"
        self.pycache_dir = self.tmp_root / "pycache"
        self.pytest_basetemp_root = self.tmp_root / "pytest_basetemp"

    def _pip(self):
        return ["conda", "run", "-n", self.conda_env, "python", "-m", "pip"]

    def prepare_environment(self):
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pycache_dir.mkdir(parents=True, exist_ok=True)
        self.pytest_basetemp_root.mkdir(parents=True, exist_ok=True)

        self.env["HOME"] = str(self.home_dir)
        self.env["XDG_CACHE_HOME"] = str(self.cache_dir)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPYCACHEPREFIX"] = str(self.pycache_dir)
        self.env["TMPDIR"] = str(self.tmp_root)
        self.env["TEMP"] = str(self.tmp_root)
        self.env["TMP"] = str(self.tmp_root)

        print(f"+ Sandbox root: {self.tmp_root}")
        print(f"+ Work dir: {self.work_dir}")

    def uninstall_package(self):
        if not self.pkg_name:
            return
        run_allow_fail(self._pip() + ["uninstall", "-y", self.pkg_name], env=self.env)

    def install_package(self):
        if not self.pkg_name:
            return

        prev_pp = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = f"{self.work_dir}{os.pathsep}{prev_pp}" if prev_pp else str(self.work_dir)

        run_or_die(
            self._pip() + ["install", "-e", ".", "--no-build-isolation"],
            cwd=self.work_dir,
            env=self.env,
            what="pip install -e .",
        )

    def checkout(self):
        if self.use_head:
            shutil.copytree(
                self.repo_root,
                self.work_dir,
                ignore=shutil.ignore_patterns(
                    ".git", ".gitignore", ".gitmodules", ".github", ".gitlab*"
                ),
            )
            return

        if self.sandbox_mode == "archive":
            checkout_commit_with_archive(self.repo_root, self.commit, self.work_dir)
        elif self.sandbox_mode == "copy":
            shutil.copytree(
                self.repo_root,
                self.work_dir,
                ignore=shutil.ignore_patterns(
                    ".git", ".gitignore", ".gitmodules", ".github", ".gitlab*"
                ),
            )
        elif self.sandbox_mode == "symlink":
            shutil.copytree(
                self.repo_root,
                self.work_dir,
                copy_function=os.symlink,
                ignore=shutil.ignore_patterns(
                    ".git", ".gitignore", ".gitmodules", ".github", ".gitlab*"
                ),
                dirs_exist_ok=True,
            )
        else:
            raise ValueError(f"Unknown sandbox_mode: {self.sandbox_mode}")

    def cleanup(self):
        if self.pkg_name:
            self.uninstall_package()
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _make_basetemp(self, tag: str) -> Path:
        p = self.pytest_basetemp_root / tag
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def pytest_cmd(self, extra_args: list[str], basetemp_tag: str) -> list[str]:
        ignore_args = []
        for p in self.ignore_paths:
            ignore_args += ["--ignore", p]

        basetemp = self._make_basetemp(basetemp_tag)

        return (
            ["conda", "run", "-n", self.conda_env, "python", "-m", "pytest"]
            + ["--cache-clear", "--basetemp", str(basetemp)]
            + ignore_args
            + self.test_paths
            + extra_args
        )


# ============================================================
# ProgramCommitRunner
# ============================================================

class ProgramCommitRunner(BaseCommitRunner):
    def run_program(self) -> int:
        self.results_root.mkdir(parents=True, exist_ok=True)
        junit = self.results_root / f"pres_{self.commit}.xml"

        cmd = self.pytest_cmd([f"--junitxml={junit}"], basetemp_tag=f"program-{self.commit}")
        rc = run_allow_fail(cmd, cwd=self.work_dir, env=self.env)

        inject_properties(junit, {
            "commit": self.commit,
            "mode": "program",
            "pytest_exit_code": str(rc),
        })

        print(f"+ program rc={rc} junit={junit}")
        return rc

    def run(self) -> int:
        try:
            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()
            return self.run_program()
        finally:
            self.cleanup()


# ============================================================
# MutantCommitRunner
# ============================================================

class MutantCommitRunner(BaseCommitRunner):
    def __init__(self, *args, mutants: list[dict], **kwargs):
        super().__init__(*args, **kwargs)
        self.mutants = mutants
        self.original_cache: dict[str, str] = {}
        self.program_junit_path: Path | None = None

    def preload_originals(self):
        for mu in self.mutants:
            rel = Path(mu["file"])
            abs_path = self.work_dir / rel
            key = rel.as_posix()
            if key not in self.original_cache:
                self.original_cache[key] = abs_path.read_text(encoding="utf-8", errors="ignore")

    def apply_mutant(self, mu: dict):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        original = self.original_cache[rel.as_posix()]
        mutated = reconstruct_mutant_code(original, mu)
        if abs_path.is_symlink():
            abs_path.unlink()
        abs_path.write_text(mutated, encoding="utf-8")

    def restore(self, mu: dict):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        original = self.original_cache[rel.as_posix()]
        if abs_path.is_symlink():
            abs_path.unlink()
        abs_path.write_text(original, encoding="utf-8")

    def run_single_mutant(self, mu: dict) -> int:
        self.results_root.mkdir(parents=True, exist_ok=True)

        safe = sanitize_mutant_key(mu["mutant_key"])
        junit = self.results_root / f"mut_{safe}_{self.commit}.xml"

        self.apply_mutant(mu)

        cmd = self.pytest_cmd([f"--junitxml={junit}"], basetemp_tag=f"mut-{safe}")
        rc = run_allow_fail(cmd, cwd=self.work_dir, env=self.env)

        inject_properties(junit, {
            "commit": self.commit,
            "mode": "mutant",
            "pytest_exit_code": str(rc),
            "mutant_key": mu["mutant_key"],
            "mutant_id": str(mu.get("mutant_id", "")),
            "mutant_file": mu.get("file", ""),
            "mutant_op_type": mu.get("op_type", ""),
            "mutant_scope": mu.get("scope", ""),
        })

        self.restore(mu)
        return rc

    def run(self) -> None:
        try:
            if self.program_junit_path is None:
                raise RuntimeError("Mutant runner requires program_junit_path")

            v = read_property_from_junit(self.program_junit_path, "pytest_exit_code")
            if v is None:
                print(f"+ Cannot read pytest_exit_code from {self.program_junit_path}; skipping mutants.")
                return

            try:
                program_rc = int(v)
            except ValueError:
                print(f"+ Invalid pytest_exit_code={v!r} in {self.program_junit_path}; skipping mutants.")
                return

            if program_rc not in (0, 1):
                print(f"+ Program rc={program_rc} (not 0/1): skipping mutants for commit {self.commit}.")
                return

            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()

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

    p.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Paths to ignore (passed through to pytest as --ignore). Can be repeated.",
    )

    p.add_argument("--sandbox-mode", choices=["archive", "copy", "symlink"], default="archive")
    p.add_argument("--use-head", action="store_true")

    p.add_argument("--mutant-history")
    p.add_argument("--mutant-limit", type=int)

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

    # load mutants list if needed
    mutants: list[dict] = []
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

    program_junit_path = results_dir / f"pres_{commit}.xml"

    # PROGRAM
    if args.run in ("program", "both"):
        prog = ProgramCommitRunner(
            repo_root=repo_root,
            commit=commit,
            conda_env=args.conda_env,
            pkg_name=args.pkg_name,
            results_root=results_dir,
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
            repo_root=repo_root,
            commit=commit,
            conda_env=args.conda_env,
            pkg_name=args.pkg_name,
            results_root=results_dir,
            use_head=args.use_head,
            sandbox_mode=args.sandbox_mode,
            test_paths=args.tests,
            ignore_paths=args.ignore,
            mutants=job_mutants,
        )
        mut.program_junit_path = program_junit_path
        mut.run()
