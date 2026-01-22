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
        junit_json: Path,
        ndjson: bool,
        append: bool,
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

        self.junit_json = junit_json
        self.ndjson = ndjson
        self.append = append

        self.env = os.environ.copy()

        base_tmp = self.env.get("TMPDIR", "/tmp")
        self.tmp_root = Path(tempfile.mkdtemp(prefix="sandbox-", dir=base_tmp))
        self.work_dir = self.tmp_root / "work"
        self.home_dir = self.tmp_root / "home"
        self.pytest_basetemp_root = self.tmp_root / "pytest_basetemp"

    def prepare_environment(self):
        for p in (self.home_dir, self.pytest_basetemp_root):
            p.mkdir(parents=True, exist_ok=True)

        self.env["HOME"] = str(self.home_dir)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPYCACHEPREFIX"] = str(self.tmp_root / "pycache")

    def uninstall_package(self):
        if self.pkg_name:
            run_allow_fail(
                ["conda", "run", "-n", self.conda_env, "python", "-m", "pip", "uninstall", "-y", self.pkg_name],
                env=self.env,
            )

    def install_package(self):
        if not self.pkg_name:
            return
        run_allow_fail(
            ["conda", "run", "-n", self.conda_env, "python", "-m", "pip", "install", "-e", "."],
            cwd=self.work_dir,
            env=self.env,
        )

    def checkout(self):
        if self.use_head:
            shutil.copytree(self.repo_root, self.work_dir, symlinks=True)
            return

        if self.sandbox_mode == "archive":
            checkout_commit_with_archive(self.repo_root, self.commit, self.work_dir)
        else:
            shutil.copytree(self.repo_root, self.work_dir, symlinks=True)

    def pytest_cmd(self, extra_args: list[str], basetemp_tag: str, meta: dict[str, object]) -> list[str]:
        basetemp = self.pytest_basetemp_root / basetemp_tag
        if basetemp.exists():
            shutil.rmtree(basetemp, ignore_errors=True)
        basetemp.mkdir(parents=True, exist_ok=True)

        ignore_args = []
        for p in self.ignore_paths:
            ignore_args += ["--ignore", p]

        junit_args = [
            f"--junit-json={self.junit_json}",
            "--junit-json-ndjson" if self.ndjson else "",
            "--junit-json-append" if self.append else "",
        ]

        for k, v in meta.items():
            junit_args.append(f"--junit-json-meta={k}={v}")

        junit_args = [x for x in junit_args if x]

        return (
            ["conda", "run", "-n", self.conda_env, "python", "-m", "pytest"]
            + ["--cache-clear", "--basetemp", str(basetemp)]
            + ignore_args
            + self.test_paths
            + junit_args
            + extra_args
        )

    def cleanup(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)


# ============================================================
# ProgramCommitRunner
# ============================================================

class ProgramCommitRunner(BaseCommitRunner):
    def run(self) -> int:
        try:
            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()

            meta = {
                "mode": "program",
                "commit": self.commit,
            }

            cmd = self.pytest_cmd([], f"program-{self.commit}", meta)
            rc = run_allow_fail(cmd, cwd=self.work_dir, env=self.env)

            print(f"+ program rc={rc}")
            return rc
        finally:
            self.cleanup()


# ============================================================
# MutantCommitRunner
# ============================================================

class MutantCommitRunner(BaseCommitRunner):
    def __init__(self, *args, mutants: list[dict], program_rc: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutants = mutants
        self.program_rc = program_rc
        self.original_cache: dict[str, str] = {}

    def preload_originals(self):
        for mu in self.mutants:
            p = self.work_dir / mu["file"]
            self.original_cache[mu["file"]] = p.read_text(encoding="utf-8", errors="ignore")

    def apply_mutant(self, mu: dict):
        p = self.work_dir / mu["file"]
        p.write_text(reconstruct_mutant_code(self.original_cache[mu["file"]], mu), encoding="utf-8")

    def restore(self, mu: dict):
        p = self.work_dir / mu["file"]
        p.write_text(self.original_cache[mu["file"]], encoding="utf-8")

    def run(self):
        if self.program_rc not in (0, 1):
            print(f"+ skipping mutants (program rc={self.program_rc})")
            return

        try:
            self.prepare_environment()
            self.uninstall_package()
            self.checkout()
            self.install_package()

            self.preload_originals()

            for mu in self.mutants:
                self.apply_mutant(mu)

                meta = {
                    "mode": "mutant",
                    "commit": self.commit,
                    "mutant_key": mu["mutant_key"],
                    "mutant_id": mu["mutant_id"],
                    "file": mu["file"],
                    "op_type": mu["op_type"],
                    "scope": mu["scope"],
                }

                cmd = self.pytest_cmd([], f"mut-{sanitize_mutant_key(mu['mutant_key'])}", meta)
                run_allow_fail(cmd, cwd=self.work_dir, env=self.env)

                self.restore(mu)
        finally:
            self.cleanup()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--conda-env", required=True)
    p.add_argument("--pkg-name")
    p.add_argument("--results", required=True)
    p.add_argument("--run", choices=["program", "mutant", "both"], default="both")
    p.add_argument("--tests", nargs="*", default=[])
    p.add_argument("--ignore", action="append", default=[])
    p.add_argument("--sandbox-mode", default="archive")
    p.add_argument("--use-head", action="store_true")
    p.add_argument("--mutant-history")
    p.add_argument("--mutant-limit", type=int)
    p.add_argument("--njobs", type=int, default=1)
    p.add_argument("--jid", type=int, default=0)

    # JSON reporter
    p.add_argument("--junit-json", required=True)
    p.add_argument("--junit-json-ndjson", action="store_true")
    p.add_argument("--junit-json-append", action="store_true")

    args = p.parse_args()

    repo_root = Path(args.repo).resolve()
    results_root = Path(args.results).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    program_rc = 0

    if args.run in ("program", "both"):
        program_rc = ProgramCommitRunner(
            repo_root,
            args.commit,
            args.conda_env,
            args.pkg_name,
            results_root,
            args.use_head,
            args.sandbox_mode,
            args.tests,
            args.ignore,
            Path(args.junit_json),
            args.junit_json_ndjson,
            args.junit_json_append,
        ).run()

    if args.run in ("mutant", "both"):
        if not args.mutant_history:
            raise SystemExit("--mutant-history required")

        mutants = get_mutants(args.commit, Path(args.mutant_history))
        if args.mutant_limit:
            mutants = mutants[: args.mutant_limit]

        start, end = compute_partition_indices(len(mutants), args.njobs, args.jid)
        job_mutants = mutants[start:end]

        MutantCommitRunner(
            repo_root,
            args.commit,
            args.conda_env,
            args.pkg_name,
            results_root,
            args.use_head,
            args.sandbox_mode,
            args.tests,
            args.ignore,
            Path(args.junit_json),
            args.junit_json_ndjson,
            args.junit_json_append,
            mutants=job_mutants,
            program_rc=program_rc,
        ).run()
