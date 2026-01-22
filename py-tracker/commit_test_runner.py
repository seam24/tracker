#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


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


def run_allow_fail(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
                   log_file: Path | None = None) -> int:
    print(f"+ (cwd={cwd}) {' '.join(cmd)}")
    cwd_s = str(cwd) if cwd is not None else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=cwd_s, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.run(cmd, cwd=cwd_s, env=env)

    if proc.returncode != 0:
        print(f"!! Command exited with {proc.returncode}")
    return proc.returncode


def run_or_die(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
               what: str = "command", log_file: Path | None = None) -> int:
    rc = run_allow_fail(cmd, cwd=cwd, env=env, log_file=log_file)
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

    chosen: list[dict] = []
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
                "mutant_id": info.get("mutant_id"),
                "file": info["file"],
                "module": info.get("module", ""),
                "op_type": info.get("op_type", ""),
                "scope": info.get("scope", ""),
                "start_pos": start,
                "end_pos": end,
                "mutant_value": info["mutant_value"],
                "commit": commit,
            }
        )
    return chosen


def coerce_meta_scalar(v: Any) -> str:
    """
    We pass meta via CLI: --junit-json-meta=k=v
    Keep it stable/portable by stringifying scalars.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


# ============================================================
# Strong environment helpers (like your XML runner)
# ============================================================

def guess_base_tmp(env: dict[str, str]) -> Path:
    base = (
        env.get("TMPDIR")
        or env.get("SLURM_TMPDIR")
        or env.get("TEMP")
        or env.get("TMP")
        or "/tmp"
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_repair_build_toolchain(conda_env: str, env: dict[str, str]) -> None:
    """
    Best-effort "make installs work across old/new projects".
    We do NOT fail the run if this fails.
    """
    # pip/setuptools/packaging are the usual culprits (your canonicalize_version error)
    run_allow_fail(
        ["conda", "run", "-n", conda_env, "python", "-m", "pip", "install", "-U",
         "pip", "setuptools", "wheel", "packaging"],
        env=env,
    )


def maybe_repair_httpx(conda_env: str, env: dict[str, str]) -> None:
    """
    Optional: If you used to have an 'httpx repair', keep it best-effort.
    This is intentionally conservative: upgrade httpx + deps.
    """
    run_allow_fail(
        ["conda", "run", "-n", conda_env, "python", "-m", "pip", "install", "-U",
         "httpx", "httpcore", "h11", "anyio", "sniffio", "certifi"],
        env=env,
    )


def should_retry_install_from_error(output_text: str) -> bool:
    # The exact error you hit:
    # TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
    if "canonicalize_version()" in output_text and "strip_trailing_zero" in output_text:
        return True
    # Generic metadata/build failures sometimes fixed by toolchain refresh
    if "metadata-generation-failed" in output_text:
        return True
    if "subprocess-exited-with-error" in output_text and "egg_info" in output_text:
        return True
    return False


# ============================================================
# Base runner
# ============================================================

class BaseCommitRunner:
    """
    Strong runner (JSON report) with robustness similar to your XML runner:
      - sandboxed HOME, XDG_CACHE_HOME, TMPDIR/TEMP/TMP
      - isolated pytest basetemp inside sandbox
      - .git ignored on copy
      - editable install with fatal failure (run_or_die)
      - best-effort toolchain repair + retry-on-known-install-failure
      - optional httpx repair
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
        junit_json: Path,
        ndjson: bool,
        append: bool,
        junit_meta: list[str] | None,
        repair_httpx: bool,
        repair_build: bool,
        pip_no_build_isolation: bool,
        pip_extra_args: list[str] | None,
        pytest_extra_args: list[str] | None,
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
        self.ndjson = bool(ndjson)
        self.append = bool(append)
        self.junit_meta = list(junit_meta) if junit_meta else []

        self.repair_httpx = bool(repair_httpx)
        self.repair_build = bool(repair_build)
        self.pip_no_build_isolation = bool(pip_no_build_isolation)
        self.pip_extra_args = list(pip_extra_args) if pip_extra_args else []
        self.pytest_extra_args = list(pytest_extra_args) if pytest_extra_args else []

        self.env = os.environ.copy()

        base_tmp = guess_base_tmp(self.env)
        self.tmp_root = Path(tempfile.mkdtemp(prefix="sandbox-", dir=str(base_tmp)))
        self.work_dir = self.tmp_root / "work"
        self.home_dir = self.tmp_root / "home"
        self.cache_dir = self.tmp_root / "cache"
        self.pycache_dir = self.tmp_root / "pycache"
        self.pytest_basetemp_root = self.tmp_root / "pytest_basetemp"

    # ----------------------------
    # command helpers
    # ----------------------------
    def _pip(self) -> list[str]:
        return ["conda", "run", "-n", self.conda_env, "python", "-m", "pip"]

    def _pytest(self) -> list[str]:
        return ["conda", "run", "-n", self.conda_env, "python", "-m", "pytest"]

    # ----------------------------
    # sandbox env
    # ----------------------------
    def prepare_environment(self):
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pycache_dir.mkdir(parents=True, exist_ok=True)
        self.pytest_basetemp_root.mkdir(parents=True, exist_ok=True)

        self.env["HOME"] = str(self.home_dir)
        self.env["XDG_CACHE_HOME"] = str(self.cache_dir)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPYCACHEPREFIX"] = str(self.pycache_dir)

        # make tmp usage consistent across tools
        self.env["TMPDIR"] = str(self.tmp_root)
        self.env["TEMP"] = str(self.tmp_root)
        self.env["TMP"] = str(self.tmp_root)

        print(f"+ Sandbox root: {self.tmp_root}")
        print(f"+ Work dir: {self.work_dir}")
        print(f"+ Isolated HOME: {self.home_dir}")
        print(f"+ Isolated XDG_CACHE_HOME: {self.cache_dir}")

    # ----------------------------
    # checkout
    # ----------------------------
    def checkout(self):
        ignore = shutil.ignore_patterns(".git", ".gitignore", ".gitmodules", ".github", ".gitlab*")

        if self.use_head:
            shutil.copytree(self.repo_root, self.work_dir, ignore=ignore)
            return

        if self.sandbox_mode == "archive":
            checkout_commit_with_archive(self.repo_root, self.commit, self.work_dir)
        elif self.sandbox_mode == "copy":
            shutil.copytree(self.repo_root, self.work_dir, ignore=ignore)
        elif self.sandbox_mode == "symlink":
            shutil.copytree(
                self.repo_root,
                self.work_dir,
                copy_function=os.symlink,
                ignore=ignore,
                dirs_exist_ok=True,
            )
        else:
            raise ValueError(f"Unknown sandbox_mode: {self.sandbox_mode}")

    # ----------------------------
    # install/uninstall
    # ----------------------------
    def uninstall_package(self):
        if not self.pkg_name:
            return
        run_allow_fail(self._pip() + ["uninstall", "-y", self.pkg_name], env=self.env)

    def _set_pythonpath_for_editable(self):
        prev_pp = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = f"{self.work_dir}{os.pathsep}{prev_pp}" if prev_pp else str(self.work_dir)

    def install_package(self):
        if not self.pkg_name:
            return

        self._set_pythonpath_for_editable()

        if self.repair_build:
            maybe_repair_build_toolchain(self.conda_env, self.env)

        if self.repair_httpx:
            maybe_repair_httpx(self.conda_env, self.env)

        install_cmd = self._pip() + ["install", "-e", "."]
        if self.pip_no_build_isolation:
            install_cmd.append("--no-build-isolation")
        install_cmd += self.pip_extra_args

        # Try once; if it fails with known toolchain mismatch, repair + retry.
        log_path = self.tmp_root / "pip_install.log"
        rc = run_allow_fail(install_cmd, cwd=self.work_dir, env=self.env, log_file=log_path)
        if rc == 0:
            return

        txt = ""
        try:
            txt = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

        if should_retry_install_from_error(txt):
            print("+ pip install failed with known pattern; attempting toolchain refresh and retry")
            maybe_repair_build_toolchain(self.conda_env, self.env)
            if self.repair_httpx:
                maybe_repair_httpx(self.conda_env, self.env)
            run_or_die(install_cmd, cwd=self.work_dir, env=self.env, what="pip install -e . (retry)", log_file=log_path)
        else:
            # fatal like XML runner
            raise RuntimeError(f"pip install -e . failed (see {log_path})")

    # ----------------------------
    # pytest cmd
    # ----------------------------
    def _make_basetemp(self, tag: str) -> Path:
        p = self.pytest_basetemp_root / tag
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def pytest_cmd(self, basetemp_tag: str, meta: dict[str, Any], extra_args: list[str] | None = None) -> list[str]:
        ignore_args: list[str] = []
        for p in self.ignore_paths:
            ignore_args += ["--ignore", p]

        basetemp = self._make_basetemp(basetemp_tag)

        # ensure target dir exists (especially if user gave absolute windows-mounted path)
        self.junit_json.parent.mkdir(parents=True, exist_ok=True)

        junit_args: list[str] = [f"--junit-json={self.junit_json}"]
        if self.ndjson:
            junit_args.append("--junit-json-ndjson")
        if self.append:
            junit_args.append("--junit-json-append")

        # add user-provided meta pairs, then runner-generated meta
        all_meta: list[tuple[str, str]] = []
        for kv in self.junit_meta:
            if "=" in kv:
                k, v = kv.split("=", 1)
                all_meta.append((k, v))
            else:
                all_meta.append((kv, ""))

        for k, v in meta.items():
            all_meta.append((k, coerce_meta_scalar(v)))

        # pass as CLI options
        for k, v in all_meta:
            # keep it simple; users can avoid '=' in values if needed
            junit_args.append(f"--junit-json-meta={k}={v}")

        cmd = (
            self._pytest()
            + ["--cache-clear", "--basetemp", str(basetemp)]
            + ignore_args
            + self.test_paths
            + junit_args
            + self.pytest_extra_args
        )

        if extra_args:
            cmd += extra_args

        return cmd

    # ----------------------------
    # cleanup
    # ----------------------------
    def cleanup(self):
        # mimic XML runner behavior: uninstall then remove sandbox
        try:
            if self.pkg_name:
                self.uninstall_package()
        finally:
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
            cmd = self.pytest_cmd(basetemp_tag=f"program-{self.commit}", meta=meta)
            rc = run_allow_fail(cmd, cwd=self.work_dir, env=self.env)

            print(f"+ program rc={rc} junit_json={self.junit_json}")
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
            rel = Path(mu["file"])
            abs_path = self.work_dir / rel
            key = rel.as_posix()
            if key not in self.original_cache:
                self.original_cache[key] = abs_path.read_text(encoding="utf-8", errors="ignore")

    def apply_mutant(self, mu: dict):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        key = rel.as_posix()
        abs_path.write_text(reconstruct_mutant_code(self.original_cache[key], mu), encoding="utf-8")

    def restore(self, mu: dict):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        key = rel.as_posix()
        abs_path.write_text(self.original_cache[key], encoding="utf-8")

    def run(self) -> None:
        # keep your old gating logic
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

                safe = sanitize_mutant_key(mu["mutant_key"])
                meta = {
                    "mode": "mutant",
                    "commit": self.commit,
                    "mutant_key": mu["mutant_key"],
                    "mutant_id": mu.get("mutant_id", ""),
                    "file": mu.get("file", ""),
                    "op_type": mu.get("op_type", ""),
                    "scope": mu.get("scope", ""),
                }

                cmd = self.pytest_cmd(
                    basetemp_tag=f"mut-{safe}",
                    meta=meta,
                )
                run_allow_fail(cmd, cwd=self.work_dir, env=self.env)

                self.restore(mu)
        finally:
            self.cleanup()


# ============================================================
# CLI
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
    p.add_argument("--ignore", action="append", default=[])

    p.add_argument("--sandbox-mode", choices=["archive", "copy", "symlink"], default="archive")
    p.add_argument("--use-head", action="store_true")

    p.add_argument("--mutant-history")
    p.add_argument("--mutant-limit", type=int)

    p.add_argument("--njobs", type=int, default=1)
    p.add_argument("--jid", type=int, default=0)

    # JSON reporter options
    p.add_argument("--junit-json", required=True, help="Path to JSON report (can be NDJSON).")
    p.add_argument("--junit-json-ndjson", action="store_true")
    p.add_argument("--junit-json-append", action="store_true")
    p.add_argument("--junit-json-meta", action="append", default=[],
                   help="Extra meta k=v passed through to the plugin (repeatable).")

    # robustness toggles
    p.add_argument("--repair-build-toolchain", action="store_true",
                   help="Best-effort upgrade pip/setuptools/packaging before install.")
    p.add_argument("--repair-httpx", action="store_true",
                   help="Best-effort upgrade httpx stack before install.")
    p.add_argument("--pip-no-build-isolation", action="store_true",
                   help="Pass --no-build-isolation to pip install -e .")
    p.add_argument("--pip-extra-args", action="append", default=[],
                   help="Extra args to pip install (repeatable; e.g. --pip-extra-args=--no-deps)")
    p.add_argument("--pytest-extra-args", action="append", default=[],
                   help="Extra args to pytest (repeatable; e.g. --pytest-extra-args=-q)")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    repo_root = Path(args.repo).resolve()
    results_root = Path(args.results).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    # Resolve junit path strongly (fixes "written somewhere else" confusion)
    junit_path = Path(args.junit_json).expanduser()
    if not junit_path.is_absolute():
        # interpret relative to current working directory, not sandbox cwd
        junit_path = (Path.cwd() / junit_path).resolve()
    else:
        junit_path = junit_path.resolve()

    commit = args.commit

    program_rc = 0

    # PROGRAM
    if args.run in ("program", "both"):
        prog = ProgramCommitRunner(
            repo_root=repo_root,
            commit=commit,
            conda_env=args.conda_env,
            pkg_name=args.pkg_name,
            results_root=results_root,
            use_head=args.use_head,
            sandbox_mode=args.sandbox_mode,
            test_paths=args.tests,
            ignore_paths=args.ignore,
            junit_json=junit_path,
            ndjson=args.junit_json_ndjson,
            append=args.junit_json_append,
            junit_meta=args.junit_json_meta,
            repair_httpx=args.repair_httpx,
            repair_build=args.repair_build_toolchain,
            pip_no_build_isolation=args.pip_no_build_isolation,
            pip_extra_args=args.pip_extra_args,
            pytest_extra_args=args.pytest_extra_args,
        )
        program_rc = prog.run()

    # MUTANT
    if args.run in ("mutant", "both"):
        if not args.mutant_history:
            raise SystemExit("--mutant-history required for mutant mode")

        mutants = get_mutants(commit, Path(args.mutant_history))
        if args.mutant_limit:
            mutants = mutants[: args.mutant_limit]

        start, end = compute_partition_indices(len(mutants), max(1, args.njobs), args.jid)
        job_mutants = mutants[start:end]

        print(f"+ Mutant partition: total={len(mutants)}, njobs={max(1, args.njobs)}, jid={args.jid}, this_job={len(job_mutants)}")

        mut = MutantCommitRunner(
            repo_root=repo_root,
            commit=commit,
            conda_env=args.conda_env,
            pkg_name=args.pkg_name,
            results_root=results_root,
            use_head=args.use_head,
            sandbox_mode=args.sandbox_mode,
            test_paths=args.tests,
            ignore_paths=args.ignore,
            junit_json=junit_path,
            ndjson=args.junit_json_ndjson,
            append=args.junit_json_append,
            junit_meta=args.junit_json_meta,
            repair_httpx=args.repair_httpx,
            repair_build=args.repair_build_toolchain,
            pip_no_build_isolation=args.pip_no_build_isolation,
            pip_extra_args=args.pip_extra_args,
            pytest_extra_args=args.pytest_extra_args,
            mutants=job_mutants,
            program_rc=program_rc,
        )
        mut.run()
