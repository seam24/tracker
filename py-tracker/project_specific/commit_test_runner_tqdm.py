#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
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


def run_allow_fail(cmd, cwd=None, env=None) -> int:
    print(f"+ (cwd={cwd}) {' '.join(cmd)}")
    if cwd is not None:
        cwd = str(cwd)

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


def meta_args_from_dict(meta: dict) -> list[str]:
    """
    Convert {"k": v} -> ["--json-report-meta", "k=v", ...]
    Values are stringified; the plugin will parse int/float if possible.
    """
    out = []
    for k, v in meta.items():
        if v is None:
            continue
        out += ["--json-report-meta", f"{k}={v}"]
    return out


def find_executable(name: str) -> str | None:
    """
    Best-effort lookup for tools like `http` / `httpie` / etc.
    Kept to preserve your 'best tries' behavior for tools that might not exist.
    """
    return shutil.which(name)


def ensure_httpie_env_fix(env: dict) -> dict:
    """
    Best-effort fixups for httpie / requests-ish tooling:
    - deterministic encoding
    - sandbox-friendly defaults
    """
    e = dict(env)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    e.setdefault("LC_ALL", "C.UTF-8")
    e.setdefault("LANG", "C.UTF-8")
    return e


def default_ndjson_path(results_root: Path, commit: str, jid: int | None = None) -> Path:
    """
    Sequential mode (default): one NDJSON file per commit.
    Parallel mutant mode (njobs>1): per-job NDJSON file to avoid concurrent appends.
    """
    if jid is None:
        return results_root / f"results_{commit}.ndjson"
    return results_root / f"results_{commit}_jid{jid}.ndjson"


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
        json_duration: str = "call",
        ndjson_path: Path | None = None,
        timeout: int | None = None,
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
        self.json_duration = json_duration
        self.ndjson_path = ndjson_path
        self.timeout = timeout  # pytest-timeout seconds (optional)

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

        # Isolate HOME
        self.env["HOME"] = str(self.home_dir)

        # Keep common caches inside sandbox
        self.env["XDG_CACHE_HOME"] = str(self.cache_dir)

        # Make python bytecode deterministic / isolated
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPYCACHEPREFIX"] = str(self.pycache_dir)

        # Keep best-effort environment tweaks
        self.env = ensure_httpie_env_fix(self.env)

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

    def clear_ndjson_file(self, ndjson_path: Path):
        """
        Clear the NDJSON output once at the start of a sequence so the next
        append writes the first line (program).
        """
        try:
            ndjson_path.unlink()
        except Exception:
            pass

    def expand_paths_in_workdir(self, paths: list[str], *, kind: str) -> list[str]:
        """
        Expand globs relative to self.work_dir.

        If a pattern matches, return the matched paths relative to self.work_dir.
        If it does not match, keep the original path unchanged so pytest can still
        handle plain paths/nodeids.

        kind is only for logging ("test" or "ignore").
        """
        expanded: list[str] = []

        for raw in paths:
            abs_pattern = self.work_dir / raw
            matches = sorted(glob.glob(str(abs_pattern), recursive=True))

            if matches:
                rel_matches = [os.path.relpath(m, self.work_dir) for m in matches]
                print(f"+ Expanded {kind} pattern: {raw} -> {len(rel_matches)} match(es)")
                for m in rel_matches:
                    print(f"  - {m}")
                expanded.extend(rel_matches)
            else:
                print(f"+ No {kind} glob matches for: {raw} (passing through as-is)")
                expanded.append(raw)

        return expanded

    def conda_pytest(
        self,
        pytest_args: list[str],
        *,
        cwd=None,
        run_tag: str = "run",
        ndjson_path: Path | None = None,
        json_meta: dict | None = None,
        json_append: bool = True,
    ) -> int:
        """
        Run pytest with cache-clearing + unique basetemp.
        Uses your NDJSON plugin as the ONLY output artifact.

        IMPORTANT: In your desired workflow, json_append=True even for the program run.
        """
        self.clear_workdir_caches()

        basetemp = self.pytest_basetemp_root / run_tag
        if basetemp.exists():
            shutil.rmtree(basetemp, ignore_errors=True)
        basetemp.mkdir(parents=True, exist_ok=True)

        expanded_test_paths = self.expand_paths_in_workdir(self.test_paths, kind="test")
        expanded_ignore_paths = self.expand_paths_in_workdir(self.ignore_paths, kind="ignore")

        ignore_args = []
        for p in expanded_ignore_paths:
            ignore_args += ["--ignore", p]

        forced = ["--cache-clear", "--basetemp", str(basetemp)]

        timeout_args = []
        if self.timeout is not None:
            try:
                t = int(self.timeout)
            except Exception:
                t = self.timeout
            timeout_args += [f"--timeout={t}"]
            print(f"+ pytest-timeout enabled: --timeout={t}")

        report_args = []
        if ndjson_path is not None:
            report_args += ["--json-report", str(ndjson_path)]
            if json_append:
                report_args += ["--json-report-append"]
            if self.json_duration:
                report_args += ["--json-report-duration", str(self.json_duration)]
            if json_meta:
                report_args += meta_args_from_dict(json_meta)

        cmd = (
            ["conda", "run", "-n", self.conda_env, "python", "-m", "pytest"]
            + forced
            + ignore_args
            + timeout_args
            + expanded_test_paths
            + report_args
            + pytest_args
        )
        return run_allow_fail(cmd, cwd=cwd, env=self.env)

    def cleanup(self):
        if self.pkg_name:
            self.uninstall_package()
        shutil.rmtree(self.tmp_root, ignore_errors=True)


# ============================================================
# ProgramCommitRunner
# ============================================================

class ProgramCommitRunner(BaseCommitRunner):
    def run_program_tests(self):
        ndjson_path = self.ndjson_path
        if ndjson_path is None:
            ndjson_path = default_ndjson_path(self.results_root, self.commit)

        # Clear once so the first append becomes the program line.
        self.clear_ndjson_file(ndjson_path)

        self.program_exit_code = self.conda_pytest(
            pytest_args=[],
            cwd=self.work_dir,
            run_tag=f"program-{self.commit}",
            ndjson_path=ndjson_path,
            json_meta={
                "kind": "program",
                "commit": self.commit,
            },
            json_append=True,
        )

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

        # to be set externally if running "both"
        # self.program_exit_code

    def preload_originals(self):
        for mu in self.mutants:
            rel = Path(mu["file"])
            abs_path = self.work_dir / rel
            if rel.as_posix() not in self.original_cache:
                self.original_cache[rel.as_posix()] = abs_path.read_text(encoding="utf-8")

    def apply_mutant(self, mu):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        original = self.original_cache[rel.as_posix()]
        mutated = reconstruct_mutant_code(original, mu)
        if abs_path.is_symlink():
            abs_path.unlink()
        abs_path.write_text(mutated, encoding="utf-8")

    def restore(self, mu):
        rel = Path(mu["file"])
        abs_path = self.work_dir / rel
        original = self.original_cache[rel.as_posix()]
        if abs_path.is_symlink():
            abs_path.unlink()
        abs_path.write_text(original, encoding="utf-8")

    def run_single_mutant(self, mu):
        self.apply_mutant(mu)

        ndjson_path = self.ndjson_path
        if ndjson_path is None:
            ndjson_path = default_ndjson_path(self.results_root, self.commit)

        safe = sanitize_mutant_key(mu["mutant_key"])

        rc = self.conda_pytest(
            pytest_args=[],
            cwd=self.work_dir,
            run_tag=f"mut-{safe}",
            ndjson_path=ndjson_path,
            json_meta={
                "kind": "mutant",
                "commit": self.commit,
                "mutant_key": mu.get("mutant_key"),
                "mutant_id": mu.get("mutant_id"),
                "file": mu.get("file"),
                "module": mu.get("module"),
                "op_type": mu.get("op_type"),
                "scope": mu.get("scope"),
            },
            json_append=True,
        )

        self.restore(mu)
        self.clear_workdir_caches()
        return rc

    def run(self):
        try:
            if hasattr(self, "program_exit_code"):
                if self.program_exit_code not in (0, 1):
                    print(
                        f"+ Skipping mutant mode: program pytest exit code was {self.program_exit_code} "
                        f"(only 0 or 1 allow mutant runs)."
                    )
                    return

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

    p.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Paths or glob patterns to ignore (passed through to pytest as --ignore after expansion)",
    )

    p.add_argument("--sandbox-mode", choices=["archive", "copy", "symlink"], default="archive")
    p.add_argument("--use-head", action="store_true")

    p.add_argument("--mutant-history")
    p.add_argument("--mutant-limit", type=int)

    p.add_argument("--njobs", type=int, default=1, help="Parallel jobs for mutant mode")
    p.add_argument("--jid", type=int, default=0, help="Job index for mutant mode")

    p.add_argument(
        "--json-duration",
        choices=["call", "total"],
        default="call",
        help="Duration mode stored by the NDJSON report plugin.",
    )

    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-test timeout in seconds (pytest-timeout; passed as --timeout).",
    )

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

    # Decide output file:
    # - sequential default (njobs==1): results_<commit>.ndjson
    # - parallel mutant jobs: results_<commit>_jidX.ndjson
    if args.run in ("program", "both"):
        ndjson_path = default_ndjson_path(results_dir, commit)
    else:
        if args.njobs and args.njobs > 1:
            ndjson_path = default_ndjson_path(results_dir, commit, jid=args.jid)
        else:
            ndjson_path = default_ndjson_path(results_dir, commit)

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

    prog = None

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
            json_duration=args.json_duration,
            ndjson_path=ndjson_path,
            timeout=args.timeout,
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
            json_duration=args.json_duration,
            ndjson_path=ndjson_path,
            timeout=args.timeout,
        )

        if args.run == "both" and prog is not None:
            mut.program_exit_code = getattr(prog, "program_exit_code", None)

        mut.run()
