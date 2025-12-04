#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pytest for each mutant in a sandboxed copy of a repo "
                    "using a per-mutant file+span JSON format."
    )

    parser.add_argument(
        "--base-repo",
        required=True,
        help="Root directory of the project (git checkout for a specific commit).",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        required=False,
        help="Test file(s) or directories (relative to base-repo or absolute).",
    )
    parser.add_argument(
        "--prog-res",
        required=True,
        help="JSON file with program execution results (input to pytest).",
    )
    parser.add_argument(
        "--mutants",
        required=True,
        help="JSON file with mutants information (list of dicts).",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Directory where mutant result files will be stored.",
    )

    # pytest / mutant options
    parser.add_argument("--stop-on-kill", action="store_true", default=False)
    parser.add_argument("--timeout-min", type=float, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--disable-incompetent", action="store_true", default=False)

    # partitioning
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="Total number of jobs for partitioning mutants (default: 1)",
    )
    parser.add_argument(
        "--jid",
        type=int,
        default=0,
        help="Job id in [0, njobs-1] indicating which partition to run (default: 0)",
    )

    parser.add_argument(
        "--mutant-ids",
        type=str,
        nargs="+",
        help="Optional list of mutant ids to run (overrides njobs/jid partition if given).",
    )

    parser.add_argument(
        "--sandbox-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Sandbox creation mode: 'copy' (default) or 'symlink'.",
    )

    return parser.parse_args()


def compute_partition_indices(total: int, njobs: int, jid: int):
    """
    Partition [0, total) into njobs almost-equal chunks and return (start, end)
    for the chunk with index jid (0 <= jid < njobs).
    """
    if njobs <= 0:
        raise SystemExit(f"njobs must be >= 1, got {njobs}")
    if jid < 0 or jid >= njobs:
        raise SystemExit(f"jid must be in [0, {njobs - 1}], got {jid}")

    base = total // njobs
    rem = total % njobs  # first 'rem' jobs get one extra

    if jid < rem:
        start = jid * (base + 1)
        end = start + (base + 1)
    else:
        start = rem * (base + 1) + (jid - rem) * base
        end = start + base

    return start, end


def resolve_paths(args):
    """
    Resolve paths before we chdir into the sandbox.
    """
    base_repo = Path(args.base_repo).expanduser().resolve()
    if not base_repo.is_dir():
        raise SystemExit(f"base_repo does not exist or is not a directory: {base_repo}")

    # Tests: resolve, if under base_repo then remember relative path
    test_paths = []
    if args.tests:
        for t in args.tests:
            p = Path(t).expanduser()
            if not p.is_absolute():
                p = (base_repo / p).resolve()
            else:
                p = p.resolve()
            try:
                rel = p.relative_to(base_repo)
                test_paths.append(("in_repo", rel))
            except ValueError:
                # Outside repo; we’ll pass absolute path directly to pytest
                test_paths.append(("external", p))

    prog_res = Path(args.prog_res).expanduser().resolve()
    mutants = Path(args.mutants).expanduser().resolve()
    save_path = Path(args.save_path).expanduser().resolve()

    return {
        "base_repo": base_repo,
        "test_paths": test_paths,
        "prog_res": prog_res,
        "mutants": mutants,
        "save_path": save_path,
    }


def load_mutants(mutants_path: Path):
    """
    Expect a list of dicts, each with at least:
      - file: path inside the repo (e.g. 'cookiecutter/hooks.py')
      - start_pos, end_pos: integer positions in the original source
      - mutant_value: string replacement
    Optionally:
      - mutant_id: integer or string identifier

    We normalize so each mutant has:
      - 'id': used as the mutant identifier
    """
    with mutants_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("mutants JSON must be a list of mutant objects")

    result = []
    for i, mu in enumerate(data):
        if not isinstance(mu, dict):
            raise SystemExit(f"Mutant at index {i} is not an object: {mu!r}")
        mu = dict(mu)
        if "id" not in mu:
            mu["id"] = mu.get("mutant_id", i)
        # sanity check for required fields
        for field in ("file", "start_pos", "end_pos", "mutant_value"):
            if field not in mu:
                raise SystemExit(f"Mutant {mu['id']} missing required field '{field}'")
        result.append(mu)

    return result


def reconstruct_mutant_code(original_src: str, mutant_info: dict) -> str:
    start = mutant_info["start_pos"]
    end = mutant_info["end_pos"]
    replacement = mutant_info["mutant_value"]
    return original_src[:start] + replacement + original_src[end:]


def build_pytest_command(
    sandbox_root: Path,
    test_paths,
    prog_res: Path,
    mutant_results_path: Path,
    args,
):
    cmd = ["pytest"]

    # Add tests: map in-repo ones into sandbox; keep external as absolute
    for kind, p in test_paths:
        if kind == "in_repo":
            cmd.append(str(sandbox_root / p))
        else:
            cmd.append(str(p))

    # Custom options for your pytest plugin
    cmd.extend(["--mode", "mutant"])
    cmd.extend(["--program-results", str(prog_res)])

    if args.stop_on_kill:
        cmd.append("--stop-on-kill")
    if args.timeout_min is not None:
        cmd.extend(["--timeout-min", str(args.timeout_min)])
    if args.timeout_multiplier is not None:
        cmd.extend(["--timeout-multiplier", str(args.timeout_multiplier)])
    if args.disable_incompetent:
        cmd.append("--disable-incompetent")

    cmd.extend(["--mutant-results", str(mutant_results_path)])

    return cmd


def main():
    args = parse_args()
    paths = resolve_paths(args)

    paths["save_path"].mkdir(parents=True, exist_ok=True)

    mutants = load_mutants(paths["mutants"])

    # --- Mutant ID mode (overrides parallel partitioning entirely) ---
    if args.mutant_ids:
        wanted = {str(x) for x in args.mutant_ids}
        mutants = [m for m in mutants if str(m.get("id")) in wanted]

        total_mutants = len(mutants)
        if total_mutants == 0:
            print(f"No mutants matched IDs: {sorted(wanted)}. Exiting.")
            return

        start_idx, end_idx = 0, total_mutants
        job_mutants = mutants  # full set

        print("\n=== MODE: MUTANT-ID SELECTION ===")
        print(f"Requested IDs: {sorted(wanted)}")
        print(f"Matched mutants: {total_mutants}\n")

    # --- Default parallel partitioning mode ---
    else:
        total_mutants = len(mutants)
        if total_mutants == 0:
            print("No mutants found, exiting.")
            return

        start_idx, end_idx = compute_partition_indices(
            total=total_mutants, njobs=args.njobs, jid=args.jid
        )
        job_mutants = mutants[start_idx:end_idx]

        print("\n=== MODE: PARALLEL PARTITIONING ===")
        print(f"Total mutants: {total_mutants}")
        print(f"njobs={args.njobs}, jid={args.jid}")
        print(f"Processing slice [{start_idx}, {end_idx}) -> {len(job_mutants)} mutants\n")

    if not job_mutants:
        print("This job has no mutants to process. Exiting.")
        return

    # Cache original source per file (relative to repo root)
    original_code_cache = {}

    # Create sandbox as a full copy of the repo
    with tempfile.TemporaryDirectory(prefix="mutant-sandbox-") as tmpdir:
        sandbox_root = Path(tmpdir)

        if args.sandbox_mode == "copy":
            shutil.copytree(paths["base_repo"], sandbox_root, dirs_exist_ok=True)
        elif args.sandbox_mode == "symlink":
            shutil.copytree(
                paths["base_repo"],
                sandbox_root,
                copy_function=os.symlink,
                dirs_exist_ok=True,
            )

        #################################################################
        # Main mutation/testing loop
        #################################################################
        for local_idx, mu in enumerate(job_mutants):
            global_idx = start_idx + local_idx
            mu_id = mu.get("id", global_idx)

            rel_file = Path(mu["file"])  # path inside repo, e.g. "cookiecutter/hooks.py"
            sandbox_src_path = sandbox_root / rel_file

            if not sandbox_src_path.exists():
                raise SystemExit(
                    f"Sandbox source path for mutant {mu_id} does not exist: {sandbox_src_path}"
                )

            # Load and cache original code for this file once
            key = rel_file.as_posix()
            if key not in original_code_cache:
                original_code_cache[key] = sandbox_src_path.read_text(encoding="utf-8")
            original_src = original_code_cache[key]

            print(
                f"=== [jid={args.jid}] Running mutant {mu_id} "
                f"(global {global_idx + 1}/{total_mutants}, "
                f"job {local_idx + 1}/{len(job_mutants)}) ==="
            )
            print(f"  File: {rel_file}")

            # Construct mutant code from original source
            mutant_code = reconstruct_mutant_code(original_src, mu)

            # Break symlink safely, then write mutated file only in sandbox
            if sandbox_src_path.is_symlink():
                sandbox_src_path.unlink()
            sandbox_src_path.write_text(mutant_code, encoding="utf-8")

            # Where to save results for this mutant
            mutant_results_path = paths["save_path"] / f"mutant_{mu_id}.json"

            # Build pytest command
            cmd = build_pytest_command(
                sandbox_root=sandbox_root,
                test_paths=paths["test_paths"],
                prog_res=paths["prog_res"],
                mutant_results_path=mutant_results_path,
                args=args,
            )

            # Run pytest with sandbox as cwd
            print("Command:", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=sandbox_root,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

            if result.returncode != 0:
                print(
                    f"Pytest returned non-zero exit code ({result.returncode}) for mutant {mu_id}",
                    file=sys.stderr,
                )

            # We do not need to restore the file: on the next mutant for this file,
            # we always rebuild from original_code_cache[key].

    print("Done.")


if __name__ == "__main__":
    main()
