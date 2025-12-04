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
        description="Run pytest for each mutant in a sandboxed repo using a per-mutant file+span JSON format."
    )

    parser.add_argument("--base-repo", required=True,
                        help="Root directory of the project for a specific commit.")
    parser.add_argument("--tests", nargs="+", required=False,
                        help="Test file(s) or directories.")
    parser.add_argument("--prog-res", required=True,
                        help="Program-mode execution results JSON.")
    parser.add_argument("--mutants", required=True,
                        help="Mutants JSON file.")
    parser.add_argument("--save-path", required=True,
                        help="Directory where mutant result files will be stored.")

    # pytest plugin options
    parser.add_argument("--stop-on-kill", action="store_true", default=False)
    parser.add_argument("--timeout-min", type=float, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--disable-incompetent", action="store_true", default=False)

    # parallel partitioning
    parser.add_argument("--njobs", type=int, default=1,
                        help="Total number of jobs for partitioning mutants (default: 1)")
    parser.add_argument("--jid", type=int, default=0,
                        help="Job id in [0, njobs-1] (default: 0)")

    # selection by mutant_key
    parser.add_argument("--mutant-ids", type=str, nargs="+",
                        help="Run only mutants whose mutant_key matches these.")

    parser.add_argument("--sandbox-mode", choices=["copy", "symlink"],
                        default="copy")

    return parser.parse_args()


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


def resolve_paths(args):
    base_repo = Path(args.base_repo).expanduser().resolve()
    if not base_repo.is_dir():
        raise SystemExit(f"base_repo is not a directory: {base_repo}")

    # Resolve tests
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
                test_paths.append(("external", p))

    return {
        "base_repo": base_repo,
        "test_paths": test_paths,
        "prog_res": Path(args.prog_res).resolve(),
        "mutants": Path(args.mutants).resolve(),
        "save_path": Path(args.save_path).resolve(),
    }


def load_mutants(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("Mutants JSON must be a list")

    result = []
    for i, mu in enumerate(data):
        if not isinstance(mu, dict):
            raise SystemExit(f"Mutant #{i} is not a JSON object")

        mu = dict(mu)

        if "mutant_key" not in mu:
            raise SystemExit(f"Mutant #{i} missing mutant_key")

        if "mutant_id" not in mu:
            mu["mutant_id"] = i

        for field in ("file", "start_pos", "end_pos", "mutant_value"):
            if field not in mu:
                raise SystemExit(f"Mutant {mu['mutant_key']} missing field '{field}'")

        result.append(mu)

    return result


def reconstruct_mutant_code(original, mu):
    return original[:mu["start_pos"]] + mu["mutant_value"] + original[mu["end_pos"]:]


def build_pytest_command(sandbox_root, test_paths, prog_res, inner_results_path, args):
    """
    inner_results_path: path INSIDE THE SANDBOX where pytest-mutation-runner
    will write its raw results mapping (test -> status).
    """
    cmd = ["pytest"]

    for kind, p in test_paths:
        if kind == "in_repo":
            cmd.append(str(sandbox_root / p))
        else:
            cmd.append(str(p))

    cmd += ["--mode", "mutant",
            "--program-results", str(prog_res),
            "--mutant-results", str(inner_results_path)]

    if args.stop_on_kill:
        cmd.append("--stop-on-kill")
    if args.timeout_min is not None:
        cmd += ["--timeout-min", str(args.timeout_min)]
    if args.timeout_multiplier is not None:
        cmd += ["--timeout-multiplier", str(args.timeout_multiplier)]
    if args.disable_incompetent:
        cmd.append("--disable-incompetent")

    return cmd


def sanitize_mutant_key(mutant_key: str) -> str:
    return mutant_key.replace("/", "_").replace("#", "_").replace('.','_').replace(os.sep, "_")


def main():
    args = parse_args()
    paths = resolve_paths(args)

    paths["save_path"].mkdir(parents=True, exist_ok=True)

    mutants = load_mutants(paths["mutants"])

    # Filter by mutant_key if requested
    if args.mutant_ids:
        wanted = {str(x) for x in args.mutant_ids}
        mutants = [m for m in mutants if m["mutant_key"] in wanted]

    if not mutants:
        print("No mutants to process.")
        return

    # Partitioning
    start, end = compute_partition_indices(len(mutants), args.njobs, args.jid)
    job_mutants = mutants[start:end]

    if not job_mutants:
        print("Job has no mutants to run.")
        return

    # Cache original content per file
    original_cache: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="mutant-sandbox-") as tmpdir:
        sandbox_root = Path(tmpdir)

        if args.sandbox_mode == "copy":
            shutil.copytree(paths["base_repo"], sandbox_root, dirs_exist_ok=True)
        else:
            shutil.copytree(paths["base_repo"], sandbox_root,
                            copy_function=os.symlink, dirs_exist_ok=True)

        # Temp file name (in sandbox) that pytest-mutation-runner writes into
        inner_results_name = ".mutant_inner_results.json"
        inner_results_path = sandbox_root / inner_results_name

        for mu in job_mutants:
            mutant_key = mu["mutant_key"]
            mutant_id = mu["mutant_id"]
            rel_file = Path(mu["file"])
            sandbox_file = sandbox_root / rel_file

            if not sandbox_file.exists() and not sandbox_file.is_symlink():
                raise SystemExit(
                    f"Sandbox source path for mutant {mutant_key} does not exist: {sandbox_file}"
                )

            # Load original source if needed
            key = rel_file.as_posix()
            if key not in original_cache:
                original_cache[key] = sandbox_file.read_text(encoding="utf-8")
            original_src = original_cache[key]

            print(f"=== Running mutant {mutant_key} (id={mutant_id}) ===")

            # Construct and write mutated file
            mutated_code = reconstruct_mutant_code(original_src, mu)
            if sandbox_file.is_symlink():
                sandbox_file.unlink()
            sandbox_file.write_text(mutated_code, encoding="utf-8")

            # Final result file (outside sandbox)
            safe_key = sanitize_mutant_key(mutant_key)
            mutant_results_path = paths["save_path"] / f"mutant_{safe_key}.json"

            # Remove any leftover inner results
            if inner_results_path.exists():
                inner_results_path.unlink()

            # Build command pointing to inner_results_path
            cmd = build_pytest_command(
                sandbox_root,
                paths["test_paths"],
                paths["prog_res"],
                inner_results_path,
                args,
            )

            print("Command:", " ".join(cmd))
            result = subprocess.run(
                cmd, cwd=sandbox_root, stdout=sys.stdout, stderr=sys.stderr
            )

            # Prepare wrapped result:
            # start with all the original mutant metadata, then add runner info.
            wrapped = dict(mu)  # shallow copy of the mutant descriptor

            # Attach execution metadata
            wrapped["pytest_exit_code"] = result.returncode
            wrapped["test_outcomes"] = None

            # If pytest wrote inner results, load and attach them
            if inner_results_path.exists():
                try:
                    with inner_results_path.open("r", encoding="utf-8") as f:
                        raw_results = json.load(f)
                    wrapped["test_outcomes"] = raw_results
                except Exception as e:
                    print(
                        f"Warning: failed to read inner results for {mutant_key}: {e}",
                        file=sys.stderr,
                    )

            # Write final wrapped file
            with mutant_results_path.open("w", encoding="utf-8") as f:
                json.dump(wrapped, f, indent=2, sort_keys=True)

            if result.returncode != 0:
                print(
                    f"Pytest exited with {result.returncode} for mutant {mutant_key}",
                    file=sys.stderr,
                )

            # Restore original file for next mutant
            if sandbox_file.is_symlink():
                sandbox_file.unlink()
            sandbox_file.write_text(original_src, encoding="utf-8")

            # Clean up inner results
            if inner_results_path.exists():
                inner_results_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
