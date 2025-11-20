#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pytest for each mutant in a sandboxed copy of a repo."
    )

    # Required named args
    parser.add_argument("--base-repo", required=True,
                        help="Root directory of the project")
    parser.add_argument("--src-path", required=True,
                        help=".py file inside the project")
    parser.add_argument("--tests", nargs="+", required=True,
                        help="Test file(s) or directories")
    parser.add_argument("--prog-res", required=True,
                        help="JSON file with program execution results")
    parser.add_argument("--mutants", required=True,
                        help="JSON file with mutants information")
    parser.add_argument("--save-path", required=True,
                        help="Where to store mutant result files")

    parser.add_argument("--stop-on-kill", action="store_true", default=False)
    parser.add_argument("--timeout-min", type=float, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--disable-incompetent", action="store_true", default=False)

    parser.add_argument(
        "--njobs",type=int,default=1,
            help="Total number of jobs for partitioning mutants (default: 1)",
    )
    parser.add_argument(
        "--jid",type=int,default=0,
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
        help="Sandbox creation mode: 'copy' (default) or 'symlink'."
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

    # src_path must be inside base_repo
    src_path = Path(args.src_path).expanduser()
    if not src_path.is_absolute():
        src_path = (base_repo / src_path).resolve()
    else:
        src_path = src_path.resolve()

    try:
        relative_src = src_path.relative_to(base_repo)
    except ValueError:
        raise SystemExit(f"src_path must be inside base_repo.\n  base_repo: {base_repo}\n  src_path: {src_path}")

    # Tests: resolve, if under base_repo then remember relative path
    test_paths = []
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
        "src_path": src_path,
        "relative_src": relative_src,
        "test_paths": test_paths,
        "prog_res": prog_res,
        "mutants": mutants,
        "save_path": save_path,
    }


def load_src_code(src_path: Path) -> str:
    return src_path.read_text(encoding="utf-8")


def load_mutants(mutants_path: Path):
    with mutants_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either a dict or a list
    if isinstance(data, dict):
        # iterate as (key, value)
        iterable = []
        for k, v in data.items():
            # store id key if not present
            if isinstance(v, dict) and "id" not in v:
                v = dict(v)
                v["id"] = k
            iterable.append(v)
        return iterable
    elif isinstance(data, list):
        # ensure each mutant has an id
        result = []
        for i, mu in enumerate(data):
            if isinstance(mu, dict) and "id" not in mu:
                mu = dict(mu)
                mu["id"] = mu.get("mutant_id", i)
            result.append(mu)
        return result
    else:
        raise SystemExit("mutants JSON must be a list or dict")


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

    # Custom options
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

    original_src = load_src_code(paths["src_path"])
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

    # Nothing left to run
    if not job_mutants:
        print("This job has no mutants to process. Exiting.")
        return

    # Create sandbox as a full copy of the repo
    with tempfile.TemporaryDirectory(prefix="mutant-sandbox-") as tmpdir:
        sandbox_root = Path(tmpdir)

        # creating the sandbox
        if args.sandbox_mode == "copy":
            shutil.copytree(paths["base_repo"], sandbox_root, dirs_exist_ok=True)
        elif args.sandbox_mode == "symlink":
            shutil.copytree(
                paths["base_repo"],
                sandbox_root,
                copy_function=os.symlink,
                dirs_exist_ok=True,
            )

        sandbox_src_path = sandbox_root / paths["relative_src"]

        if not sandbox_src_path.exists():
            raise SystemExit(
                f"Sandbox source path does not exist after copy: {sandbox_src_path}"
            )

        #################################################################
        # Main mutation/testing loop
        #################################################################

        for local_idx, mu in enumerate(job_mutants):
            global_idx = start_idx + local_idx
            mu_id = mu.get("id", global_idx)
            print(
                f"=== [jid={args.jid}] Running mutant {mu_id} "
                f"(global {global_idx + 1}/{total_mutants}, "
                f"job {local_idx + 1}/{len(job_mutants)}) ==="
            )

            # Reconstruct mutant code
            mutant_code = reconstruct_mutant_code(original_src, mu)

            # Break symlink safely, then write mutated file only in sandbox
            if sandbox_src_path.is_symlink():
                sandbox_src_path.unlink()
            # Write mutant into sandbox
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

            # NOTE:
            # We do NOT restore here.
            # This is fine as long as all mutants target the same file and we always
            # overwrite it with the next mutant.
            # If in the future mutants can target different files, we'll want to:
            #   1) track the original content per file, and
            #   2) restore only the file that was mutated or needed.

            # Break symlink safely, then write mutated file only in sandbox
            # if sandbox_src_path.is_symlink():
            #     sandbox_src_path.unlink()
            # sandbox_src_path.write_text(original_src, encoding="utf-8")

    print("Done.")



if __name__ == "__main__":
    main()
