#!/usr/bin/env python3
"""
Map (MASZZ) bug-inducing commit (BIC) candidates onto the repository’s first-parent
(mainline) history.

- mode=exact: candidate is on first-parent chain -> boundary=candidate
- mode=date : candidate is off-mainline -> boundary chosen by timestamp projection
- dist      : inclusive first-parent distance from boundary..HEAD

Reporting:
- default: verbose report (per-candidate + chosen best)
- --compact: one CSV-like line per bug:
    project,bug_id,chosen_candidate,boundary,mode
  (empty fields if no valid candidate)
"""

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_BIC_TMPL = "/home/rooti/shared/LatentMu/data/maszz/bic_{project}_maszz.json"
DEFAULT_REPO_TMPL = "/home/rooti/shared/LatentMu/chk/{project}-{bid}b/{project}"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--project", required=True, help="Project name, e.g. youtube-dl, thefuck")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bids", nargs="+", help="Explicit bug IDs, e.g. --bids 1 2 3 10")
    g.add_argument("--bid-range", nargs=2, type=int, metavar=("MIN", "MAX"),
                   help="Inclusive range, e.g. --bid-range 1 43")

    p.add_argument(
        "--bic-file",
        default=None,
        help=f"Path to MASZZ BIC JSON. Default: {DEFAULT_BIC_TMPL}",
    )
    p.add_argument(
        "--repo-root-tmpl",
        default=DEFAULT_REPO_TMPL,
        help=f"Checkout path template (uses {{project}} and {{bid}}). Default: {DEFAULT_REPO_TMPL}",
    )

    p.add_argument("--compact", action="store_true", help="Print one CSV line per bug.")
    return p.parse_args()


def git(repo, *args) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip())
    return res.stdout.strip()


def get_head(repo) -> str:
    return git(repo, "rev-parse", "HEAD")


def first_parent_chain_with_times(repo, head_sha: str):
    out = git(repo, "rev-list", "--first-parent", "--format=%H %ct", head_sha)
    chain = []
    times = {}
    for line in out.splitlines():
        if not line or line.startswith("commit "):
            continue
        sha, ct = line.split()
        chain.append(sha)
        times[sha] = int(ct)
    return chain, times


def bulk_commit_times(repo, shas):
    shas = [s for s in shas if s]
    if not shas:
        return {}

    p = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert p.stdin and p.stdout and p.stderr
    p.stdin.write("\n".join(shas) + "\n")
    p.stdin.flush()
    p.stdin.close()

    times = {}
    while True:
        header = p.stdout.readline()
        if not header:
            break

        header = header.strip()
        if not header:
            continue

        parts = header.split()
        if len(parts) == 2 and parts[1] == "missing":
            continue
        if len(parts) < 3:
            continue

        sha, obj_type, size_s = parts[0], parts[1], parts[2]
        size = int(size_s)

        data = p.stdout.read(size)
        p.stdout.read(1)  # trailing newline

        if obj_type != "commit":
            continue

        for line in data.splitlines():
            if line.startswith("committer "):
                toks = line.split()
                if len(toks) >= 3:
                    try:
                        times[sha] = int(toks[-2])
                    except ValueError:
                        pass
                break

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(p.stderr.read().strip())

    return times


def boundary_like_list_commits_since(chain, chain_times, chain_set, start_sha, start_time):
    if start_sha in chain_set:
        return start_sha, "exact"

    prev = None
    for c in chain:  # newest -> oldest
        t = chain_times[c]
        if t <= start_time:
            if t == start_time:
                return c, "date"
            return (prev or c), "date"
        prev = c

    return chain[-1], "date"


def count_commits_inclusive_first_parent(repo, start_sha, head_sha) -> int:
    return int(git(repo, "rev-list", "--count", "--first-parent", f"{start_sha}^..{head_sha}"))


def expand_bids(args) -> list[str]:
    if args.bids is not None:
        return [str(x) for x in args.bids]
    lo, hi = args.bid_range
    return [str(x) for x in range(lo, hi + 1)]


def main():
    args = parse_args()
    project = args.project
    bug_ids = expand_bids(args)

    bic_path = Path(args.bic_file) if args.bic_file else Path(DEFAULT_BIC_TMPL.format(project=project))
    bic_data = json.loads(bic_path.read_text(encoding="utf-8"))

    if args.compact:
        print("project,bug_id,chosen_candidate,boundary,mode")
    else:
        print(f"Project: {project}")
        print(f"BIC JSON: {bic_path}")
        print(f"Repo tmpl: {args.repo_root_tmpl}")

    for bid in bug_ids:
        repo_root = Path(args.repo_root_tmpl.format(project=project, bid=bid)).resolve()

        rows = [x for x in bic_data if x.get("bug_num") == bid and x.get("repo_name") == project]
        if not rows:
            if args.compact:
                print(f"{project},{bid},,,")
            else:
                print(f"\nBug ID: {bid}")
                print("No MASZZ row found for this bug.")
                print("-" * 50)
            continue

        for i, row in enumerate(rows):
            try:
                head_sha = get_head(repo_root)
            except Exception as e:
                if args.compact:
                    print(f"{project},{bid},,,")
                else:
                    print(f"\nBug ID: {bid}")
                    print(f"Repo missing/unreadable: {repo_root} ({e})")
                    print("-" * 50)
                continue

            chain, chain_times = first_parent_chain_with_times(repo_root, head_sha)
            chain_set = set(chain)
            chain_index = {sha: j for j, sha in enumerate(chain)}  # 0=head newest

            candidates = row.get("inducing_commit_hash", []) or []
            cand_times = bulk_commit_times(repo_root, candidates)

            if not args.compact:
                print(f"\nBug ID: {bid}")
                print("HEAD:", head_sha)
                print(f"\nRow {i} candidates:")

            best = None  # (idx_on_chain, cand_sha, boundary_sha, mode, dist)

            for cand in candidates:
                if cand not in cand_times:
                    if not args.compact:
                        print(f"  {cand} -> FAIL (unresolvable or not a commit)")
                    continue
                try:
                    boundary, mode = boundary_like_list_commits_since(
                        chain, chain_times, chain_set, cand, cand_times[cand]
                    )
                    dist = count_commits_inclusive_first_parent(repo_root, boundary, head_sha)

                    if not args.compact:
                        print(f"  {cand} -> {dist} (boundary={boundary}, mode={mode})")

                    idx = chain_index[boundary]
                    if best is None or idx < best[0]:
                        best = (idx, cand, boundary, mode, dist)
                except Exception as e:
                    if not args.compact:
                        print(f"  {cand} -> FAIL ({e})")

            if args.compact:
                if best is None:
                    print(f"{project},{bid},,,")
                else:
                    _, cand, boundary, mode, _dist = best
                    print(f"{project},{bid},{cand},{boundary},{mode}")
            else:
                if best is None:
                    print(f"Row {i}: no valid candidates")
                else:
                    _, cand, boundary, mode, dist = best
                    print(
                        f"Row {i}: NEWEST effective mainline = {cand} "
                        f"(boundary={boundary}, mode={mode}, dist={dist})"
                    )
                print("-" * 50)


if __name__ == "__main__":
    main()
