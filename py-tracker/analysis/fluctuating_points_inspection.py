from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================
# STATUS CONSTANTS
# =========================
STATUS_INCOMPETENT = "incompetent"
STATUS_KILLED = "killed"
STATUS_SURVIVED = "survived"
STATUS_UNKNOWN = "unknown"
ALL_STATUSES = (STATUS_INCOMPETENT, STATUS_KILLED, STATUS_SURVIVED, STATUS_UNKNOWN)


# =========================
# VOLATILE MESSAGE NORMALIZATION
# =========================
HEX_TOKEN = "<HEX>"
RE_HEX_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b")
RE_HEX_ESCAPE = re.compile(r"\\x[0-9a-fA-F]{2}")
RE_SANDBOX_ID = re.compile(r"sandbox-[A-Za-z0-9_-]+")


def normalize_msg(msg: Any) -> str:
    """
    Canonicalize noisy parts of messages:
      - 0x... addresses
      - \\xNN escape sequences
      - sandbox-<random>
    """
    if msg is None:
        return ""
    s = str(msg)
    s = RE_HEX_ADDR.sub(HEX_TOKEN, s)
    s = RE_HEX_ESCAPE.sub(HEX_TOKEN, s)
    s = RE_SANDBOX_ID.sub("sandbox-<ID>", s)
    return s


# =========================
# I/O + BASIC HELPERS
# =========================
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_status(s: Any) -> str:
    if s is None:
        return STATUS_UNKNOWN
    s = str(s).strip().lower()
    if not s:
        return STATUS_UNKNOWN
    return s


def safe_div(num: int, den: int) -> float:
    return (num / den) if den != 0 else 0.0


def load_commit_file(results_dir: Path, commit: str) -> Optional[Dict[str, Any]]:
    p = results_dir / f"results_{commit}.json"
    if not p.exists():
        return None
    return load_json(p)


# =========================
# PROGRAM FAILING TESTS
# =========================
def failing_tests_from_program(data: Dict[str, Any]) -> Set[str]:
    tests = ((data.get("program") or {}).get("tests")) or {}
    return {
        tid
        for tid, td in tests.items()
        if normalize_status(td.get("outcome")) == "failed"
    }


# =========================
# STATUS AGGREGATION LOGIC
# =========================
def status_counts_and_rates_from_tests(mtests: Dict[str, Any]) -> Tuple[Counter, Dict[str, float], int]:
    """
    Returns (counts, rates, total_tests) over per-test statuses for this mutant in this commit.
    """
    cnt = Counter()
    if not mtests:
        for s in ALL_STATUSES:
            cnt[s] = 0
        return cnt, {s: 0.0 for s in ALL_STATUSES}, 0

    for t in mtests.values():
        st = normalize_status((t or {}).get("status"))
        if st not in ALL_STATUSES:
            st = STATUS_UNKNOWN
        cnt[st] += 1

    total = sum(cnt.values())
    for s in ALL_STATUSES:
        cnt.setdefault(s, 0)

    rates = {s: (cnt[s] / total if total > 0 else 0.0) for s in ALL_STATUSES}
    return cnt, rates, total


def overall_mutant_status_priority(mtests: Dict[str, Any]) -> str:
    """
    Priority rules:
      - if any incompetent -> incompetent
      - else if any killed -> killed
      - else if any survived -> survived (even if some unknown)
      - else unknown
    """
    cnt, _, total = status_counts_and_rates_from_tests(mtests)
    if total == 0:
        return STATUS_UNKNOWN
    if cnt[STATUS_INCOMPETENT] > 0:
        return STATUS_INCOMPETENT
    if cnt[STATUS_KILLED] > 0:
        return STATUS_KILLED
    if cnt[STATUS_SURVIVED] > 0:
        return STATUS_SURVIVED
    return STATUS_UNKNOWN


def killed_by_failing_tests(mutant_md: Dict[str, Any], failing_tests: Set[str]) -> bool:
    """
    True iff this mutant has per-test status 'killed' for at least one failing test id.
    """
    mtests = (mutant_md.get("tests") or {})
    for tid in failing_tests:
        td = mtests.get(tid)
        if not td:
            continue
        if normalize_status(td.get("status")) == STATUS_KILLED:
            return True
    return False


# =========================
# PARALLEL COMMIT WORKER
# =========================
def commit_worker(commit: str, path: str) -> Tuple[str, Dict[str, Dict[str, Any]], bool, Dict[str, int]]:
    """
    For one commit file, return:
      (commit, per_mutant_summary, missing, commit_overall_counts)

    per_mutant_summary[mid] = {
        "overall_status": ...,
        "counts": {...},
        "rates": {...},
        "total_tests": int
    }

    commit_overall_counts is a Counter over per-mutant overall_status in that commit file.
    """
    p = Path(path)
    if not p.exists():
        return commit, {}, True, {s: 0 for s in ALL_STATUSES}

    data = load_json(p)
    mutants = data.get("mutants", {}) or {}

    out: Dict[str, Dict[str, Any]] = {}
    commit_counts = Counter()

    for mid, md in mutants.items():
        mtests = (md.get("tests") or {})
        cnt, rates, total = status_counts_and_rates_from_tests(mtests)
        overall = overall_mutant_status_priority(mtests)

        out[mid] = {
            "overall_status": overall,
            "counts": {s: int(cnt[s]) for s in ALL_STATUSES},
            "rates": {s: float(rates[s]) for s in ALL_STATUSES},
            "total_tests": int(total),
        }
        commit_counts[overall] += 1

    for s in ALL_STATUSES:
        commit_counts.setdefault(s, 0)

    return commit, out, False, {s: int(commit_counts[s]) for s in ALL_STATUSES}


# =========================
# FLIPFLOP INSPECTION (use precomputed flipflops)
# =========================
def get_killing_test_ids_by_status(mutant_tests: Dict[str, Any]) -> List[str]:
    """
    Uses stored per-test label: mt["status"] == "killed"
    """
    killers = []
    for tid, mt in (mutant_tests or {}).items():
        if normalize_status((mt or {}).get("status")) == STATUS_KILLED:
            killers.append(tid)
    return killers


def pretty_test_row(tid: str, td: Dict[str, Any]) -> str:
    out = normalize_status((td or {}).get("outcome"))
    st = normalize_status((td or {}).get("status"))
    dur = (td or {}).get("duration", None)
    msg = normalize_msg((td or {}).get("raw_short_msg"))
    if len(msg) > 260:
        msg = msg[:260] + "…"
    return f"- {tid}\n  outcome={out}  status={st}  duration={dur}\n  msg={msg}"


def inspect_middle_killed_commit_for_flipflop(
    *,
    mutant_id: str,
    commit_before: str,
    commit_killed: str,
    commit_after: str,
    results_dir: Path,
) -> str:
    """
    Inspects the middle killed commit: shows killer test ids and program vs mutant test data.
    Returns one of: "continue", "skip_mutant", "next_flipflop", "quit"
    """
    data = load_commit_file(results_dir, commit_killed)
    if not data:
        print(f"(missing file for middle commit {commit_killed})", flush=True)
        return "next_flipflop"

    program_tests = ((data.get("program") or {}).get("tests")) or {}
    md = ((data.get("mutants") or {}).get(mutant_id)) or {}
    mutant_tests = (md.get("tests") or {}) or {}

    killer_tids = get_killing_test_ids_by_status(mutant_tests)

    print("\n" + "=" * 110, flush=True)
    print(f"FLIPFLOP: {mutant_id}", flush=True)
    print(f"  {commit_before} (survived)  ->  {commit_killed} (killed)  ->  {commit_after} (survived)", flush=True)
    print(f"  middle commit killing tests: {len(killer_tids)}", flush=True)

    if not killer_tids:
        cmd0 = input("No per-test 'killed' entries. Enter=next flipflop | s=skip mutant | q=quit > ").strip().lower()
        if cmd0 == "q":
            return "quit"
        if cmd0 == "s":
            return "skip_mutant"
        return "next_flipflop"

    for tid in killer_tids:
        pt = program_tests.get(tid) or {}
        mt = mutant_tests.get(tid) or {}

        print("\nPROGRAM", flush=True)
        print(pretty_test_row(tid, pt), flush=True)
        print("\nMUTANT", flush=True)
        print(pretty_test_row(tid, mt), flush=True)

        cmd = input("\nEnter=next test | n=next flipflop | s=skip mutant | q=quit > ").strip().lower()
        if cmd == "q":
            return "quit"
        if cmd == "s":
            return "skip_mutant"
        if cmd == "n":
            return "next_flipflop"

    return "continue"


def interactive_inspect_flipflops_from_list(
    *,
    flipflops: List[Dict[str, Any]],
    results_dir: Path,
    limit: int = 200,
) -> None:
    """
    Uses already-found flipflops from the report stage (fast).
    Each dict must have: {"mutant_id": ..., "commits": [c1,c2,c3]}
    """
    flipflops = flipflops[:limit]
    print(f"\nFound flipflops (survived->killed->survived): {len(flipflops)} (showing up to {limit})", flush=True)

    skip_mutants: Set[str] = set()

    for i, ff in enumerate(flipflops, 1):
        mid = ff["mutant_id"]
        if mid in skip_mutants:
            continue

        c1, c2, c3 = ff["commits"]

        print("\n" + "-" * 110, flush=True)
        print(f"[{i}/{len(flipflops)}] {mid}: {c1} -> {c2} -> {c3}", flush=True)

        action = inspect_middle_killed_commit_for_flipflop(
            mutant_id=mid,
            commit_before=c1,
            commit_killed=c2,
            commit_after=c3,
            results_dir=results_dir,
        )

        if action == "quit":
            return
        if action == "skip_mutant":
            skip_mutants.add(mid)
            continue
        if action == "next_flipflop":
            continue

        cmd2 = input("\nEnter=next flipflop | q=quit > ").strip().lower()
        if cmd2 == "q":
            return


# =========================
# MAIN (stats + report + interactive flipflop inspector)
# =========================
def main() -> None:
    # ---- USER PATHS ----
    base = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-7b")
    commits_file = base / "commits_to_run.txt"
    results_dir = base / "with_lbl"

    # ---- SETTINGS ----
    N_WORKERS = 4
    OUT_JSON = base / "mutant_timeline_report.json"

    UNRESOLVED_LIMIT = 30
    ALLKILLED_LIMIT = 30
    FLIPFLOP_LIMIT = 200  # also used for interactive list
    INTERACTIVE_FLIPFLOP_LIMIT = 200

    commits = [ln.strip() for ln in commits_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not commits:
        raise SystemExit(f"No commits found in {commits_file}")

    head = commits[0]
    head_path = results_dir / f"results_{head}.json"
    if not head_path.exists():
        raise SystemExit(f"Missing HEAD results file: {head_path}")

    # ----- Load HEAD fully -----
    head_data = load_json(head_path)
    failing_tests = failing_tests_from_program(head_data)
    head_mutants = head_data.get("mutants", {}) or {}

    # Per-mutant HEAD summaries (counts/rates/overall) using same logic
    _head_commit, head_summaries, _missing, _head_commit_counts = commit_worker(head, str(head_path))

    # HEAD special flags
    head_flags: Dict[str, Dict[str, Any]] = {}
    for mid, md in head_mutants.items():
        overall = head_summaries.get(mid, {}).get("overall_status", STATUS_UNKNOWN)
        is_killed = (overall == STATUS_KILLED)
        killed_by_fail = killed_by_failing_tests(md, failing_tests)
        head_flags[mid] = {
            "head_overall_status": overall,
            "head_is_killed": bool(is_killed),
            "head_killed_by_failing_tests": bool(killed_by_fail),
        }

    # HEAD overall status counts
    head_overall_counts = Counter(s["overall_status"] for s in head_summaries.values())
    head_incomp = int(head_overall_counts.get(STATUS_INCOMPETENT, 0))
    head_killed = int(head_overall_counts.get(STATUS_KILLED, 0))
    head_survived = int(head_overall_counts.get(STATUS_SURVIVED, 0))
    head_unknown = int(head_overall_counts.get(STATUS_UNKNOWN, 0))

    head_mutation_score = safe_div(head_killed, head_killed + head_survived)

    killed_at_head: Set[str] = {mid for mid, f in head_flags.items() if f["head_is_killed"]}
    killed_by_failing_at_head: Set[str] = {mid for mid in killed_at_head if head_flags[mid]["head_killed_by_failing_tests"]}

    # ----- Process earlier commits in parallel -----
    jobs = [(c, str(results_dir / f"results_{c}.json")) for c in commits[1:]]
    missing_files = 0
    timeline: Dict[str, List[Dict[str, Any]]] = {}

    unresolved_examples: List[Tuple[str, str, str, int]] = []
    all_killed_commits: List[Tuple[str, Dict[str, int]]] = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(commit_worker, c, p) for (c, p) in jobs]
        for fut in as_completed(futs):
            commit, per_mutant, missing, commit_counts = fut.result()
            if missing:
                missing_files += 1
                continue

            total_in_file = sum(commit_counts.values())
            if total_in_file > 0 and commit_counts[STATUS_KILLED] == total_in_file:
                if len(all_killed_commits) < ALLKILLED_LIMIT:
                    all_killed_commits.append((commit, commit_counts))

            for mid, summ in per_mutant.items():
                if (summ["overall_status"] == STATUS_UNKNOWN) or (summ["total_tests"] == 0):
                    if len(unresolved_examples) < UNRESOLVED_LIMIT:
                        unresolved_examples.append((commit, mid, summ["overall_status"], summ["total_tests"]))

                timeline.setdefault(mid, []).append({"commit": commit, **summ})

    # ----- Insert HEAD entries + latent flags -----
    mutants_report: Dict[str, Any] = {}
    all_mutant_ids = set(head_summaries.keys()) | set(timeline.keys())

    for mid in sorted(all_mutant_ids):
        head_entry = head_summaries.get(mid, {
            "overall_status": STATUS_UNKNOWN,
            "counts": {s: 0 for s in ALL_STATUSES},
            "rates": {s: 0.0 for s in ALL_STATUSES},
            "total_tests": 0,
        })

        hist = timeline.get(mid, [])
        hist_by_commit = {e["commit"]: e for e in hist}
        ordered_hist = [hist_by_commit[c] for c in commits[1:] if c in hist_by_commit]

        survived_commits = [e["commit"] for e in ordered_hist if e["overall_status"] == STATUS_SURVIVED]
        survived_earlier = len(survived_commits) > 0

        is_killed_head = head_flags.get(mid, {}).get("head_is_killed", False)
        killed_by_fail_head = head_flags.get(mid, {}).get("head_killed_by_failing_tests", False)

        latent_overall_killed = bool(is_killed_head and survived_earlier)
        latent_killed_by_failing = bool(is_killed_head and killed_by_fail_head and survived_earlier)

        mutants_report[mid] = {
            "head": {
                "commit": head,
                **head_entry,
                **head_flags.get(mid, {
                    "head_overall_status": head_entry["overall_status"],
                    "head_is_killed": False,
                    "head_killed_by_failing_tests": False,
                }),
            },
            "history": ordered_hist,
            "survived_commits": survived_commits,
            "latent": {
                "latent_overall_killed": latent_overall_killed,
                "latent_killed_by_failing": latent_killed_by_failing,
            },
        }

    latent1 = [mid for mid, r in mutants_report.items() if r["latent"]["latent_overall_killed"]]
    latent2 = [mid for mid, r in mutants_report.items() if r["latent"]["latent_killed_by_failing"]]

    # ----- Flipflop examples (survived -> killed -> survived) from HISTORY only -----
    # NOTE: This matches your earlier behavior (uses ordered_hist which excludes HEAD).
    flipflop_examples: List[Dict[str, Any]] = []
    for mid in sorted(all_mutant_ids):
        hist = mutants_report[mid]["history"]
        seq = [{"commit": e["commit"], "status": e["overall_status"]} for e in hist]

        for i in range(1, len(seq) - 1):
            a, b, c = seq[i - 1], seq[i], seq[i + 1]
            if a["status"] == STATUS_SURVIVED and b["status"] == STATUS_KILLED and c["status"] == STATUS_SURVIVED:
                flipflop_examples.append({
                    "mutant_id": mid,
                    "pattern": "survived -> killed -> survived",
                    "commits": [a["commit"], b["commit"], c["commit"]],
                })
                if len(flipflop_examples) >= FLIPFLOP_LIMIT:
                    break
        if len(flipflop_examples) >= FLIPFLOP_LIMIT:
            break

    report = {
        "meta": {
            "base": str(base),
            "results_dir": str(results_dir),
            "commits_file": str(commits_file),
            "num_commits": len(commits),
            "head_commit": head,
            "missing_commit_files": missing_files,
            "workers": N_WORKERS,
        },
        "head_summary": {
            "total_mutants": sum(1 for _ in head_summaries),
            "overall_status_counts": {s: int(head_overall_counts.get(s, 0)) for s in ALL_STATUSES},
            "failing_tests_count": len(failing_tests),
            "killed_at_head_count": len(killed_at_head),
            "killed_by_failing_at_head_count": len(killed_by_failing_at_head),
            "mutation_score": head_mutation_score,
            "mutation_score_definition": "killed / (killed + survived), excluding incompetent and unknown",
        },
        "latent_summary": {
            "latent_overall_killed_count": len(latent1),
            "latent_killed_by_failing_count": len(latent2),
        },
        "anomalies": {
            "unresolved_examples": [
                {"commit": c, "mutant_id": m, "overall_status": st, "total_tests": tt}
                for (c, m, st, tt) in unresolved_examples
            ],
            "all_killed_commits_examples": [
                {"commit": c, "overall_status_counts": counts}
                for (c, counts) in all_killed_commits
            ],
            "flipflop_examples": flipflop_examples,
            "definitions": {
                "unresolved": "overall_status == 'unknown' OR total_tests == 0",
                "all_killed_commit": "in that commit file, all mutants' overall_status == 'killed'",
                "flipflop": "three consecutive commits where status is survived->killed->survived",
            },
        },
        "mutants": mutants_report,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    print(f"Wrote report: {OUT_JSON}")
    print(f"HEAD: {head}")
    print(
        f"HEAD status counts: killed={head_killed}, survived={head_survived}, "
        f"incompetent={head_incomp}, unknown={head_unknown}"
    )
    print(f"HEAD mutation score: {head_mutation_score:.4f}  (killed={head_killed}, survived={head_survived})")
    print(f"Latent (overall killed at head): {len(latent1)} [killed at head: {len(killed_at_head)}]")
    print(
        f"Latent (killed-by-failing at head): {len(latent2)} "
        f"[killed-by-failing at head: {len(killed_by_failing_at_head)}, failing tests at head: {len(failing_tests)}]"
    )
    print(f"Missing commit files: {missing_files}")

    print("\n=== Examples: unresolved mutants (commit, mutant_id, status, total_tests) ===")
    for commit, mid, st, tt in unresolved_examples[:10]:
        print(f"{commit}  {mid}  status={st}  total_tests={tt}")

    print("\n=== Examples: commits where all mutants are overall killed (commit, counts) ===")
    for commit, counts in all_killed_commits[:10]:
        total = sum(counts.values())
        print(f"{commit}  total={total}  counts={counts}")

    print("\n=== Examples: immediate flip-flops survived->killed->survived ===")
    for ex in flipflop_examples[:10]:
        c1, c2, c3 = ex["commits"]
        print(f'{ex["mutant_id"]}: {c1} (survived) -> {c2} (killed) -> {c3} (survived)')

    # ----- INTERACTIVE: reuse already-computed flipflops and inspect the middle killed commit -----
    print("\n\n=== INTERACTIVE FLIPFLOP INSPECTION ===", flush=True)
    print("Controls: Enter=next test, n=next flipflop, s=skip mutant (all its flipflops), q=quit", flush=True)
    interactive_inspect_flipflops_from_list(
        flipflops=flipflop_examples,
        results_dir=results_dir,
        limit=INTERACTIVE_FLIPFLOP_LIMIT,
    )


if __name__ == "__main__":
    main()
