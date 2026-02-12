from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


STATUS_INCOMPETENT = "incompetent"
STATUS_ANOMALOUS = "anomalous"   
STATUS_KILLED = "killed"
STATUS_SURVIVED = "survived"
STATUS_UNKNOWN = "unknown"

ALL_STATUSES = (
    STATUS_INCOMPETENT,
    STATUS_ANOMALOUS,           
    STATUS_KILLED,
    STATUS_SURVIVED,
    STATUS_UNKNOWN,
)


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


def failing_tests_from_program(data: Dict[str, Any]) -> Set[str]:
    tests = ((data.get("program") or {}).get("tests")) or {}
    return {
        tid
        for tid, td in tests.items()
        if normalize_status(td.get("outcome")) == "failed"
    }


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
    Priority rules (recommended):
      - if any incompetent -> incompetent
      - else if any killed -> killed
      - else if any anomalous -> anomalous
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
    if cnt[STATUS_ANOMALOUS] > 0:
        return STATUS_ANOMALOUS
    if cnt[STATUS_SURVIVED] > 0:
        return STATUS_SURVIVED
    return STATUS_UNKNOWN


def killed_by_failing_tests(mutant_md: Dict[str, Any], failing_tests: Set[str]) -> bool:
    """
    True iff this mutant has per-test status 'killed' for at least one failing test id.
    (NOTE: anomalous does NOT count as killed here.)
    """
    mtests = (mutant_md.get("tests") or {})
    for tid in failing_tests:
        td = mtests.get(tid)
        if not td:
            continue
        if normalize_status(td.get("status")) == STATUS_KILLED:
            return True
    return False


def commit_worker(commit: str, path: str) -> Tuple[str, Dict[str, Dict[str, Any]], bool]:
    """
    For one commit file, return:
      (commit, per_mutant_summary, missing)

    per_mutant_summary[mid] = {
        "overall_status": ...,
        "counts": {...},
        "rates": {...},
        "total_tests": int
    }
    """
    p = Path(path)
    if not p.exists():
        return commit, {}, True

    data = load_json(p)
    mutants = data.get("mutants", {}) or {}

    out: Dict[str, Dict[str, Any]] = {}
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

    return commit, out, False


def safe_div(num: int, den: int) -> float:
    return (num / den) if den != 0 else 0.0


def main() -> None:
    base = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-4b")
    commits_file = base / "commits_to_run.txt"
    results_dir = base / "with_lbl"

    N_WORKERS = 4
    OUT_JSON = base / "mutant_timeline_report.json"

    commits = [ln.strip() for ln in commits_file.read_text(encoding="utf-8").splitlines() if ln.strip()]  # [:50]
    if not commits:
        raise SystemExit(f"No commits found in {commits_file}")

    head = commits[0]
    head_path = results_dir / f"results_{head}.json"
    if not head_path.exists():
        raise SystemExit(f"Missing HEAD results file: {head_path}")

    # ----- Load HEAD fully (for failing tests + special flags) -----
    head_data = load_json(head_path)
    failing_tests = failing_tests_from_program(head_data)
    head_mutants = head_data.get("mutants", {}) or {}

    # Per-mutant HEAD summaries (counts/rates/overall) using same logic
    _head_commit, head_summaries, _missing = commit_worker(head, str(head_path))

    # HEAD special flags
    head_flags: Dict[str, Dict[str, Any]] = {}
    for mid, md in head_mutants.items():
        overall = head_summaries.get(mid, {}).get("overall_status", STATUS_UNKNOWN)
        is_killed = (overall == STATUS_KILLED)
        is_anomalous = (overall == STATUS_ANOMALOUS) 
        killed_by_fail = killed_by_failing_tests(md, failing_tests)

        head_flags[mid] = {
            "head_overall_status": overall,
            "head_is_killed": bool(is_killed),
            "head_is_anomalous": bool(is_anomalous), 
            "head_killed_by_failing_tests": bool(killed_by_fail),
        }

    # HEAD overall status counts (per mutant, after priority rule)
    head_overall_counts = Counter(s["overall_status"] for s in head_summaries.values())
    head_incomp = int(head_overall_counts.get(STATUS_INCOMPETENT, 0))
    head_anom = int(head_overall_counts.get(STATUS_ANOMALOUS, 0)) 
    head_killed = int(head_overall_counts.get(STATUS_KILLED, 0))
    head_survived = int(head_overall_counts.get(STATUS_SURVIVED, 0))
    head_unknown = int(head_overall_counts.get(STATUS_UNKNOWN, 0))

    # HEAD mutation score: killed / (killed + survived), excluding incompetent/anomalous/unknown
    head_mutation_score = safe_div(head_killed, head_killed + head_survived)

    # Candidate sets for latent definitions:
    # We require overall killed at head (so NOT incompetent/anomalous at head).
    killed_at_head: Set[str] = {mid for mid, f in head_flags.items() if f["head_is_killed"]}
    killed_by_failing_at_head: Set[str] = {
        mid for mid in killed_at_head if head_flags[mid]["head_killed_by_failing_tests"]
    }

    # ----- Process earlier commits in parallel: produce per-mutant per-commit summaries -----
    jobs = [(c, str(results_dir / f"results_{c}.json")) for c in commits[1:]]
    missing_files = 0

    timeline: Dict[str, List[Dict[str, Any]]] = {}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(commit_worker, c, p) for (c, p) in jobs]
        for fut in as_completed(futs):
            commit, per_mutant, missing = fut.result()
            if missing:
                missing_files += 1
                continue

            for mid, summ in per_mutant.items():
                timeline.setdefault(mid, []).append({
                    "commit": commit,
                    **summ,
                })

    # ----- Insert HEAD entries and compute latent flags -----
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
                    "head_is_anomalous": False,
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
            "mutation_score_definition": "killed / (killed + survived), excluding incompetent, anomalous and unknown",
        },
        "latent_summary": {
            "latent_overall_killed_count": len(latent1),
            "latent_killed_by_failing_count": len(latent2),
        },
        "mutants": mutants_report,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    print(f"Wrote report: {OUT_JSON}")
    print(f"HEAD: {head}")

    print(
        f"HEAD status counts: "
        f"killed={head_killed}, survived={head_survived}, "
        f"incompetent={head_incomp}, anomalous={head_anom}, unknown={head_unknown}"
    )

    print(f"HEAD mutation score: {head_mutation_score:.4f}  (killed={head_killed}, survived={head_survived})")

    print(
        f"Latent (overall killed at head): {len(latent1)} "
        f"[killed at head: {len(killed_at_head)}]"
    )
    print(
        f"Latent (killed-by-failing at head): {len(latent2)} "
        f"[killed-by-failing at head: {len(killed_by_failing_at_head)}, "
        f"failing tests at head: {len(failing_tests)}]"
    )

    print(f"Missing commit files: {missing_files}")


if __name__ == "__main__":
    main()
