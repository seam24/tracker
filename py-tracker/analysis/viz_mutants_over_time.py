from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ================== CONFIG ==================
CHRONOLOGICAL = True        # True: oldest->newest on x-axis, False: newest->oldest
USE_HATCHING = False        # Toggle hatching for "not propagated / not expected"
HATCH_PATTERN = "//"        # Hatch style, e.g. "//", "\\\\", "xx", ".."

SPLIT_INCOMPETENT = True   # default: incompetent counts as killed
# ============================================


# ---------- status logic ----------
def mutant_status_from_tests(mtests: Dict[str, Any]) -> str:
    """
    Returns: 'killed', 'incompetent', 'survived', or 'unknown'

    Behavior note:
      - If SPLIT_INCOMPETENT is False, we still return 'incompetent' here,
        but it will be mapped to KILLED in the matrix fill stage.
    """
    if not mtests:
        return "unknown"

    statuses = [(t.get("status") or "unknown").lower() for t in mtests.values()]

    # precedence: killed > incompetent > survived > unknown
    if any(s == "killed" for s in statuses):
        return "killed"
    if any(s == "incompetent" for s in statuses):
        return "incompetent"
    if all(s == "survived" for s in statuses):
        return "survived"
    return "unknown"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def head_mutants(head_path: Path) -> List[str]:
    data = load_json(head_path)
    return sorted((data.get("mutants") or {}).keys())


# ---------- worker ----------
def worker_parse_commit(commit: str, results_path: str) -> Tuple[str, Dict[str, str], bool]:
    p = Path(results_path)
    if not p.exists():
        return commit, {}, True

    data = load_json(p)
    mutants = data.get("mutants", {}) or {}
    st: Dict[str, str] = {}
    for mid, md in mutants.items():
        st[mid] = mutant_status_from_tests((md.get("tests", {}) or {}))
    return commit, st, False


# ---------- expected existence (from mutant_history.json) ----------
def compute_expected_from_history(
    mutant_ids: List[str],
    commits_newest_to_oldest: List[str],
    mutant_history: Dict[str, Any],
) -> np.ndarray:
    """
    expected[r, c] = True means mutant row r is expected to exist at commit column c
    (in newest->oldest index space), based on mutant_history.json using the prefix rule:

      - If mutant is NOT present in newest commit => expected is all False.
      - If it is present in newest commit => expected is True for a consecutive prefix starting at 0.

    IMPORTANT: This uses only the witness commits in mutant_history['history'].
    """
    commit_to_idx = {c: i for i, c in enumerate(commits_newest_to_oldest)}
    n_rows = len(mutant_ids)
    n_cols = len(commits_newest_to_oldest)
    expected = np.zeros((n_rows, n_cols), dtype=bool)

    for r, mid in enumerate(mutant_ids):
        mh = mutant_history.get(mid)
        if not mh:
            continue
        hist = mh.get("history", []) or []
        idxs = sorted({commit_to_idx[h["commit"]] for h in hist if h.get("commit") in commit_to_idx})
        if not idxs:
            continue

        # must exist at newest commit, otherwise never
        if idxs[0] != 0:
            continue

        # compute prefix length k using witness indices
        s = set(idxs)
        k = 0
        while k < n_cols and k in s:
            k += 1

        expected[r, :k] = True

    return expected


def main() -> None:
    base = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-4b")
    commits_file = base / "commits_to_run.txt"
    results_dir = base / "with_lbl"
    mutant_history_path = base / "mutant_history.json"  # adjust if elsewhere

    commits = [c.strip() for c in commits_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    if not commits:
        raise SystemExit(f"No commits in {commits_file}")

    # commits is newest -> oldest
    head = commits[0]
    head_path = results_dir / f"results_{head}.json"
    if not head_path.exists():
        raise SystemExit(f"Missing HEAD results: {head_path}")

    # Rows = all mutants at HEAD
    mutant_ids = head_mutants(head_path)
    mid_to_row = {mid: i for i, mid in enumerate(mutant_ids)}
    n_rows = len(mutant_ids)

    # Expected existence from history in newest->oldest space
    if not mutant_history_path.exists():
        raise SystemExit(f"Missing mutant history file: {mutant_history_path}")
    mutant_history = load_json(mutant_history_path)
    expected0 = compute_expected_from_history(mutant_ids, commits, mutant_history)  # newest->oldest

    # X axis ordering for visualization
    x_commits = list(reversed(commits)) if CHRONOLOGICAL else commits
    n_cols = len(x_commits)
    col_of_x = {c: i for i, c in enumerate(x_commits)}
    commit_to_idx0 = {c: i for i, c in enumerate(commits)}  # newest->oldest

    # Map expected0 into x-axis order
    expected = np.zeros((n_rows, n_cols), dtype=bool)
    for c in x_commits:
        expected[:, col_of_x[c]] = expected0[:, commit_to_idx0[c]]

    # Matrix encoding:
    # 0 = gray (expected-but-unresolved OR not-propagated/not-expected)
    # 1 = green (survived)
    # 2 = red (killed)
    # 3 = orange (incompetent) [only used if SPLIT_INCOMPETENT=True]
    NOT_PROP_OR_UNRESOLVED, SURVIVED, KILLED, INCOMPETENT = 0, 1, 2, 3
    mat = np.full((n_rows, n_cols), NOT_PROP_OR_UNRESOLVED, dtype=np.uint8)

    # Parse results in parallel for all commits
    jobs = [(c, str(results_dir / f"results_{c}.json")) for c in x_commits]
    missing_commits: List[str] = []

    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(worker_parse_commit, c, p) for c, p in jobs]
        for fut in as_completed(futs):
            commit, stmap, missing = fut.result()
            if missing:
                missing_commits.append(commit)
                continue

            col = col_of_x[commit]
            for mid, st in stmap.items():
                r = mid_to_row.get(mid)
                if r is None:
                    continue  # only show HEAD mutants
                if st == "survived":
                    mat[r, col] = SURVIVED
                elif st == "killed":
                    mat[r, col] = KILLED
                elif st == "incompetent":
                    mat[r, col] = INCOMPETENT if SPLIT_INCOMPETENT else KILLED
                else:
                    # unknown in results => keep gray (unresolved)
                    mat[r, col] = NOT_PROP_OR_UNRESOLVED

    # ---- plotting ----
    if SPLIT_INCOMPETENT:
        cmap = ListedColormap([
            "#e0e0e0",  # 0 = gray
            "#2ca02c",  # 1 = survived
            "#d62728",  # 2 = killed
            "#ff7f0e",  # 3 = incompetent (orange)
        ])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ticks = [0, 1, 2, 3]
        ticklabels = [
            "unresolved",
            "survived",
            "killed",
            "incompetent",
        ]
    else:
        cmap = ListedColormap([
            "#e0e0e0",  # 0 = gray
            "#2ca02c",  # 1 = survived
            "#d62728",  # 2 = killed (incl. incompetent)
        ])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        ticks = [0, 1, 2]
        ticklabels = [
            "expected but unresolved (gray)",
            "survived",
            "killed (& incl. incompetent)",
        ]

    norm = BoundaryNorm(bounds, cmap.N)

    fig_w = min(24, max(10, n_cols / 25))
    fig_h = min(20, max(6, n_rows / 250))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    # Hatch ONLY cells that are not expected (not propagated / not expected)
    if USE_HATCHING:
        hatch_mask = (~expected).astype(int)
        ax.contourf(
            hatch_mask,
            levels=[0.5, 1.5],
            hatches=[HATCH_PATTERN],
            colors="none",
            alpha=0.0,
        )

    ax.set_title("Mutant status over commits")
    ax.set_xlabel("Commits (time)")
    ax.set_ylabel("Mutants")
    ax.set_yticks([])

    # X ticks: show subset for readability
    max_xticks = 30
    if n_cols <= max_xticks:
        xt = np.arange(n_cols)
    else:
        xt = np.linspace(0, n_cols - 1, max_xticks, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([x_commits[i][:8] for i in xt], rotation=90, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, ticks=ticks, fraction=0.02, pad=0.02)
    cbar.ax.set_yticklabels(ticklabels)

    fig.tight_layout()
    out = base / "mutants_over_time.png"
    fig.savefig(out)
    plt.close(fig)

    print(f"Wrote: {out}")
    print(f"Matrix size: {n_rows} mutants × {n_cols} commits")
    if missing_commits:
        missing_set = set(missing_commits)
        missing_in_order = [c for c in x_commits if c in missing_set]
        print(f"Warning: missing results files: {len(missing_in_order)}")
        print("First 10 missing commits:", [c[:10] for c in missing_in_order[:10]])


if __name__ == "__main__":
    main()
