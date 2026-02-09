from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, List, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

BASE = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-7b")

# X axis order
CHRONOLOGICAL = True              # True: oldest->newest, False: newest->oldest

# What to plot
PLOT_MODE = "count"              # "status" or "counts"

# Which tests to use
TEST_MODE = "all"                 # "all" | "baseline" | "new"
#   all      : use all tests in each commit
#   baseline : use ONLY tests from oldest commit (T_base)
#   new      : use ONLY tests added since oldest commit (T' - T_base)

# --- status plot options ---
SPLIT_INCOMPETENT = True          # if False -> incompetent is counted as killed (red)

ENABLE_FLAKY = True               # classify killed-by-timeouts as "flaky"
TIMEOUT_PREFIX = "Failed: Timeout"

# ---- NOT PROPAGATED visualization (NO HATCHING) ----
PAINT_NOT_PROPAGATED = True
NOT_PROPAGATED_COLOR = "#f5f5f5"  # overlay color for (~expected) cells

# --- counts plot options (killed-test count heatmaps) ---
COUNT_INCOMPETENT_AS_KILL = False
MASK_NOT_EXPECTED_IN_COUNTS = False

WRITE_RAW_COUNTS = True
WRITE_ROW_NORMALIZED = False
WRITE_ROW_ZSCORE = False

VMAX_PERCENTILE = 99              # for robust vmax in raw counts

# ============================================================

COMMITS_FILE = BASE / "commits_to_run.txt"
RESULTS_DIR = BASE / "with_lbl"
MUTANT_HISTORY_PATH = BASE / "mutant_history.json"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def head_mutants(path: Path) -> List[str]:
    data = load_json(path)
    return sorted((data.get("mutants") or {}).keys())


def extract_test_set(data: Dict[str, Any]) -> Set[str]:
    """
    Best-effort extraction of tests present at a commit.
    Tries:
      1) top-level "tests" dict/list
      2) union of per-mutant tests keys
    """
    top = data.get("tests")
    if isinstance(top, dict):
        return set(top.keys())
    if isinstance(top, list):
        return set(str(x) for x in top)

    tests: Set[str] = set()
    mutants = data.get("mutants") or {}
    if isinstance(mutants, dict):
        for md in mutants.values():
            if not isinstance(md, dict):
                continue
            mtests = md.get("tests") or {}
            if isinstance(mtests, dict):
                tests.update(mtests.keys())
    return tests


def filter_tests(mtests: Dict[str, Any], allowed: Optional[Set[str]]) -> Dict[str, Any]:
    if not mtests:
        return {}
    if allowed is None:
        return mtests
    if not allowed:
        return {}
    return {k: v for k, v in mtests.items() if k in allowed}


def choose_xticks(n_cols: int, x_commits: List[str], max_xticks: int = 30):
    if n_cols <= max_xticks:
        xt = np.arange(n_cols)
    else:
        xt = np.linspace(0, n_cols - 1, max_xticks, dtype=int)
    labels = [x_commits[i][:8] for i in xt]
    return xt, labels


# ------------------------------------------------------------
# Expected existence (mutant_history prefix logic)
# ------------------------------------------------------------
def compute_expected(mutant_ids: List[str], commits_newest_to_oldest: List[str], mutant_history: Dict[str, Any]) -> np.ndarray:
    """
    expected[r,c] in newest->oldest index space, using prefix rule:
      - if mutant not in newest => never expected
      - if in newest => expected is True for a consecutive prefix starting at 0
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
        if idxs[0] != 0:
            continue

        s = set(idxs)
        k = 0
        while k < n_cols and k in s:
            k += 1
        expected[r, :k] = True

    return expected


# ------------------------------------------------------------
# STATUS logic (+ flaky)
# ------------------------------------------------------------
def is_timeout(msg: Any) -> bool:
    return isinstance(msg, str) and msg.startswith(TIMEOUT_PREFIX)


def mutant_status(mtests: Dict[str, Any], program_tests: Dict[str, Any]) -> str:
    """
    Returns: 'survived' | 'killed' | 'incompetent' | 'flaky' | 'unknown'
    """
    if not mtests:
        return "unknown"

    sts = [(t.get("status") or "unknown").lower() for t in mtests.values()]

    if any(s == "incompetent" for s in sts):
        return "incompetent"

    if any(s == "killed" for s in sts):
        if ENABLE_FLAKY:
            killers = [tn for tn, t in mtests.items() if (t.get("status") or "").lower() == "killed"]
            if killers:
                def killer_is_timeout(tn: str) -> bool:
                    mmsg = (mtests.get(tn) or {}).get("raw_short_msg")
                    pmsg = (program_tests.get(tn) or {}).get("raw_short_msg")
                    return is_timeout(mmsg) or is_timeout(pmsg)

                if all(killer_is_timeout(tn) for tn in killers):
                    return "flaky"
        return "killed"

    if all(s == "survived" for s in sts):
        return "survived"

    return "unknown"


# ------------------------------------------------------------
# Worker (returns either status strings or (killed,total))
# ------------------------------------------------------------
def worker(commit: str, results_path: str, base_tests: Set[str]) -> Tuple[str, Dict[str, Any], bool]:
    p = Path(results_path)
    if not p.exists():
        return commit, {}, True

    data = load_json(p)
    mutants = data.get("mutants", {}) or {}
    program_tests = ((data.get("program") or {}).get("tests") or {})

    # choose allowed tests for this commit
    if TEST_MODE == "all":
        allowed = None
    elif TEST_MODE == "baseline":
        allowed = base_tests
    elif TEST_MODE == "new":
        commit_tests = extract_test_set(data)
        allowed = commit_tests - base_tests
    else:
        raise ValueError(f"Unknown TEST_MODE: {TEST_MODE}")

    out: Dict[str, Any] = {}

    for mid, md in mutants.items():
        if not isinstance(md, dict):
            continue
        mtests_full = (md.get("tests") or {})
        if not isinstance(mtests_full, dict):
            mtests_full = {}

        mtests = filter_tests(mtests_full, allowed)

        if PLOT_MODE == "status":
            out[mid] = mutant_status(mtests, program_tests)
        else:
            total = len(mtests)
            killed = 0
            for t in mtests.values():
                st = (t.get("status") or "unknown").lower()
                if st == "killed":
                    killed += 1
                elif COUNT_INCOMPETENT_AS_KILL and st == "incompetent":
                    killed += 1
            out[mid] = (killed, total)

    return commit, out, False


# ------------------------------------------------------------
# COUNT plot helpers
# ------------------------------------------------------------
def robust_vmax(mat: np.ndarray, percentile: float) -> float:
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(finite, percentile))


def row_max_normalize(mat: np.ndarray, eps: float = 1e-9, min_row_max_to_keep: float = 1.0) -> np.ndarray:
    row_max = np.nanmax(mat, axis=1)
    out = mat / (row_max[:, None] + eps)
    bad = ~np.isfinite(row_max) | (row_max < min_row_max_to_keep)
    out[bad, :] = np.nan
    return out


def row_zscore(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = np.nanmean(mat, axis=1)
    sigma = np.nanstd(mat, axis=1)
    out = (mat - mu[:, None]) / (sigma[:, None] + eps)
    bad = ~np.isfinite(sigma) | (sigma < eps)
    out[bad, :] = np.nan
    return out


def save_heatmap_continuous(
    mat: np.ndarray,
    x_commits: List[str],
    title: str,
    cbar_label: str,
    out_path: Path,
    *,
    vmin: float | None,
    vmax: float | None,
):
    n_rows, n_cols = mat.shape
    fig_w = min(24, max(10, n_cols / 25))
    fig_h = min(20, max(6, n_rows / 250))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Commits (time)")
    ax.set_ylabel("Mutants")
    ax.set_yticks([])

    xt, labels = choose_xticks(n_cols, x_commits, max_xticks=30)
    ax.set_xticks(xt)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    commits = [c.strip() for c in COMMITS_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
    if not commits:
        raise SystemExit(f"No commits in {COMMITS_FILE}")

    # commits are newest -> oldest
    head = commits[0]
    head_path = RESULTS_DIR / f"results_{head}.json"
    if not head_path.exists():
        raise SystemExit(f"Missing HEAD results: {head_path}")

    mutant_ids = head_mutants(head_path)
    mid_to_row = {mid: i for i, mid in enumerate(mutant_ids)}
    n_rows = len(mutant_ids)

    if not MUTANT_HISTORY_PATH.exists():
        raise SystemExit(f"Missing mutant_history.json: {MUTANT_HISTORY_PATH}")
    mutant_history = load_json(MUTANT_HISTORY_PATH)

    expected0 = compute_expected(mutant_ids, commits, mutant_history)  # newest->oldest

    # axis order
    x_commits = list(reversed(commits)) if CHRONOLOGICAL else commits
    n_cols = len(x_commits)

    col_of_x = {c: i for i, c in enumerate(x_commits)}
    commit_to_idx0 = {c: i for i, c in enumerate(commits)}  # newest->oldest

    # map expected0 -> expected in x-axis order
    expected = np.zeros((n_rows, n_cols), dtype=bool)
    for c in x_commits:
        expected[:, col_of_x[c]] = expected0[:, commit_to_idx0[c]]

    # baseline tests (from oldest commit)
    base_tests: Set[str] = set()
    if TEST_MODE in ("baseline", "new"):
        oldest_commit = commits[-1]
        oldest_path = RESULTS_DIR / f"results_{oldest_commit}.json"
        if not oldest_path.exists():
            raise SystemExit(f"Missing OLDEST results (needed for base test set): {oldest_path}")
        oldest_data = load_json(oldest_path)
        base_tests = extract_test_set(oldest_data)

    # allocate matrices
    if PLOT_MODE == "status":
        # original status codes
        # 0 = gray (unresolved)
        # 1 = survived
        # 2 = killed
        # 3 = incompetent (if split)
        # 4 = flaky
        mat = np.zeros((n_rows, n_cols), dtype=np.uint8)
    else:
        killed_count_mat = np.full((n_rows, n_cols), np.nan, dtype=float)
        total_tests_mat = np.full((n_rows, n_cols), np.nan, dtype=float)

    # parallel parse
    jobs = [(c, str(RESULTS_DIR / f"results_{c}.json")) for c in x_commits]
    missing_commits: List[str] = []

    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(worker, c, p, base_tests) for c, p in jobs]
        for fut in as_completed(futs):
            commit, outmap, missing = fut.result()
            if missing:
                missing_commits.append(commit)
                continue

            col = col_of_x[commit]
            for mid, val in outmap.items():
                r = mid_to_row.get(mid)
                if r is None:
                    continue

                if PLOT_MODE == "status":
                    if val == "survived":
                        mat[r, col] = 1
                    elif val == "killed":
                        mat[r, col] = 2
                    elif val == "incompetent":
                        mat[r, col] = 3 if SPLIT_INCOMPETENT else 2
                    elif val == "flaky":
                        mat[r, col] = 4
                    else:
                        mat[r, col] = 0
                else:
                    killed, total = val
                    killed_count_mat[r, col] = float(killed)
                    total_tests_mat[r, col] = float(total)

    # ------------------------------------------------------------
    # Plot STATUS (paint not propagated with overlay color)
    # ------------------------------------------------------------
    if PLOT_MODE == "status":
        if ENABLE_FLAKY and SPLIT_INCOMPETENT:
            cmap = ListedColormap(["#bdbdbd", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"])
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            ticks = [0, 1, 2, 3, 4]
            ticklabels = ["unresolved", "survived", "killed", "incompetent", "flaky (timeout kills)"]
        elif ENABLE_FLAKY and not SPLIT_INCOMPETENT:
            cmap = ListedColormap(["#bdbdbd", "#2ca02c", "#d62728", "#9467bd"])
            bounds = [-0.5, 0.5, 1.5, 2.5, 4.5]
            ticks = [0, 1, 2, 4]
            ticklabels = ["unresolved", "survived", "killed (incl. incompetent)", "flaky (timeout kills)"]
        elif (not ENABLE_FLAKY) and SPLIT_INCOMPETENT:
            cmap = ListedColormap(["#bdbdbd", "#2ca02c", "#d62728", "#ff7f0e"])
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            ticks = [0, 1, 2, 3]
            ticklabels = ["unresolved", "survived", "killed", "incompetent"]
        else:
            cmap = ListedColormap(["#bdbdbd", "#2ca02c", "#d62728"])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            ticks = [0, 1, 2]
            ticklabels = ["unresolved", "survived", "killed (incl. incompetent)"]

        norm = BoundaryNorm(bounds, cmap.N)

        fig_w = min(24, max(10, n_cols / 25))
        fig_h = min(20, max(6, n_rows / 250))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

        # 1) draw normal status heatmap
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

        # 2) overlay "not propagated" cells with a SOLID color (no hatching)
        if PAINT_NOT_PROPAGATED:
            # mask where expected=True; keep only (~expected)
            overlay = np.ma.masked_where(expected, np.ones_like(expected, dtype=float))
            ax.imshow(
                overlay,
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap([NOT_PROPAGATED_COLOR]),
                alpha=1.0,
            )

        title = "Mutant status over commits"
        if TEST_MODE == "baseline":
            title += " (baseline tests only: T_base from oldest commit)"
        elif TEST_MODE == "new":
            title += " (new tests only: T'−T from oldest commit)"
        ax.set_title(title)

        ax.set_xlabel("Commits (time)")
        ax.set_ylabel("Mutants")
        ax.set_yticks([])

        xt, labels = choose_xticks(n_cols, x_commits, max_xticks=30)
        ax.set_xticks(xt)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, ticks=ticks, fraction=0.02, pad=0.02)
        cbar.ax.set_yticklabels(ticklabels)

        fig.tight_layout()
        out = BASE / f"mutants_status_tests-{TEST_MODE}.png"
        fig.savefig(out)
        plt.close(fig)

        print(f"Wrote: {out}")
        print(f"Matrix size: {n_rows} mutants × {n_cols} commits")

    # ------------------------------------------------------------
    # Plot COUNTS
    # ------------------------------------------------------------
    else:
        if MASK_NOT_EXPECTED_IN_COUNTS:
            killed_count_mat[~expected] = np.nan
            total_tests_mat[~expected] = np.nan

        # Sort rows by row max killed count
        row_max = np.nanmax(killed_count_mat, axis=1)
        order = np.argsort(np.nan_to_num(row_max, nan=-1.0))[::-1]
        killed_count_mat = killed_count_mat[order, :]
        total_tests_mat = total_tests_mat[order, :]

        suffix = f"tests-{TEST_MODE}" + ("_masked" if MASK_NOT_EXPECTED_IN_COUNTS else "")

        if WRITE_RAW_COUNTS:
            vmax = robust_vmax(killed_count_mat, VMAX_PERCENTILE)
            save_heatmap_continuous(
                killed_count_mat,
                x_commits,
                title="Killed-test count per mutant per commit (raw counts)",
                cbar_label="# killing tests",
                out_path=BASE / f"mutants_killed_test_count_matrix_{suffix}.png",
                vmin=0.0,
                vmax=vmax,
            )

        if WRITE_ROW_NORMALIZED:
            row_norm = row_max_normalize(killed_count_mat, eps=1e-9, min_row_max_to_keep=1.0)
            save_heatmap_continuous(
                row_norm,
                x_commits,
                title="Killed-test count per mutant per commit (row-normalized by mutant max)",
                cbar_label="killed_count / row_max (0..1)",
                out_path=BASE / f"mutants_killed_test_count_row_normalized_{suffix}.png",
                vmin=0.0,
                vmax=1.0,
            )

        if WRITE_ROW_ZSCORE:
            z = row_zscore(killed_count_mat, eps=1e-9)
            z_clip = np.clip(z, -3, 3)
            save_heatmap_continuous(
                z_clip,
                x_commits,
                title="Killed-test count per mutant per commit (row z-score, clipped to [-3,3])",
                cbar_label="z-score (clipped)",
                out_path=BASE / f"mutants_killed_test_count_row_zscore_{suffix}.png",
                vmin=-3.0,
                vmax=3.0,
            )

        print(f"Matrix size: {n_rows} mutants × {n_cols} commits")

    if missing_commits:
        missing_set = set(missing_commits)
        missing_in_order = [c for c in x_commits if c in missing_set]
        print(f"Warning: missing results files: {len(missing_in_order)}")
        print("First 10 missing commits:", [c[:10] for c in missing_in_order[:10]])


if __name__ == "__main__":
    main()
