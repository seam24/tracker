from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import difflib
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle


# ================== CONFIG ==================
BASE = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-4b")
COMMITS_FILE = BASE / "commits_to_run.txt"
RESULTS_DIR = BASE / "with_lbl"
MUTANT_HISTORY_PATH = BASE / "mutant_history.json"

REPO_ROOT = Path("/home/rooti/shared/LatentMu/chk/thefuck-4b/thefuck")

CHRONOLOGICAL = True
SPLIT_INCOMPETENT = True

# NEW: show anomalous as its own color; if False -> treat as killed (red)
SPLIT_ANOMALOUS = True

# UI tweaks
MAX_XTICKS = 30
MAX_KILLER_TESTS_TO_PRINT = 10
MAX_SAMPLE_TESTS_FOR_GRAY = 10

# Diff printing
PRINT_UNIFIED_DIFF = True

# ---- Expected mask + colors (no hatching) ----
PAINT_NOT_PROPAGATED = True
NOT_PROPAGATED_COLOR = "#ffffff"  # not expected / not propagated
UNRESOLVED_COLOR = "#bdbdbd"      # expected but unresolved (darker gray)
# ============================================

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

# Make zooming crisp (avoid blurry interpolation)
mpl.rcParams["image.interpolation"] = "nearest"
mpl.rcParams["image.resample"] = False


# ---------- status logic ----------
def mutant_status_from_tests(mtests: Dict[str, Any]) -> str:
    if not mtests:
        return "unknown"
    statuses = [(t.get("status") or "unknown").lower() for t in mtests.values()]

    # Priority: incompetent/anomalous override killed for aggregated status
    if any(s == "incompetent" for s in statuses):
        return "incompetent"
    if any(s == "anomalous" for s in statuses):
        return "anomalous"
    if any(s == "killed" for s in statuses):
        return "killed"
    if all(s == "survived" for s in statuses):
        return "survived"
    return "unknown"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def head_mutants(head_path: Path) -> List[str]:
    return sorted((load_json(head_path).get("mutants") or {}).keys())


def summarize_status_counts(mtests: Dict[str, Any]) -> Dict[str, int]:
    c = Counter()
    for td in (mtests or {}).values():
        c[(td.get("status") or "unknown").lower()] += 1
    return dict(c)


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


def editor_hint(path: Path) -> str:
    ed = os.environ.get("EDITOR")
    if ed:
        return f"{ed} {path}"
    return f"less {path}   # or: jq . {path} | less"


def clip_msg(x: Any, limit: int = 240) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\n", "\\n")
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


# ================== UNIFIED DIFF FROM mutant_history.json ==================
def reconstruct_mutant_code(original: str, start: int, end: int, val: str) -> str:
    return original[:start] + val + original[end:]


def git_show_file(repo: Path, commit: str, rel_path: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{rel_path}"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        return None


def unified_diff(original: str, mutated: str, path: str, commit: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(True),
            mutated.splitlines(True),
            fromfile=f"{path}@{commit[:8]} (original)",
            tofile=f"{path}@{commit[:8]} (mutant)",
            lineterm="",
            n=4,
        )
    )


def get_mutant_meta(history: Dict[str, Any], mid: str, commit: str) -> Optional[Tuple[str, int, int, str]]:
    """
    Return (file, start_pos, end_pos, mutant_value) for this mutant at this commit.

    Prefer exact commit entry in mh["history"].
    Fallback to mh["info"]["mutation_start_pos"/"mutation_end_pos"] if exact not present.
    """
    mh = history.get(mid)
    if not mh:
        return None

    info = mh.get("info") or {}
    file_path = info.get("file") or mh.get("file")
    mutant_value = info.get("mutant_value")

    if not file_path or mutant_value is None:
        return None

    for h in (mh.get("history") or []):
        if h.get("commit") == commit:
            sp = h.get("start_pos")
            ep = h.get("end_pos")
            if isinstance(sp, int) and isinstance(ep, int):
                return file_path, sp, ep, str(mutant_value)

    sp = info.get("mutation_start_pos")
    ep = info.get("mutation_end_pos")
    if isinstance(sp, int) and isinstance(ep, int):
        return file_path, sp, ep, str(mutant_value)

    return None


def show_unified_diff(mutant_history: Dict[str, Any], commit: str, mid: str) -> None:
    if not PRINT_UNIFIED_DIFF:
        return

    meta = get_mutant_meta(mutant_history, mid, commit)
    if not meta:
        print("\nUNIFIED DIFF: ❌ No mutant metadata available (mutant_history.json missing entry or fields).")
        return

    path, start, end, val = meta
    original = git_show_file(REPO_ROOT, commit, path)
    if original is None:
        print("\nUNIFIED DIFF: ❌ Could not load source file from git show.")
        return

    if not (0 <= start <= end <= len(original)):
        print("\nUNIFIED DIFF: ❌ Offsets invalid for this commit/file.")
        print(f"  file_len={len(original)} start={start} end={end}")
        return

    mutated = reconstruct_mutant_code(original, start, end, val)

    print("\nUNIFIED DIFF:")
    print(unified_diff(original, mutated, path, commit))


# ================== MAIN ==================
def main() -> None:
    commits = [c.strip() for c in COMMITS_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
    if not commits:
        raise SystemExit(f"No commits in {COMMITS_FILE}")

    # commits: newest -> oldest
    head = commits[0]
    head_path = RESULTS_DIR / f"results_{head}.json"
    if not head_path.exists():
        raise SystemExit(f"Missing HEAD results: {head_path}")

    if not MUTANT_HISTORY_PATH.exists():
        raise SystemExit(f"Missing mutant history file: {MUTANT_HISTORY_PATH}")
    mutant_history = load_json(MUTANT_HISTORY_PATH)

    mutant_ids = head_mutants(head_path)
    mid_to_row = {mid: i for i, mid in enumerate(mutant_ids)}

    # -------- expected existence mask --------
    expected0 = compute_expected_from_history(mutant_ids, commits, mutant_history)  # newest->oldest

    x_commits = list(reversed(commits)) if CHRONOLOGICAL else commits
    col_of_x = {c: i for i, c in enumerate(x_commits)}
    commit_to_idx0 = {c: i for i, c in enumerate(commits)}  # newest->oldest

    expected = np.zeros((len(mutant_ids), len(x_commits)), dtype=bool)
    for c in x_commits:
        expected[:, col_of_x[c]] = expected0[:, commit_to_idx0[c]]

    # Matrix encoding
    GRAY, SURVIVED, KILLED, INCOMPETENT, ANOMALOUS = 0, 1, 2, 3, 4
    mat = np.full((len(mutant_ids), len(x_commits)), GRAY, dtype=np.uint8)

    # Build matrix
    jobs = [(c, str(RESULTS_DIR / f"results_{c}.json")) for c in x_commits]
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(worker_parse_commit, c, p) for c, p in jobs]
        for fut in as_completed(futs):
            commit, stmap, missing = fut.result()
            if missing:
                continue
            col = col_of_x[commit]
            for mid, st in stmap.items():
                r = mid_to_row.get(mid)
                if r is None:
                    continue
                if st == "survived":
                    mat[r, col] = SURVIVED
                elif st == "killed":
                    mat[r, col] = KILLED
                elif st == "incompetent":
                    mat[r, col] = INCOMPETENT if SPLIT_INCOMPETENT else KILLED
                elif st == "anomalous":
                    mat[r, col] = ANOMALOUS if SPLIT_ANOMALOUS else KILLED
                else:
                    mat[r, col] = GRAY

    # ---- plot (adaptive sizing like before) ----
    n_rows, n_cols = mat.shape
    fig_w = min(24, max(10, n_cols / 25))
    fig_h = min(20, max(6, n_rows / 250))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)

    # Unresolved color is configurable now
    # Colors: unresolved, survived, killed, incompetent, anomalous
    if SPLIT_INCOMPETENT and SPLIT_ANOMALOUS:
        cmap = ListedColormap([UNRESOLVED_COLOR, "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        ticks = [0, 1, 2, 3, 4]
        ticklabels = ["unresolved (expected)", "survived", "killed", "incompetent", "anomalous"]
    elif SPLIT_INCOMPETENT and not SPLIT_ANOMALOUS:
        cmap = ListedColormap([UNRESOLVED_COLOR, "#2ca02c", "#d62728", "#ff7f0e"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ticks = [0, 1, 2, 3]
        ticklabels = ["unresolved (expected)", "survived", "killed (incl. anomalous)", "incompetent"]
    elif not SPLIT_INCOMPETENT and SPLIT_ANOMALOUS:
        cmap = ListedColormap([UNRESOLVED_COLOR, "#2ca02c", "#d62728", "#9467bd"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 4.5]
        ticks = [0, 1, 2, 4]
        ticklabels = ["unresolved (expected)", "survived", "killed (incl. incompetent)", "anomalous"]
    else:
        cmap = ListedColormap([UNRESOLVED_COLOR, "#2ca02c", "#d62728"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        ticks = [0, 1, 2]
        ticklabels = ["unresolved (expected)", "survived", "killed (incl. incompetent/anomalous)"]

    norm = BoundaryNorm(bounds, cmap.N)

    # 1) draw normal status heatmap
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",  # crisp
        cmap=cmap,
        norm=norm,
    )
    try:
        im.set_resample(False)  # if supported
    except Exception:
        pass

    # 2) overlay "not propagated" as solid color (treat as "mutant doesn't exist here")
    if PAINT_NOT_PROPAGATED:
        overlay = np.ma.masked_where(expected, np.ones_like(expected, dtype=float))
        ax.imshow(
            overlay,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap([NOT_PROPAGATED_COLOR]),
            alpha=1.0,
        )

    ax.set_title("Click a cell → inspect: status counts + unified diff")
    ax.set_xlabel("Commits")
    ax.set_ylabel("Mutants")
    ax.set_yticks([])

    # X ticks subset
    if n_cols <= MAX_XTICKS:
        xt = np.arange(n_cols)
    else:
        xt = np.linspace(0, n_cols - 1, MAX_XTICKS, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([x_commits[i][:8] for i in xt], rotation=90, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, ticks=ticks, fraction=0.02, pad=0.02)
    cbar.ax.set_yticklabels(ticklabels)

    plt.tight_layout()

    # highlight rectangle
    highlight = Rectangle((0, 0), 1, 1, fill=False, linewidth=2)
    ax.add_patch(highlight)
    highlight.set_visible(False)

    # cache commit JSONs
    commit_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def get_commit_data(commit: str) -> Optional[Dict[str, Any]]:
        if commit in commit_cache:
            return commit_cache[commit]
        p = RESULTS_DIR / f"results_{commit}.json"
        commit_cache[commit] = load_json(p) if p.exists() else None
        return commit_cache[commit]

    def explain_kill(commit_data: Dict[str, Any], mid: str) -> None:
        program_tests = ((commit_data.get("program") or {}).get("tests")) or {}
        md = ((commit_data.get("mutants") or {}).get(mid)) or {}
        mtests = (md.get("tests") or {}) or {}

        killers = []
        for tid, td in mtests.items():
            st = (td.get("status") or "unknown").lower()
            # include anomalous too (umbrella)
            if st in ("killed", "incompetent", "anomalous"):
                killers.append((tid, st, td))

        if not killers:
            print("No killer/anomalous test rows found in mutant tests (unexpected for this cell).")
            return

        print(f"\nDETAILS (up to {MAX_KILLER_TESTS_TO_PRINT} relevant tests):")
        for tid, st, mtd in killers[:MAX_KILLER_TESTS_TO_PRINT]:
            ptd = program_tests.get(tid) or {}
            pm = clip_msg(ptd.get("raw_short_msg"))
            mm = clip_msg(mtd.get("raw_short_msg"))

            print(f"\n- Test: {tid}")
            print(f"  status: {st}")
            print(f"  program raw_short_msg: {pm!r}" if pm else "  program raw_short_msg: (null/empty)")
            print(f"  mutant  raw_short_msg: {mm!r}" if mm else "  mutant  raw_short_msg: (null/empty)")

        if len(killers) > MAX_KILLER_TESTS_TO_PRINT:
            print(f"\n(+ {len(killers) - MAX_KILLER_TESTS_TO_PRINT} more relevant tests not shown)")

    # ---------- CLICK HANDLER ----------
    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if row < 0 or col < 0 or row >= mat.shape[0] or col >= mat.shape[1]:
            return

        highlight.set_xy((col - 0.5, row - 0.5))
        highlight.set_visible(True)
        fig.canvas.draw_idle()

        commit = x_commits[col]

        # IMPORTANT: if NOT expected -> treat as "mutant does not exist here" and STOP
        if not expected[row, col]:
            print("\n" + "=" * 120)
            print(f"CLICK row={row} col={col}")
            print(f"Commit: {commit} ({commit[:8]})")
            print("Cell: not propagated (mutant not expected / does not exist at this commit)")
            return

        # From here on, expected==True, so mutant is considered present / meaningful
        mid = mutant_ids[row]
        results_path = RESULTS_DIR / f"results_{commit}.json"

        cell_val = int(mat[row, col])
        if cell_val == GRAY:
            cell_label = "unresolved (expected)"
        elif cell_val == SURVIVED:
            cell_label = "survived"
        elif cell_val == KILLED:
            cell_label = "killed"
        elif cell_val == INCOMPETENT:
            cell_label = "incompetent"
        elif cell_val == ANOMALOUS:
            cell_label = "anomalous"
        else:
            cell_label = f"unknown_val({cell_val})"

        print("\n" + "=" * 120)
        print(f"CLICK row={row} col={col}")
        print(f"Commit: {commit} ({commit[:8]})")
        print(f"Mutant: {mid}")
        print(f"Cell: {cell_val} => {cell_label}")
        print(f"Results file: {results_path}")
        print(f"Open with: {editor_hint(results_path)}")

        data = get_commit_data(commit)
        if data is None:
            print("❌ results file is MISSING (no test details).")
            # diff is still meaningful if expected==True
            show_unified_diff(mutant_history, commit, mid)
            return

        program_tests = ((data.get("program") or {}).get("tests")) or {}
        mutants = data.get("mutants", {}) or {}
        print(f"Program tests in file: {len(program_tests)}")

        if mid not in mutants:
            print("⚠ Mutant entry is NOT present in this commit file.")
            show_unified_diff(mutant_history, commit, mid)
            return

        md = mutants[mid] or {}
        mtests = (md.get("tests") or {}) or {}
        agg = mutant_status_from_tests(mtests)
        counts = summarize_status_counts(mtests)

        print("Mutant entry exists: YES")
        print(f"Aggregated status (from tests): {agg}")
        print(f"Per-status counts: {counts}")

        # unified diff (original vs reconstructed mutant)
        show_unified_diff(mutant_history, commit, mid)

        # details for killed/incompetent/anomalous cells
        if cell_val in (KILLED, INCOMPETENT, ANOMALOUS):
            explain_kill(data, mid)
            return

        # gray cell details sample
        if cell_val == GRAY:
            items = list(mtests.items())
            if not items:
                print("Mutant has no tests dictionary → nothing to show.")
                return
            print(f"\nGRAY CELL DETAILS (up to {MAX_SAMPLE_TESTS_FOR_GRAY} test rows):")
            for tid, td in items[:MAX_SAMPLE_TESTS_FOR_GRAY]:
                st = (td.get("status") or "unknown").lower()
                mm = clip_msg(td.get("raw_short_msg"))
                pm = clip_msg((program_tests.get(tid) or {}).get("raw_short_msg"))
                line = f" - {tid}: {st}"
                if mm:
                    line += f" | mutant_msg={mm!r}"
                if pm:
                    line += f" | program_msg={pm!r}"
                print(line)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
