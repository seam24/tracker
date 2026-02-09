from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, List, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# pip install plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# ========================= CONFIG ===========================
# ============================================================

BASE = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-7b")

COMMITS_FILE = BASE / "commits_to_run.txt"
RESULTS_DIR = BASE / "with_lbl"
MUTANT_HISTORY_PATH = BASE / "mutant_history.json"

# X axis order
CHRONOLOGICAL = True              # True: oldest->newest, False: newest->oldest

# Which tests to use
TEST_MODE = "all"                 # "all" | "baseline" | "new"

# Count semantics
COUNT_INCOMPETENT_AS_KILL = False

# Mask not expected (optional)
MASK_NOT_EXPECTED = False

# Rate scaling for log visualization of small rates
RATE_SCALE = 1000.0

# Output html
OUT_HTML = BASE / "mutant_kill_explorer.html"

# Performance / UI
MAX_X_TICKS = 40                  # for labels; hover still has full
SHOW_Y_LABELS = False             # y labels are huge; set True only for small sets

# ---- Timeout handling (applies to COUNTING) ----
TIMEOUT_PREFIX = "Failed: Timeout"
# "count"   -> timeout-kills are counted as killed (default)
# "exclude" -> timeout-kill tests are removed from BOTH total and killed
TIMEOUT_KILL_MODE = "exclude"       # "count" | "exclude"

# ============================================================


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def head_mutants(path: Path) -> List[str]:
    data = load_json(path)
    return sorted((data.get("mutants") or {}).keys())


def extract_test_set(data: Dict[str, Any]) -> Set[str]:
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
# Timeout-aware helpers (mirrors your status logic idea)
# ------------------------------------------------------------
def is_timeout(msg: Any) -> bool:
    return isinstance(msg, str) and msg.startswith(TIMEOUT_PREFIX)


def is_timeout_kill(test_obj: Dict[str, Any], program_test_obj: Dict[str, Any]) -> bool:
    """
    True iff this test is marked 'killed' AND the short message indicates a timeout
    on either mutant run or program run.
    """
    st = (test_obj.get("status") or "unknown").lower()
    if st != "killed":
        return False
    mmsg = test_obj.get("raw_short_msg")
    pmsg = (program_test_obj or {}).get("raw_short_msg")
    return is_timeout(mmsg) or is_timeout(pmsg)


# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------
def worker(commit: str, results_path: str, base_tests: Set[str]) -> Tuple[str, Dict[str, Tuple[int, int]], bool]:
    p = Path(results_path)
    if not p.exists():
        return commit, {}, True

    data = load_json(p)
    mutants = data.get("mutants", {}) or {}
    program_tests = ((data.get("program") or {}).get("tests") or {})

    if TEST_MODE == "all":
        allowed = None
    elif TEST_MODE == "baseline":
        allowed = base_tests
    elif TEST_MODE == "new":
        commit_tests = extract_test_set(data)
        allowed = commit_tests - base_tests
    else:
        raise ValueError(f"Unknown TEST_MODE: {TEST_MODE}")

    out: Dict[str, Tuple[int, int]] = {}
    for mid, md in mutants.items():
        if not isinstance(md, dict):
            continue
        mtests_full = (md.get("tests") or {})
        if not isinstance(mtests_full, dict):
            mtests_full = {}

        mtests = filter_tests(mtests_full, allowed)

        killed = 0
        total = 0

        for tn, t in mtests.items():
            # Optional exclusion of timeout-kill tests from BOTH total and killed
            if TIMEOUT_KILL_MODE == "exclude":
                ptest = (program_tests.get(tn) or {})
                if is_timeout_kill(t, ptest):
                    continue

            total += 1

            st = (t.get("status") or "unknown").lower()
            if st == "killed":
                killed += 1
            elif COUNT_INCOMPETENT_AS_KILL and st == "incompetent":
                killed += 1

        out[mid] = (killed, total)

    return commit, out, False


def downsample_xticks(labels: List[str], max_ticks: int) -> Tuple[List[int], List[str]]:
    n = len(labels)
    if n <= max_ticks:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, max_ticks, dtype=int).tolist()
        idx = sorted(set(idx))
    lab = [labels[i][:8] for i in idx]
    return idx, lab


def col_median_center(mat: np.ndarray) -> np.ndarray:
    med = np.nanmedian(mat, axis=0)
    return mat - med[None, :]


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

    killed = np.full((n_rows, n_cols), np.nan, dtype=float)
    total = np.full((n_rows, n_cols), np.nan, dtype=float)

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
            for mid, (k, t) in outmap.items():
                r = mid_to_row.get(mid)
                if r is None:
                    continue
                killed[r, col] = float(k)
                total[r, col] = float(t)

    if missing_commits:
        ms = set(missing_commits)
        print(f"Warning: missing results files: {len(ms)}")
        print("First 10 missing commits:", [c[:10] for c in x_commits if c in ms][:10])

    if MASK_NOT_EXPECTED:
        killed[~expected] = np.nan
        total[~expected] = np.nan

    # Derived representations
    rate = killed / np.maximum(total, 1.0)
    rate[~np.isfinite(rate)] = np.nan

    log1p_killed = np.log1p(killed)
    log_rate = np.log1p(rate * RATE_SCALE)

    cc_log1p_killed = col_median_center(log1p_killed)
    cc_log_rate = col_median_center(log_rate)

    # Views
    views = {
        "killed": (killed, "Killed tests"),
        "rate": (rate, "Kill rate (killed/total)"),
        "log1p(killed)": (log1p_killed, "log1p(killed)"),
        f"log1p(rate*{RATE_SCALE:g})": (log_rate, f"log1p(rate*{RATE_SCALE:g})"),
        "cc log1p(killed)": (cc_log1p_killed, "log1p(killed) - median(commit)"),
        f"cc log1p(rate*{RATE_SCALE:g})": (cc_log_rate, f"log1p(rate*{RATE_SCALE:g}) - median(commit)"),
    }
    view_names = list(views.keys())

    # Initial view
    z0, ylabel0 = views[view_names[0]]

    # Labels
    x_labels = [c[:8] for c in x_commits]
    if SHOW_Y_LABELS and n_rows <= 200:
        y_labels = mutant_ids
    else:
        y_labels = [str(i) for i in range(n_rows)]

    # Hover context: show mutant id, commit, killed/total (after timeout filtering!), value
    customdata = np.empty((n_rows, n_cols, 3), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            customdata[r, c, 0] = mutant_ids[r]
            customdata[r, c, 1] = x_labels[c]
            kk = killed[r, c]
            tt = total[r, c]
            customdata[r, c, 2] = f"{int(kk)}/{int(tt)}" if np.isfinite(kk) and np.isfinite(tt) else "NA"

    # Build figure
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.04,
        subplot_titles=("Heatmap (click a cell/row)", "Selected mutant over time")
    )

    heat = go.Heatmap(
        z=z0,
        x=x_labels,
        y=y_labels,
        customdata=customdata,
        hovertemplate=(
            "mutant: %{customdata[0]}<br>"
            "commit: %{customdata[1]}<br>"
            "killed/total: %{customdata[2]}<br>"
            "value: %{z}<extra></extra>"
        ),
        colorbar=dict(title=ylabel0),
    )
    fig.add_trace(heat, row=1, col=1)

    r0 = 0
    line = go.Scatter(
        x=x_labels,
        y=z0[r0, :],
        mode="lines",
        name=mutant_ids[r0],
        hovertemplate="commit: %{x}<br>value: %{y}<extra></extra>"
    )
    fig.add_trace(line, row=1, col=2)
    fig.update_yaxes(title_text=ylabel0, row=1, col=2)

    tick_idx, tick_lab = downsample_xticks(x_labels, MAX_X_TICKS)
    fig.update_xaxes(tickmode="array", tickvals=[x_labels[i] for i in tick_idx], ticktext=tick_lab, row=1, col=1)
    fig.update_xaxes(tickmode="array", tickvals=[x_labels[i] for i in tick_idx], ticktext=tick_lab, row=1, col=2)

    fig.update_layout(
        title=f"Mutant kill explorer (TEST_MODE={TEST_MODE}, TIMEOUT_KILL_MODE={TIMEOUT_KILL_MODE})",
        height=800,
        margin=dict(l=40, r=20, t=70, b=40),
    )

    # Dropdown menus
    buttons = []
    for i, name in enumerate(view_names):
        z, yl = views[name]
        buttons.append(dict(
            label=name,
            method="update",
            args=[
                {"z": [z, None], "y": [None, z[r0, :]]},
                {"yaxis2": {"title": yl},
                 "title": f"Mutant kill explorer — view: {name} (TEST_MODE={TEST_MODE}, TIMEOUT_KILL_MODE={TIMEOUT_KILL_MODE})"},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            x=0.01, y=1.08,
            showactive=True,
            active=0,
            buttons=buttons
        )]
    )

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")

    # click-to-update-line JS
    js = r"""
<script>
document.addEventListener("DOMContentLoaded", function() {
  var gd = document.querySelectorAll(".plotly-graph-div")[0];
  if(!gd) return;

  function updateLine(rowIndex) {
    var z = gd.data[0].z;      // current view's z matrix
    var y = z[rowIndex];       // row series
    var mutantId = gd.data[0].customdata[rowIndex][0][0]; // customdata[r][c][0], c=0
    Plotly.restyle(gd, {y: [y], name: [mutantId]}, [1]);  // trace 1 is line
  }

  gd.on('plotly_click', function(ev) {
    if(!ev || !ev.points || ev.points.length === 0) return;
    var pt = ev.points[0];
    var row = pt.pointIndex[0];
    updateLine(row);
  });
});
</script>
"""
    html = html.replace("</body>", js + "\n</body>")

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote interactive explorer: {OUT_HTML}")


if __name__ == "__main__":
    main()
