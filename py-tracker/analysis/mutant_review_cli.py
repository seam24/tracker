import argparse
import difflib
import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from pprint import pprint


EVENT_TYPE_AND_FORMAT = {
    "survived_to_killed": ["survived_commit", "killed_commit"],
    "killed_to_survived": ["killed_commit", "survived_commit"],
    "killed_survived_killed": [
        "killed_commit_before_survived",
        "survived_commit_first",
        "survived_commit_last",
        "killed_commit_after_survived",
    ],
    "survived_killed_survived": [
        "survived_commit_before_killed",
        "killed_commit_first",
        "killed_commit_last",
        "survived_commit_after_killed",
    ],
}


def reconstruct_mutant_code(original: str, start: int, end: int, val: str) -> str:
    """Reconstruct mutant source code from original source and mutation span."""
    return original[:start] + val + original[end:]


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def run_git(repo_path: Path, *args: str) -> str:
    """Run a git command inside repo_path and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def get_commit_subject(repo_path: Path, commit: str) -> str:
    return run_git(repo_path, "log", "--format=%s", "-n", "1", commit).strip()


def get_commit_body(repo_path: Path, commit: str) -> str:
    return run_git(repo_path, "log", "--format=%B", "-n", "1", commit).strip()


def get_file_at_commit(repo_path: Path, commit: str, file_path: str) -> str:
    """Return file content at a specific commit."""
    return run_git(repo_path, "show", f"{commit}:{file_path}")


def get_python_diff_between_commits(repo_path: Path, old_commit: str, new_commit: str) -> str:
    """Return diff between two commits restricted to .py files."""
    return run_git(repo_path, "diff", old_commit, new_commit, "--", "*.py")


def make_unified_diff(
    old_text: str | None,
    new_text: str | None,
    old_name: str = "old",
    new_name: str = "new",
) -> str:
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = (new_text or "").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_name,
            tofile=new_name,
            lineterm="",
        )
    )


def summarize_test_results(exe_res: dict, mutant_id: str) -> dict:
    """
    Return normalized test results for one mutant at one commit:
    {
        test_name: {
            "mutant_status": ...,
            "mutant_outcome": ...,
            "mutant_raw_short_msg": ...,
            "program_outcome": ...,
            "program_raw_short_msg": ...
        }
    }
    """
    mutant_tests = exe_res["mutants"][mutant_id]["tests"]
    program_tests = exe_res["program"]["tests"]

    out = {}
    for test_name, mres in mutant_tests.items():
        pres = program_tests.get(test_name, {})
        out[test_name] = {
            "mutant_status": mres.get("status"),
            "mutant_outcome": mres.get("outcome"),
            "mutant_raw_short_msg": mres.get("raw_short_msg"),
            "program_outcome": pres.get("outcome"),
            "program_raw_short_msg": pres.get("raw_short_msg"),
        }
    return out


def diff_test_results(old_tests: dict, new_tests: dict) -> dict:
    """
    Compare mutant test results between two commits.
    Keeps only changed tests.
    """
    all_tests = sorted(set(old_tests) | set(new_tests))
    changed = {}

    for test_name in all_tests:
        a = old_tests.get(test_name)
        b = new_tests.get(test_name)
        if a != b:
            changed[test_name] = {
                "old": a,
                "new": b,
            }

    return changed


def format_commit_block(repo_path: Path, commit: str) -> dict:
    return {
        "commit": commit,
        "subject": get_commit_subject(repo_path, commit),
        "message": get_commit_body(repo_path, commit),
    }


def find_mutant(saved_mutants: list, mutant_id: str | None, index: int | None):
    if mutant_id is not None:
        for m in saved_mutants:
            if m["mutant_id"] == mutant_id:
                return m
        raise ValueError(f"Mutant id not found: {mutant_id}")

    if index is not None:
        return saved_mutants[index]

    return saved_mutants[0]


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _charpos_to_lineno(text: str, pos: int) -> int:
    """0-based char pos -> 1-based line number."""
    starts = _line_starts(text)
    for i in range(len(starts) - 1):
        if starts[i] <= pos < starts[i + 1]:
            return i + 1
    return len(starts)


def extract_code_window(text: str, start: int, end: int, context_lines: int = 5) -> dict:
    """
    Return a small line-based window around [start:end].
    Output:
    {
        "start_line": ...,
        "end_line": ...,
        "mutant_start_line": ...,
        "mutant_end_line": ...,
        "snippet": ...
    }
    """
    lines = text.splitlines()
    if not lines:
        return {
            "start_line": 1,
            "end_line": 1,
            "mutant_start_line": 1,
            "mutant_end_line": 1,
            "snippet": "",
        }

    mutant_start_line = _charpos_to_lineno(text, start)
    mutant_end_line = _charpos_to_lineno(text, max(start, end - 1))

    win_start = max(1, mutant_start_line - context_lines)
    win_end = min(len(lines), mutant_end_line + context_lines)

    width = len(str(win_end))
    rendered = []
    for lineno in range(win_start, win_end + 1):
        marker = ">>" if mutant_start_line <= lineno <= mutant_end_line else "  "
        rendered.append(f"{marker} {lineno:>{width}} | {lines[lineno - 1]}")

    return {
        "start_line": win_start,
        "end_line": win_end,
        "mutant_start_line": mutant_start_line,
        "mutant_end_line": mutant_end_line,
        "snippet": "\n".join(rendered),
    }


def try_copy_to_clipboard(text: str) -> tuple[bool, str]:
    """
    Try to copy text to clipboard.
    Returns (success, method_or_reason).
    """
    # 1) pyperclip if available
    try:
        import pyperclip  # type: ignore

        pyperclip.copy(text)
        return True, "pyperclip"
    except Exception:
        pass

    # 2) platform-specific clipboard commands
    candidates = []

    if sys.platform == "darwin":
        candidates.append((["pbcopy"], text))
    elif os_name_is_windows():
        candidates.append((["clip"], text))
    else:
        if shutil.which("wl-copy"):
            candidates.append((["wl-copy"], text))
        if shutil.which("xclip"):
            candidates.append((["xclip", "-selection", "clipboard"], text))
        if shutil.which("xsel"):
            candidates.append((["xsel", "--clipboard", "--input"], text))

    for cmd, payload in candidates:
        try:
            subprocess.run(cmd, input=payload, text=True, check=True)
            return True, " ".join(cmd)
        except Exception:
            continue

    return False, "no clipboard backend found"


def os_name_is_windows() -> bool:
    return sys.platform.startswith("win")


def build_event_view(
    pid: str,
    bid: str,
    mutant_id: str | None = None,
    mutant_index: int | None = None,
    event_index: int = 0,
    context_lines: int = 5,
):
    base_data = Path("../../data/data_from_hpc")
    saved_mutants_path = base_data / "saved_mu" / f"{pid}-{bid}b.json"
    exe_res_path = base_data / f"{pid}-{bid}b" / "with_lbl"
    git_repo_path = Path(f"../../commits2/{pid}-{bid}b/{pid}")
    mutant_history_path = base_data / f"{pid}-{bid}b" / "mutant_history.json"

    saved_mutants = load_json(saved_mutants_path)
    mutant_history = load_json(mutant_history_path)

    mutant = find_mutant(saved_mutants, mutant_id, mutant_index)
    current_mutant_hist = mutant_history[mutant["mutant_id"]]

    mu_key = mutant["mutant_id"]
    mu_info = current_mutant_hist["info"]
    mu_hist = current_mutant_hist["history"]
    reformatted_mu_hist = {item["commit"]: item for item in mu_hist}

    events = list(mutant["event_details"].keys())
    if not events:
        raise ValueError(f"No events found for mutant {mu_key}")

    if event_index < 0 or event_index >= len(events):
        raise IndexError(
            f"event_index={event_index} out of range, mutant has {len(events)} event types: {events}"
        )

    event_type = events[event_index]
    event_detail = mutant["event_details"][event_type][0]  # dataset-dependent
    ordered_commits = [event_detail[k] for k in EVENT_TYPE_AND_FORMAT[event_type]]

    view = {
        "mutant_id": mu_key,
        "mutant_info": {
            "muid": mu_info["mutant_id"],
            "file": mu_info["file"],
            "module": mu_info["module"],
            "op_type": mu_info["op_type"],
            "scope": mu_info["scope"],
            "mutant_value": mu_info["mutant_value"],
        },
        "event_type": event_type,
        "event_detail": event_detail,
        "ordered_commits": ordered_commits,
        "commits": [],
    }

    for commit in ordered_commits:
        commit_view = format_commit_block(git_repo_path, commit)
        if commit in reformatted_mu_hist:
            commit_view["mutant_span"] = {
                "start_pos": reformatted_mu_hist[commit]["start_pos"],
                "end_pos": reformatted_mu_hist[commit]["end_pos"],
                "date": reformatted_mu_hist[commit]["date"],
            }
        else:
            commit_view["mutant_span"] = None

        exe_res_file = exe_res_path / f"results_{commit}.json"
        if exe_res_file.exists():
            exe_res = load_json(exe_res_file)
            commit_view["mutant_test_results"] = summarize_test_results(exe_res, mu_key)
        else:
            commit_view["mutant_test_results"] = None

        view["commits"].append(commit_view)

    if len(ordered_commits) >= 2:
        old_commit = ordered_commits[0]
        new_commit = ordered_commits[1]
        mu_file = mu_info["file"]
        mu_value = mu_info["mutant_value"]

        old_file = get_file_at_commit(git_repo_path, old_commit, mu_file)
        new_file = get_file_at_commit(git_repo_path, new_commit, mu_file)

        old_span = reformatted_mu_hist.get(old_commit)
        new_span = reformatted_mu_hist.get(new_commit)

        old_mutant_code = None
        new_mutant_code = None
        old_window = None
        new_window = None

        if old_span is not None:
            old_mutant_code = reconstruct_mutant_code(
                old_file,
                old_span["start_pos"],
                old_span["end_pos"],
                mu_value,
            )
            old_window = extract_code_window(
                old_mutant_code,
                old_span["start_pos"],
                old_span["start_pos"] + len(mu_value),
                context_lines=context_lines,
            )

        if new_span is not None:
            new_mutant_code = reconstruct_mutant_code(
                new_file,
                new_span["start_pos"],
                new_span["end_pos"],
                mu_value,
            )
            new_window = extract_code_window(
                new_mutant_code,
                new_span["start_pos"],
                new_span["start_pos"] + len(mu_value),
                context_lines=context_lines,
            )

        view["comparison"] = {
            "old_commit": format_commit_block(git_repo_path, old_commit),
            "new_commit": format_commit_block(git_repo_path, new_commit),
            "python_diff": get_python_diff_between_commits(git_repo_path, old_commit, new_commit),
            "old_original_file": old_file,
            "new_original_file": new_file,
            "old_reconstructed_mutant_file": old_mutant_code,
            "new_reconstructed_mutant_file": new_mutant_code,
            "old_reconstructed_mutant_window": old_window,
            "new_reconstructed_mutant_window": new_window,
            "reconstructed_mutant_diff": make_unified_diff(
                old_mutant_code,
                new_mutant_code,
                old_name=f"{mu_file}@{old_commit}",
                new_name=f"{mu_file}@{new_commit}",
            ),
            "test_diff": diff_test_results(
                view["commits"][0]["mutant_test_results"] or {},
                view["commits"][1]["mutant_test_results"] or {},
            ),
        }

    return view


def print_test_result_summary(test_results: dict):
    statuses = {}
    for _, v in test_results.items():
        s = v["mutant_status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"tests    : {statuses}")


def print_test_diff(test_diff: dict, show_full_test_diff: bool = False, max_changed_tests: int = 20):
    print("=" * 100)
    print("TEST RESULT DIFF")
    print("=" * 100)

    if not test_diff:
        print("<no changed tests>")
        print()
        return

    if show_full_test_diff:
        pprint(test_diff)
        print()
        return

    print(f"changed tests: {len(test_diff)}")
    for i, (test_name, diff_item) in enumerate(test_diff.items()):
        if i >= max_changed_tests:
            print(f"... truncated, use --full-test-diff to show all or raise --max-changed-tests")
            break

        old = diff_item.get("old")
        new = diff_item.get("new")

        print(f"- {test_name}")

        print(f"    old mutant_status       : {old.get('mutant_status') if old else None}")
        print(f"    new mutant_status       : {new.get('mutant_status') if new else None}")

        print(f"    old mutant_outcome      : {old.get('mutant_outcome') if old else None}")
        print(f"    new mutant_outcome      : {new.get('mutant_outcome') if new else None}")

        print(f"    old mutant_raw_short_msg: {old.get('mutant_raw_short_msg') if old else None}")
        print(f"    new mutant_raw_short_msg: {new.get('mutant_raw_short_msg') if new else None}")

        print(f"    old program_outcome     : {old.get('program_outcome') if old else None}")
        print(f"    new program_outcome     : {new.get('program_outcome') if new else None}")

        print(f"    old program_short_msg   : {old.get('program_raw_short_msg') if old else None}")
        print(f"    new program_short_msg   : {new.get('program_raw_short_msg') if new else None}")

    print()


def print_code_window_block(title: str, window: dict | None):
    print("=" * 100)
    print(title)
    print("=" * 100)
    if not window:
        print("<not available>")
        print()
        return

    print(
        f"window lines: {window['start_line']}..{window['end_line']} "
        f"(mutant lines: {window['mutant_start_line']}..{window['mutant_end_line']})"
    )
    print(window["snippet"])
    print()


def print_event_view(
    view: dict,
    show_full_code: bool = False,
    show_code_window: bool = False,
    show_full_mutant_diff: bool = False,
    show_full_test_diff: bool = False,
    max_changed_tests: int = 20,
):
    print("=" * 100)
    print("MUTANT")
    print("=" * 100)
    pprint(view["mutant_info"])
    print()

    print("=" * 100)
    print("EVENT")
    print("=" * 100)
    print("event_type:", view["event_type"])
    print("ordered_commits:")
    for c in view["ordered_commits"]:
        print("  ", c)
    print()

    print("=" * 100)
    print("COMMITS IN THIS EVENT")
    print("=" * 100)
    for item in view["commits"]:
        print(f"commit   : {item['commit']}")
        print(f"subject  : {item['subject']}")
        print(f"message  : {item['message'].strip()}")
        print(f"span     : {item['mutant_span']}")
        if item["mutant_test_results"] is None:
            print("tests    : <missing execution result>")
        else:
            print_test_result_summary(item["mutant_test_results"])
        print("-" * 100)

    if "comparison" not in view:
        return

    cmpv = view["comparison"]

    print()
    print("=" * 100)
    print("COMPARISON")
    print("=" * 100)
    print("old_commit:", cmpv["old_commit"]["commit"])
    print("old_subject:", cmpv["old_commit"]["subject"])
    print("new_commit:", cmpv["new_commit"]["commit"])
    print("new_subject:", cmpv["new_commit"]["subject"])
    print()

    print("=" * 100)
    print("PY DIFF BETWEEN COMMITS")
    print("=" * 100)
    print(cmpv["python_diff"] if cmpv["python_diff"].strip() else "<no .py diff>")
    print()

    print_test_diff(
        cmpv["test_diff"],
        show_full_test_diff=show_full_test_diff,
        max_changed_tests=max_changed_tests,
    )

    if show_code_window:
        print_code_window_block(
            "RECONSTRUCTED MUTANT WINDOW @ OLD COMMIT",
            cmpv["old_reconstructed_mutant_window"],
        )
        print_code_window_block(
            "RECONSTRUCTED MUTANT WINDOW @ NEW COMMIT",
            cmpv["new_reconstructed_mutant_window"],
        )

    if show_full_mutant_diff:
        print("=" * 100)
        print("RECONSTRUCTED MUTANT DIFF")
        print("=" * 100)
        print(cmpv["reconstructed_mutant_diff"] or "<no reconstructed mutant diff>")
        print()

    if show_full_code:
        print("=" * 100)
        print("RECONSTRUCTED MUTANT FILE @ OLD COMMIT")
        print("=" * 100)
        print(cmpv["old_reconstructed_mutant_file"] or "<not available>")
        print()

        print("=" * 100)
        print("RECONSTRUCTED MUTANT FILE @ NEW COMMIT")
        print("=" * 100)
        print(cmpv["new_reconstructed_mutant_file"] or "<not available>")
        print()


def render_event_view_to_string(
    view: dict,
    show_full_code: bool = False,
    show_code_window: bool = False,
    show_full_mutant_diff: bool = False,
    show_full_test_diff: bool = False,
    max_changed_tests: int = 20,
) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_event_view(
            view,
            show_full_code=show_full_code,
            show_code_window=show_code_window,
            show_full_mutant_diff=show_full_mutant_diff,
            show_full_test_diff=show_full_test_diff,
            max_changed_tests=max_changed_tests,
        )
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="View mutation-event transitions for analysis")
    parser.add_argument("--pid", required=True, help="project id, e.g. thefuck")
    parser.add_argument("--bid", required=True, help="bug id, e.g. 3")
    parser.add_argument("--mutant-id", default=None, help="full mutant id, e.g. thefuck/shells/fish.py#102")
    parser.add_argument("--mutant-index", type=int, default=None, help="index in saved_mutants if mutant-id not given")
    parser.add_argument("--event-index", type=int, default=0, help="which event type from mutant['event_details']")
    parser.add_argument("--full-code", action="store_true", help="print reconstructed mutant files")
    parser.add_argument(
        "--full-mutant-diff",
        action="store_true",
        help="print unified diff between reconstructed mutant files",
    )
    parser.add_argument(
        "--code-window",
        action="store_true",
        help="print only a small window around the reconstructed mutant location",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=5,
        help="number of context lines before/after mutant for --code-window",
    )
    parser.add_argument("--full-test-diff", action="store_true", help="print full test diff")
    parser.add_argument(
        "--max-changed-tests",
        type=int,
        default=20,
        help="truncate changed test printing after this many tests",
    )
    parser.add_argument("--save-json", default=None, help="optional path to save the whole event view as json")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy printed output to clipboard as well",
    )
    parser.add_argument(
        "--save-txt",
        default=None,
        help="optional path to save the printed report as plain text",
    )

    args = parser.parse_args()

    selected_code_views = sum(
        [args.full_code, args.code_window, args.full_mutant_diff]
    )
    if selected_code_views > 1:
        parser.error("Use only one of --full-code, --code-window, or --full-mutant-diff")

    view = build_event_view(
        pid=args.pid,
        bid=args.bid,
        mutant_id=args.mutant_id,
        mutant_index=args.mutant_index,
        event_index=args.event_index,
        context_lines=args.context_lines,
    )

    rendered = render_event_view_to_string(
        view,
        show_full_code=args.full_code,
        show_code_window=args.code_window,
        show_full_mutant_diff=args.full_mutant_diff,
        show_full_test_diff=args.full_test_diff,
        max_changed_tests=args.max_changed_tests,
    )

    print(rendered, end="")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(view, f, indent=2)
        print(f"\nSaved event view JSON to {args.save_json}")

    if args.save_txt:
        with open(args.save_txt, "w") as f:
            f.write(rendered)
        print(f"Saved printed report to {args.save_txt}")

    if args.copy:
        ok, method = try_copy_to_clipboard(rendered)
        if ok:
            print(f"Copied printed output to clipboard via: {method}")
        else:
            print(f"Could not copy to clipboard: {method}")


if __name__ == "__main__":
    main()