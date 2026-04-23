import json
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from difflib import SequenceMatcher
from functools import lru_cache


def reconstruct_mutant_code(original: str, start: int, end: int, val: str) -> str:
    """Give source code of the file of the mutant in that commit,
    and specify start and end and val then get the mutant's source code."""
    return original[:start] + val + original[end:]


# ----------------------------
# CONFIG
# ----------------------------
pid, bid = "cookiecutter", "2"

saved_mutants_path = Path(f"../../data/data_from_hpc/saved_mu/{pid}-{bid}b.json")
exe_res_path = Path(f"../../data/data_from_hpc/{pid}-{bid}b/with_lbl")
git_repo_path = Path(f"../../commits2/{pid}-{bid}b/{pid}")
mutant_history_path = Path(f"../../data/data_from_hpc/{pid}-{bid}b/mutant_history.json")

event_type_and_format = {
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


# ----------------------------
# DATA HELPERS
# ----------------------------
def load_json(path: Path):
    with open(path, "r") as fr:
        return json.load(fr)


def run_git(args):
    result = subprocess.run(
        ["git", *args],
        cwd=git_repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git command failed: {' '.join(args)}")
    return result.stdout


@lru_cache(maxsize=None)
def get_file_content_at_commit(commit: str, file_path: str) -> str:
    try:
        return run_git(["show", f"{commit}:{file_path}"])
    except Exception:
        return ""


@lru_cache(maxsize=None)
def get_python_changed_files(commit_a: str, commit_b: str):
    try:
        out = run_git(["diff", "--name-only", commit_a, commit_b, "--", "*.py"])
        files = [line.strip() for line in out.splitlines() if line.strip().endswith(".py")]
        return files
    except Exception:
        return []


@lru_cache(maxsize=None)
def get_unified_diff_for_file(commit_a: str, commit_b: str, file_path: str) -> str:
    try:
        return run_git(["diff", commit_a, commit_b, "--", file_path])
    except Exception:
        return ""


def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_events(mutant):
    """Return all events for a mutant, even if there are multiple per type."""
    out = []
    event_details_all = mutant.get("event_details", {})
    for event_type, events in event_details_all.items():
        for idx, details in enumerate(events):
            out.append(
                {
                    "event_type": event_type,
                    "event_index_inside_type": idx,
                    "details": details,
                }
            )
    return out


def build_commit_pairs_for_event(event_type: str, event_details: dict):
    """
    For a 4-commit event:
      c0->c1, c1->c2, c2->c3
    For a 2-commit event:
      c0->c1
    """
    labels = event_type_and_format[event_type]
    commits = [event_details.get(label) for label in labels]

    pairs = []
    for i in range(len(commits) - 1):
        if commits[i] and commits[i + 1]:
            pairs.append(
                {
                    "left_label": labels[i],
                    "right_label": labels[i + 1],
                    "left_commit": commits[i],
                    "right_commit": commits[i + 1],
                }
            )
    return pairs


def load_execution_results_for_commit(commit: str):
    path = exe_res_path / f"results_{commit}.json"
    if not path.exists():
        return None
    return load_json(path)


def compare_tests_between_commits(mutant_id: str, commit_a: str, commit_b: str):
    """
    Compare mutant test results between two commits.
    Returns only changed tests to keep it useful in the GUI.
    """
    res_a = load_execution_results_for_commit(commit_a)
    res_b = load_execution_results_for_commit(commit_b)

    if res_a is None or res_b is None:
        return []

    tests_a = safe_get(res_a, "mutants", mutant_id, "tests", default={}) or {}
    tests_b = safe_get(res_b, "mutants", mutant_id, "tests", default={}) or {}

    prog_a = safe_get(res_a, "program", "tests", default={}) or {}
    prog_b = safe_get(res_b, "program", "tests", default={}) or {}

    all_tests = sorted(set(tests_a.keys()) | set(tests_b.keys()))
    rows = []

    for test_name in all_tests:
        ma = tests_a.get(test_name, {})
        mb = tests_b.get(test_name, {})
        pa = prog_a.get(test_name, {})
        pb = prog_b.get(test_name, {})

        row = {
            "test": test_name,
            "mutant_status_a": ma.get("status", ""),
            "mutant_status_b": mb.get("status", ""),
            "mutant_outcome_a": ma.get("outcome", ""),
            "mutant_outcome_b": mb.get("outcome", ""),
            "mutant_msg_a": ma.get("raw_short_msg", ""),
            "mutant_msg_b": mb.get("raw_short_msg", ""),
            "prog_outcome_a": pa.get("outcome", ""),
            "prog_outcome_b": pb.get("outcome", ""),
            "prog_msg_a": pa.get("raw_short_msg", ""),
            "prog_msg_b": pb.get("raw_short_msg", ""),
        }

        changed = (
            row["mutant_status_a"] != row["mutant_status_b"]
            or row["mutant_outcome_a"] != row["mutant_outcome_b"]
            or row["mutant_msg_a"] != row["mutant_msg_b"]
            or row["prog_outcome_a"] != row["prog_outcome_b"]
            or row["prog_msg_a"] != row["prog_msg_b"]
        )

        if changed:
            rows.append(row)

    return rows


def get_mutant_info(saved_mutant, all_mutant_history):
    current_mutant_hist = all_mutant_history[saved_mutant["mutant_id"]]

    mu_hist = current_mutant_hist["history"]
    reformatted_mu_hist = {item["commit"]: item for item in mu_hist}

    info = current_mutant_hist["info"]
    return {
        "mu_key": saved_mutant["mutant_id"],
        "muid": info["mutant_id"],
        "mu_file": info["file"],
        "mu_module": info["module"],
        "mu_op": info["op_type"],
        "mu_scope": info["scope"],
        "mu_value": info["mutant_value"],
        "mu_hist": mu_hist,
        "reformatted_mu_hist": reformatted_mu_hist,
    }


def reconstruct_mutant_source_for_commit(mu_file, mu_value, reformatted_mu_hist, commit):
    """
    Reconstruct mutant source at a specific commit if that commit is present in mutant history.
    Returns:
      {
        "original": str,
        "mutant": str,
        "start": int or None,
        "end": int or None,
      }
    """
    original = get_file_content_at_commit(commit, mu_file)
    hist_entry = reformatted_mu_hist.get(commit)

    if not original:
        return {"original": "", "mutant": "", "start": None, "end": None}

    if hist_entry is None:
        return {"original": original, "mutant": original, "start": None, "end": None}

    start = hist_entry["start_pos"]
    end = hist_entry["end_pos"]
    mutant = reconstruct_mutant_code(original, start, end, mu_value)
    return {"original": original, "mutant": mutant, "start": start, "end": end}


# ----------------------------
# GUI
# ----------------------------
class MutantViewerApp(tk.Tk):
    def __init__(self, saved_mutants, mutant_history):
        super().__init__()
        self.title("Mutant Event Viewer")
        self.geometry("1700x950")

        self.saved_mutants = saved_mutants
        self.mutant_history = mutant_history

        self.mutant_index = 0
        self.event_index = 0
        self.transition_index = 0

        self.current_events = []
        self.current_pairs = []
        self.current_changed_files = []

        self._build_ui()
        self._load_current_mutant()

    def _build_ui(self):
        # top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Button(top, text="Prev Mutant", command=self.prev_mutant).pack(side="left")
        ttk.Button(top, text="Next Mutant", command=self.next_mutant).pack(side="left", padx=(4, 12))

        ttk.Label(top, text="Mutant index:").pack(side="left")
        self.mutant_index_var = tk.StringVar()
        self.mutant_index_entry = ttk.Entry(top, textvariable=self.mutant_index_var, width=8)
        self.mutant_index_entry.pack(side="left", padx=4)
        ttk.Button(top, text="Go", command=self.go_to_mutant).pack(side="left", padx=(0, 12))

        ttk.Button(top, text="Prev Event", command=self.prev_event).pack(side="left")
        ttk.Button(top, text="Next Event", command=self.next_event).pack(side="left", padx=(4, 12))

        ttk.Button(top, text="Prev Transition", command=self.prev_transition).pack(side="left")
        ttk.Button(top, text="Next Transition", command=self.next_transition).pack(side="left", padx=(4, 12))

        self.summary_label = ttk.Label(top, text="", anchor="w")
        self.summary_label.pack(side="left", fill="x", expand=True, padx=8)

        # info panel
        info_frame = ttk.LabelFrame(self, text="Mutant Info")
        info_frame.pack(fill="x", padx=8, pady=(0, 8))

        self.info_text = tk.Text(info_frame, height=7, wrap="word")
        self.info_text.pack(fill="x", padx=6, pady=6)
        self.info_text.configure(state="disabled")

        # notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # tab 1: mutant code
        self.code_tab = ttk.Frame(notebook)
        notebook.add(self.code_tab, text="Mutant Code")

        code_top = ttk.Frame(self.code_tab)
        code_top.pack(fill="x", padx=6, pady=6)

        self.code_pair_label = ttk.Label(code_top, text="")
        self.code_pair_label.pack(anchor="w")

        code_pane = ttk.PanedWindow(self.code_tab, orient="horizontal")
        code_pane.pack(fill="both", expand=True, padx=6, pady=6)

        left_code_frame = ttk.LabelFrame(code_pane, text="Left Commit Mutant Source")
        right_code_frame = ttk.LabelFrame(code_pane, text="Right Commit Mutant Source")
        code_pane.add(left_code_frame, weight=1)
        code_pane.add(right_code_frame, weight=1)

        self.left_code = tk.Text(left_code_frame, wrap="none")
        self.right_code = tk.Text(right_code_frame, wrap="none")
        self.left_code.pack(fill="both", expand=True)
        self.right_code.pack(fill="both", expand=True)

        self._configure_code_tags(self.left_code)
        self._configure_code_tags(self.right_code)

        # tab 2: tests
        self.tests_tab = ttk.Frame(notebook)
        notebook.add(self.tests_tab, text="Test Result Diff")

        self.tests_label = ttk.Label(self.tests_tab, text="")
        self.tests_label.pack(anchor="w", padx=6, pady=(6, 2))

        test_columns = (
            "test",
            "mutant_status_a",
            "mutant_status_b",
            "mutant_outcome_a",
            "mutant_outcome_b",
            "prog_outcome_a",
            "prog_outcome_b",
            "mutant_msg_a",
            "mutant_msg_b",
            "prog_msg_a",
            "prog_msg_b",
        )

        self.tests_tree = ttk.Treeview(self.tests_tab, columns=test_columns, show="headings")
        for col in test_columns:
            self.tests_tree.heading(col, text=col)
            self.tests_tree.column(col, width=140, anchor="w")
        self.tests_tree.pack(fill="both", expand=True, padx=6, pady=6)

        # tab 3: file diff
        self.diff_tab = ttk.Frame(notebook)
        notebook.add(self.diff_tab, text=".py File Diffs")

        diff_top = ttk.Frame(self.diff_tab)
        diff_top.pack(fill="x", padx=6, pady=6)

        self.diff_label = ttk.Label(diff_top, text="")
        self.diff_label.pack(anchor="w")

        diff_main = ttk.PanedWindow(self.diff_tab, orient="horizontal")
        diff_main.pack(fill="both", expand=True, padx=6, pady=6)

        file_list_frame = ttk.LabelFrame(diff_main, text="Changed .py files")
        file_view_frame = ttk.Frame(diff_main)
        diff_main.add(file_list_frame, weight=0)
        diff_main.add(file_view_frame, weight=1)

        self.file_listbox = tk.Listbox(file_list_frame, width=45)
        self.file_listbox.pack(fill="both", expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        file_pane = ttk.PanedWindow(file_view_frame, orient="horizontal")
        file_pane.pack(fill="both", expand=True)

        left_file_frame = ttk.LabelFrame(file_pane, text="Left Commit")
        right_file_frame = ttk.LabelFrame(file_pane, text="Right Commit")
        file_pane.add(left_file_frame, weight=1)
        file_pane.add(right_file_frame, weight=1)

        self.left_file_text = tk.Text(left_file_frame, wrap="none")
        self.right_file_text = tk.Text(right_file_frame, wrap="none")
        self.left_file_text.pack(fill="both", expand=True)
        self.right_file_text.pack(fill="both", expand=True)

        self._configure_diff_tags(self.left_file_text)
        self._configure_diff_tags(self.right_file_text)

    def _configure_code_tags(self, widget):
        widget.tag_configure("mutant_span", background="#fff2a8")

    def _configure_diff_tags(self, widget):
        widget.tag_configure("same", background="white")
        widget.tag_configure("replace", background="#fff2a8")
        widget.tag_configure("delete", background="#ffd6d6")
        widget.tag_configure("insert", background="#d9ffd9")

    def set_info_text(self, text):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def go_to_mutant(self):
        try:
            idx = int(self.mutant_index_var.get())
        except ValueError:
            messagebox.showerror("Error", "Mutant index must be an integer")
            return

        if not (0 <= idx < len(self.saved_mutants)):
            messagebox.showerror("Error", f"Mutant index must be in [0, {len(self.saved_mutants)-1}]")
            return

        self.mutant_index = idx
        self.event_index = 0
        self.transition_index = 0
        self._load_current_mutant()

    def prev_mutant(self):
        if self.mutant_index > 0:
            self.mutant_index -= 1
            self.event_index = 0
            self.transition_index = 0
            self._load_current_mutant()

    def next_mutant(self):
        if self.mutant_index < len(self.saved_mutants) - 1:
            self.mutant_index += 1
            self.event_index = 0
            self.transition_index = 0
            self._load_current_mutant()

    def prev_event(self):
        if self.event_index > 0:
            self.event_index -= 1
            self.transition_index = 0
            self._load_current_event()

    def next_event(self):
        if self.event_index < len(self.current_events) - 1:
            self.event_index += 1
            self.transition_index = 0
            self._load_current_event()

    def prev_transition(self):
        if self.transition_index > 0:
            self.transition_index -= 1
            self._load_current_transition()

    def next_transition(self):
        if self.transition_index < len(self.current_pairs) - 1:
            self.transition_index += 1
            self._load_current_transition()

    def _load_current_mutant(self):
        saved_mutant = self.saved_mutants[self.mutant_index]
        self.current_saved_mutant = saved_mutant
        self.current_mutant_info = get_mutant_info(saved_mutant, self.mutant_history)
        self.current_events = flatten_events(saved_mutant)

        self.mutant_index_var.set(str(self.mutant_index))

        info = self.current_mutant_info
        info_text = (
            f"mutant_index: {self.mutant_index}\n"
            f"mu_key: {info['mu_key']}\n"
            f"muid: {info['muid']}\n"
            f"file: {info['mu_file']}\n"
            f"module: {info['mu_module']}\n"
            f"scope: {info['mu_scope']}\n"
            f"op: {info['mu_op']}\n"
            f"mutant_value: {info['mu_value']}\n"
            f"num_events: {len(self.current_events)}"
        )
        self.set_info_text(info_text)

        if not self.current_events:
            self.summary_label.config(text="No events for this mutant")
            self.current_pairs = []
            self._clear_all_views()
            return

        self._load_current_event()

    def _load_current_event(self):
        if not self.current_events:
            return

        event = self.current_events[self.event_index]
        self.current_event = event
        self.current_pairs = build_commit_pairs_for_event(event["event_type"], event["details"])

        if self.transition_index >= len(self.current_pairs):
            self.transition_index = 0

        self.summary_label.config(
            text=(
                f"Mutant {self.mutant_index + 1}/{len(self.saved_mutants)} | "
                f"Event {self.event_index + 1}/{len(self.current_events)} | "
                f"type={event['event_type']} | "
                f"Transitions={len(self.current_pairs)}"
            )
        )

        if not self.current_pairs:
            self._clear_all_views()
            return

        self._load_current_transition()

    def _load_current_transition(self):
        if not self.current_pairs:
            return

        pair = self.current_pairs[self.transition_index]
        self.current_pair = pair

        left_commit = pair["left_commit"]
        right_commit = pair["right_commit"]

        self.code_pair_label.config(
            text=f"{pair['left_label']} ({left_commit})  ->  {pair['right_label']} ({right_commit})"
        )
        self.tests_label.config(
            text=f"Changed test results between commits: {left_commit} -> {right_commit}"
        )
        self.diff_label.config(
            text=f"Changed .py files between commits: {left_commit} -> {right_commit}"
        )

        self._update_code_tab(left_commit, right_commit)
        self._update_tests_tab(left_commit, right_commit)
        self._update_file_list(left_commit, right_commit)

        self.summary_label.config(
            text=(
                f"Mutant {self.mutant_index + 1}/{len(self.saved_mutants)} | "
                f"Event {self.event_index + 1}/{len(self.current_events)} | "
                f"Transition {self.transition_index + 1}/{len(self.current_pairs)} | "
                f"{pair['left_label']} -> {pair['right_label']}"
            )
        )

    def _clear_all_views(self):
        for widget in [self.left_code, self.right_code, self.left_file_text, self.right_file_text]:
            widget.delete("1.0", "end")

        for item in self.tests_tree.get_children():
            self.tests_tree.delete(item)

        self.file_listbox.delete(0, "end")

    def _update_code_tab(self, left_commit, right_commit):
        info = self.current_mutant_info
        mu_file = info["mu_file"]
        mu_value = info["mu_value"]
        reformatted_mu_hist = info["reformatted_mu_hist"]

        left_data = reconstruct_mutant_source_for_commit(mu_file, mu_value, reformatted_mu_hist, left_commit)
        right_data = reconstruct_mutant_source_for_commit(mu_file, mu_value, reformatted_mu_hist, right_commit)

        self._set_code_widget(self.left_code, left_data["mutant"], left_data["start"], left_data["end"], mu_value)
        self._set_code_widget(self.right_code, right_data["mutant"], right_data["start"], right_data["end"], mu_value)

    def _set_code_widget(self, widget, text, start, end, mu_value):
        widget.delete("1.0", "end")
        widget.insert("1.0", text)

        if start is not None and end is not None:
            # highlight inserted mutant span in reconstructed source
            highlight_end = start + len(str(mu_value))
            try:
                start_idx = f"1.0+{start}c"
                end_idx = f"1.0+{highlight_end}c"
                widget.tag_remove("mutant_span", "1.0", "end")
                widget.tag_add("mutant_span", start_idx, end_idx)
                widget.see(start_idx)
            except Exception:
                pass

    def _update_tests_tab(self, left_commit, right_commit):
        for item in self.tests_tree.get_children():
            self.tests_tree.delete(item)

        rows = compare_tests_between_commits(
            self.current_saved_mutant["mutant_id"],
            left_commit,
            right_commit,
        )

        for row in rows:
            self.tests_tree.insert(
                "",
                "end",
                values=(
                    row["test"],
                    row["mutant_status_a"],
                    row["mutant_status_b"],
                    row["mutant_outcome_a"],
                    row["mutant_outcome_b"],
                    row["prog_outcome_a"],
                    row["prog_outcome_b"],
                    row["mutant_msg_a"],
                    row["mutant_msg_b"],
                    row["prog_msg_a"],
                    row["prog_msg_b"],
                ),
            )

    def _update_file_list(self, left_commit, right_commit):
        self.file_listbox.delete(0, "end")
        self.current_changed_files = get_python_changed_files(left_commit, right_commit)

        for f in self.current_changed_files:
            self.file_listbox.insert("end", f)

        self.left_file_text.delete("1.0", "end")
        self.right_file_text.delete("1.0", "end")

        if self.current_changed_files:
            self.file_listbox.selection_set(0)
            self.on_file_select(None)
        else:
            self.left_file_text.insert("1.0", "No changed .py files")
            self.right_file_text.insert("1.0", "No changed .py files")

    def on_file_select(self, _event):
        if not self.file_listbox.curselection():
            return

        idx = self.file_listbox.curselection()[0]
        file_path = self.current_changed_files[idx]

        left_commit = self.current_pair["left_commit"]
        right_commit = self.current_pair["right_commit"]

        left_content = get_file_content_at_commit(left_commit, file_path)
        right_content = get_file_content_at_commit(right_commit, file_path)

        self._show_side_by_side_diff(self.left_file_text, self.right_file_text, left_content, right_content)

    def _show_side_by_side_diff(self, left_widget, right_widget, left_content, right_content):
        left_widget.delete("1.0", "end")
        right_widget.delete("1.0", "end")

        left_lines = left_content.splitlines()
        right_lines = right_content.splitlines()

        sm = SequenceMatcher(a=left_lines, b=right_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for l, r in zip(left_lines[i1:i2], right_lines[j1:j2]):
                    self._insert_diff_line(left_widget, l, "same")
                    self._insert_diff_line(right_widget, r, "same")

            elif tag == "replace":
                max_len = max(i2 - i1, j2 - j1)
                left_chunk = left_lines[i1:i2]
                right_chunk = right_lines[j1:j2]
                for k in range(max_len):
                    l = left_chunk[k] if k < len(left_chunk) else ""
                    r = right_chunk[k] if k < len(right_chunk) else ""
                    self._insert_diff_line(left_widget, l, "replace")
                    self._insert_diff_line(right_widget, r, "replace")

            elif tag == "delete":
                for l in left_lines[i1:i2]:
                    self._insert_diff_line(left_widget, l, "delete")
                    self._insert_diff_line(right_widget, "", "delete")

            elif tag == "insert":
                for r in right_lines[j1:j2]:
                    self._insert_diff_line(left_widget, "", "insert")
                    self._insert_diff_line(right_widget, r, "insert")

    def _insert_diff_line(self, widget, line, tag):
        start = widget.index("end-1c")
        widget.insert("end", line + "\n")
        end = widget.index("end-1c")
        widget.tag_add(tag, start, end)


def main():
    try:
        saved_mutants = load_json(saved_mutants_path)
        mutant_history = load_json(mutant_history_path)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Loading Error", str(e))
        return

    app = MutantViewerApp(saved_mutants, mutant_history)
    app.mainloop()


if __name__ == "__main__":
    main()