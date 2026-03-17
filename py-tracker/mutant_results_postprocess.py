"""
Read program results, process and store structured more compact data.
Optionally label mutant test results (killed/survived/unknown) by comparing to program tests.

Feature: one can ENABLE/DISABLE writing the intermediate (extracted) JSON files.
- If ENABLE_LABELING=True and WRITE_INTERMEDIATE=False:
    no extracted JSON bundle is written (least I/O)
- If ENABLE_LABELING=True and WRITE_INTERMEDIATE=True:
    writes extracted JSON bundle too
- If ENABLE_LABELING=False:
    labeling skipped; (optionally) writes extracted JSON bundle

Extra feature:
- If FLAG_INCOMPETENT=True:
    when a mutant would be labeled "killed" and its (normalized) message starts with "TypeError",
    label it as "incompetent" instead.

Extra feature:
- Anomalous labeling (umbrella):
    when a mutant would be labeled "killed" and its (normalized) message starts with one of the
    configured anomalous prefixes, label it as "anomalous" instead.

Output modes (independent switches):
- WRITE_LABELED_DATA (requires ENABLE_LABELING=True):
    writes labeled *bundle* file: OUT_DIR_LABELED/<out_name>.json

- WRITE_LBL_PER_FILE (works with or without labeling):
    writes split files in OUT_DIR_LBL_PER_FILE/<commit>/:
      - program_<commit>.json  (or pres_<commit>.json if input stem == pres_fix)
      - mutant_<sanitized_mutant_key>_<commit>.json for each mutant

    Filename rules:
      mutant_key "thefuck/shells/fish.py#0"  ->  mutant_thefuck_shells_fish_py_0_<commit>.json
      program file (normal results file)     ->  program_<commit>.json
      pres_fix file (file stem == "pres_fix")->  pres_<commit>.json

- WRITE_INTERMEDIATE:
    writes extracted (unlabeled) bundle file: OUT_DIR_EXTRACTED/<out_name>.json

IMPORTANT INVARIANT:
- "intermediate" output is ALWAYS WITHOUT "status" (raw extracted only).
"""

from __future__ import annotations

from pathlib import Path
import ndjson
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import copy


# -------------------------
# USER SETTINGS (edit these)
# -------------------------
DATA_DIR = Path("/home/rooti/shared/LatentMu/data/data_from_hpc/thefuck-4b/program_results/")

OUT_DIR_EXTRACTED = DATA_DIR.parent / "no_lbl"
OUT_DIR_LABELED = DATA_DIR.parent / "with_lbl"

# Label computation
ENABLE_LABELING = True

# Write outputs
WRITE_INTERMEDIATE = False        # writes OUT_DIR_EXTRACTED/<out_name>.json (UNLABELED bundle)
WRITE_LABELED_DATA = True        # writes OUT_DIR_LABELED/<out_name>.json (requires ENABLE_LABELING=True)
WRITE_LBL_PER_FILE = False       # writes OUT_DIR_LBL_PER_FILE/<commit>/... (works with or without labeling)

# If enabled: killed + message startswith TypeError => incompetent
FLAG_INCOMPETENT = True

OUT_DIR_LBL_PER_FILE = DATA_DIR.parent / "lbl_per_file"

MAX_WORKERS = 4
NDJSON_PATTERN = "*.ndjson"

# Outcomes to keep from pytest report
KEEP = {"error", "passed", "failed", "xfailed", "xpassed", "skipped"}


# -------------------------
# Labeling helpers
# -------------------------
HEX_TOKEN = "<HEX>"
RE_HEX_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b")
RE_HEX_ESCAPE = re.compile(r"\\x[0-9a-fA-F]{2}")

RE_OBJ_REPR_ID = re.compile(r"<[^>]*\bid='[0-9]+'[^>]*>")
RE_OBJ_HEX_AT = re.compile(r"<[^>]*\bobject at 0x[0-9a-fA-F]+\b[^>]*>")

RE_SANDBOX_ID = re.compile(r"sandbox-[A-Za-z0-9_-]+")

# Umbrella label for unexpected/anomalous run failures
ANOMALOUS_LABEL = "anomalous"
ANOMALOUS_PREFIXES = (
    "OSError",
    "TimeoutError",
    "Failed: Timeout",
    "Resource temporarily unavailable",
    "Too many open files",
    "Disk quota exceeded",
)


def normalize_noise(msg: str | None) -> str:
    if msg is None:
        return ""

    # Normalize runtime object repr noise
    s = RE_OBJ_REPR_ID.sub("<OBJ id='<ID>'>", msg)
    s = RE_OBJ_HEX_AT.sub("<OBJ at <HEX>>", s)

    # Normalize remaining hex noise
    s = RE_HEX_ADDR.sub(HEX_TOKEN, s)
    s = RE_HEX_ESCAPE.sub(HEX_TOKEN, s)

    # Normalize sandbox ids
    s = RE_SANDBOX_ID.sub("sandbox-<ID>", s)

    return s


def is_anomalous_message(msg: str) -> bool:
    m = (msg or "").lstrip()
    return m.startswith(ANOMALOUS_PREFIXES)


def compare_and_label(program_test: dict | None, mutant_test: dict, flag_incompetent: bool) -> str:
    if program_test is None:
        return "unknown"

    p_outcome = program_test.get("outcome")
    m_outcome = mutant_test.get("outcome")

    # Normalize messages once
    p_msg = normalize_noise(program_test.get("raw_short_msg"))
    m_msg = normalize_noise(mutant_test.get("raw_short_msg"))

    killed = (p_outcome != m_outcome) or (p_msg != m_msg)
    if not killed:
        return "survived"

    # umbrella anomalous classification (only when it would otherwise be "killed")
    # if is_anomalous_message(m_msg):
    #     return ANOMALOUS_LABEL

    if killed and (is_anomalous_message(m_msg) or is_anomalous_message(p_msg)): 
        return ANOMALOUS_LABEL
    

    # Existing: incompetent
    if flag_incompetent:
        m0 = (m_msg or "").lstrip()
        if m0.startswith("TypeError"):
            return "incompetent"

    return "killed"


def label_commit_data_inplace(commit_data: dict, flag_incompetent: bool) -> dict:
    """
    MUTATES and returns commit_data by adding mt["status"] = killed/survived/unknown/incompetent/anomalous
    for each mutant test, based on the program's corresponding test.
    """
    program_tests = (commit_data.get("program", {}) or {}).get("tests", {}) or {}
    mutants = commit_data.get("mutants", {}) or {}

    for _, md in mutants.items():
        mtests = (md.get("tests", {}) or {})
        for test_id, mt in mtests.items():
            pt = program_tests.get(test_id)
            mt["status"] = compare_and_label(pt, mt, flag_incompetent)

    return commit_data


# -------------------------
# Per-file split output helpers
# -------------------------
_RE_BAD_FS_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_mutant_key_for_filename(mutant_key: str) -> str:
    """
    "thefuck/shells/fish.py#0" -> "thefuck_shells_fish_py_0"
    """
    s = mutant_key.replace("/", "_").replace("\\", "_").replace("#", "_").replace(".", "_")
    s = _RE_BAD_FS_CHARS.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def write_lbl_per_file_outputs(
    *,
    data: dict,
    out_base_dir: Path,
    commit: str,
    source_stem: str,
) -> None:
    """
    Writes:
      out_base_dir/<commit>/program_<commit>.json OR pres_<commit>.json
      out_base_dir/<commit>/mutant_<sanitized_key>_<commit>.json for each mutant

    Works for both labeled and unlabeled data:
      - uses per-test "status" if present, otherwise falls back to raw "outcome".
    """
    commit_dir = out_base_dir / commit
    commit_dir.mkdir(parents=True, exist_ok=True)

    program = data.get("program", {}) or {}
    mutants = data.get("mutants", {}) or {}

    # Decide program filename based on input file stem
    if source_stem == "pres_fix":
        prog_name = f"pres_{commit}.json"
    else:
        prog_name = f"program_{commit}.json"

    # Program file content: list of {id, outcome, duration?}
    prog_tests = (program.get("tests", {}) or {})
    prog_list = []
    for tid, t in prog_tests.items():
        entry = {"id": tid, "outcome": t.get("outcome")}
        if "duration" in t and t["duration"] is not None:
            entry["duration"] = t["duration"]
        prog_list.append(entry)

    (commit_dir / prog_name).write_text(json.dumps(prog_list, indent=2, ensure_ascii=False), encoding="utf-8")

    # Mutant files
    for mutant_key, md in mutants.items():
        meta = (md.get("meta", {}) or {}).copy()
        meta.setdefault("mutant_key", mutant_key)
        meta["commit"] = commit
        meta["pytest_exit_code"] = md.get("exitstatus")
        meta["collection_ok"] = md.get("collection_ok")

        mtests = (md.get("tests", {}) or {})
        test_outcomes = {}
        for tid, t in mtests.items():
            test_outcomes[tid] = t.get("status") or t.get("outcome")
        meta["test_outcomes"] = test_outcomes

        safe_key = sanitize_mutant_key_for_filename(meta["mutant_key"])
        mname = f"mutant_{safe_key}_{commit}.json"
        (commit_dir / mname).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# -------------------------
# Extract one file (and optionally label)
# -------------------------
def process_one_file(
    file_path_str: str,
    out_extracted_str: str,
    out_labeled_str: str,
    out_lbl_per_file_str: str,
    enable_labeling: bool,
    write_intermediate: bool,
    flag_incompetent: bool,
    write_lbl_per_file: bool,
    write_labeled_data: bool,
) -> str:
    file_path = Path(file_path_str)
    out_extracted = Path(out_extracted_str)
    out_labeled = Path(out_labeled_str)
    out_lbl_per_file = Path(out_lbl_per_file_str)

    out_name = file_path.stem  # results_<commit> or pres_fix
    commit_data = {"program": {}, "mutants": {}}
    seen_commit = None

    # Read ndjson -> commit_data
    with file_path.open("r", encoding="utf-8") as f:
        reader = ndjson.reader(f)
        for obj in reader:
            session = obj.get("session", {}) or {}
            exitstatus = session.get("exitstatus")
            collection = obj.get("collection", {}) or {}
            meta = session.get("meta", {}) or {}

            commit = meta.get("commit")
            kind = meta.get("kind")
            if not commit or kind not in ("program", "mutant"):
                continue

            # sanity: one commit per file
            if seen_commit is None:
                seen_commit = commit
            elif commit != seen_commit:
                raise RuntimeError(f"Multiple commits in one file: {file_path}")

            if kind == "program":
                pd = commit_data["program"]
                pd["exitstatus"] = exitstatus
                pd["collection_ok"] = collection.get("ok")
                pd.setdefault("tests", {})
                pd["meta"] = {k: v for k, v in meta.items() if k != "kind"}
            else:
                mkey = meta.get("mutant_key")
                if not mkey:
                    continue
                md = commit_data["mutants"].setdefault(str(mkey), {})
                md["exitstatus"] = exitstatus
                md["collection_ok"] = collection.get("ok")
                md.setdefault("tests", {})
                md["meta"] = {k: v for k, v in meta.items() if k != "kind"}

            tests = obj.get("tests", {}) or {}
            for test_id, test in tests.items():
                outcome = test.get("outcome")
                if outcome in KEEP:
                    v = {
                        "outcome": outcome,
                        "raw_short_msg": test.get("message", None),
                        "duration": test.get("duration", None),
                    }
                    if kind == "program":
                        commit_data["program"]["tests"][test_id] = v
                    else:
                        commit_data["mutants"][str(meta["mutant_key"])]["tests"][test_id] = v

    # Validate switches
    if write_labeled_data and not enable_labeling:
        raise RuntimeError("WRITE_LABELED_DATA=True requires ENABLE_LABELING=True")

    outputs = []

    # 1) Intermediate bundle must ALWAYS be WITHOUT status -> write commit_data (raw) only
    if write_intermediate:
        extracted_path = out_extracted / f"{out_name}.json"
        with extracted_path.open("w", encoding="utf-8") as out:
            json.dump(commit_data, out, indent=2, ensure_ascii=False)
        outputs.append("intermediate_bundle")

    # 2) If labeling enabled, compute labels on a DEEP COPY so commit_data stays raw
    if enable_labeling:
        labeled_data = label_commit_data_inplace(copy.deepcopy(commit_data), flag_incompetent)

        if write_labeled_data:
            labeled_path = out_labeled / f"{out_name}.json"
            with labeled_path.open("w", encoding="utf-8") as out:
                json.dump(labeled_data, out, indent=2, ensure_ascii=False)
            outputs.append("labeled_bundle")

        if write_lbl_per_file:
            if seen_commit is None:
                raise RuntimeError(f"Could not determine commit for file: {file_path}")
            write_lbl_per_file_outputs(
                data=labeled_data,
                out_base_dir=out_lbl_per_file,
                commit=seen_commit,
                source_stem=out_name,
            )
            outputs.append("lbl_per_file")

    # 3) If labeling disabled, per-file output uses raw outcomes
    else:
        if write_lbl_per_file:
            if seen_commit is None:
                raise RuntimeError(f"Could not determine commit for file: {file_path}")
            write_lbl_per_file_outputs(
                data=commit_data,
                out_base_dir=out_lbl_per_file,
                commit=seen_commit,
                source_stem=out_name,
            )
            outputs.append("lbl_per_file")

    if not outputs:
        outputs.append("nothing (all write flags off)")

    return f"{out_name}.json (wrote: {', '.join(outputs)})"


def main():
    files = sorted(DATA_DIR.glob(NDJSON_PATTERN))
    if not files:
        raise SystemExit(f"No ndjson files matched {NDJSON_PATTERN} in: {DATA_DIR}")

    # Create output dirs once (only if needed)
    if WRITE_INTERMEDIATE:
        OUT_DIR_EXTRACTED.mkdir(parents=True, exist_ok=True)

    if ENABLE_LABELING and WRITE_LABELED_DATA:
        OUT_DIR_LABELED.mkdir(parents=True, exist_ok=True)

    if WRITE_LBL_PER_FILE:
        OUT_DIR_LBL_PER_FILE.mkdir(parents=True, exist_ok=True)

    max_workers = MAX_WORKERS

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_one_file,
                str(p),
                str(OUT_DIR_EXTRACTED),
                str(OUT_DIR_LABELED),
                str(OUT_DIR_LBL_PER_FILE),
                ENABLE_LABELING,
                WRITE_INTERMEDIATE,
                FLAG_INCOMPETENT,
                WRITE_LBL_PER_FILE,
                WRITE_LABELED_DATA,
            )
            for p in files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            name = fut.result()  # raises if worker crashed
            tqdm.write(f"✔ wrote {name}")


if __name__ == "__main__":
    main()
