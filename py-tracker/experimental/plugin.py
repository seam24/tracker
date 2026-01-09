# src/pytest_mutation_runner/plugin.py

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
import pytest
from typing import Any, Dict, Optional

MODE = "--mode"
PROG_RES = "--program-results"
MUT_RES = "--mutant-results"
STOP_ON_KILL = "--stop-on-kill"
TIMEOUT_MIN = "--timeout-min"
TIMEOUT_MUL = "--timeout-multiplier"
INCOMP_DISABLE = "--disable-incompetent"
DROP_MISSING = "--drop-missing-tests"

# Similarity threshold for "same failure reason".
# 1.0 means exact match; lower means more tolerant.
_REASON_SIM_THRESHOLD = 0.90


def _extract_failure_reason(report) -> Optional[Dict[str, Any]]:
    """
    Return a structured failure reason (or None) from a pytest report.
    Stores a concise crash message (when available) and a fuller longrepr text.
    """
    try:
        lr = getattr(report, "longrepr", None)
    except Exception:
        lr = None

    if lr is None:
        return None

    crash_msg = None
    try:
        reprcrash = getattr(lr, "reprcrash", None)
        if reprcrash is not None:
            crash_msg = getattr(reprcrash, "message", None)
    except Exception:
        crash_msg = None

    long_text = None
    try:
        long_text = getattr(report, "longreprtext", None)
    except Exception:
        long_text = None

    if not long_text:
        try:
            long_text = str(lr)
        except Exception:
            long_text = None

    if isinstance(crash_msg, str) and crash_msg.strip() == "":
        crash_msg = None
    if isinstance(long_text, str) and long_text.strip() == "":
        long_text = None

    if crash_msg is None and long_text is None:
        return None

    return {
        "message": crash_msg,
        "longrepr": long_text,
    }


def _normalize_reason_text(text: Optional[str]) -> str:
    """
    Normalize failure text to make comparisons robust:
    - remove Windows/Unix absolute paths
    - remove line/column numbers
    - remove hex addresses / object ids
    - collapse whitespace
    """
    if not text:
        return ""

    s = text

    # Normalize slashes a bit
    s = s.replace("\\", "/")

    # Remove common hex addresses: 0x7f....
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)

    # Remove pytest "at 0x..." object address patterns (already covered, but keep)
    s = re.sub(r"\bat\s+0xADDR\b", "at 0xADDR", s)

    # Replace /tmp/... and similar temp dirs with placeholder
    s = re.sub(r"/tmp/[^ \n\t\r\"']+", "/tmp/PATH", s)
    s = re.sub(r"/var/folders/[^ \n\t\r\"']+", "/var/folders/PATH", s)

    # Replace absolute Unix paths like /home/user/... or /mnt/c/...
    s = re.sub(r"/(home|mnt|Users|usr|opt|private)/[^ \n\t\r\"']+", "/ABS/PATH", s, flags=re.IGNORECASE)

    # Replace Windows drive paths like C:/Users/...
    s = re.sub(r"\b[A-Za-z]:/[^ \n\t\r\"']+", "DRIVE:/PATH", s)

    # Remove common file:line:col patterns
    s = re.sub(r"(:\d+)(:\d+)?", ":N", s)

    # Remove "line 123" / "File ..., line 123"
    s = re.sub(r"\bline\s+\d+\b", "line N", s, flags=re.IGNORECASE)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _reason_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _same_failure_reason(
    program_reason: Optional[Dict[str, Any]],
    mutant_reason: Optional[Dict[str, Any]],
    threshold: float = _REASON_SIM_THRESHOLD,
) -> bool:
    """
    Compare two failure_reason dicts using normalization + similarity.
    Strategy:
      1) Compare normalized 'message' similarity (preferred).
      2) If either message missing, fall back to normalized 'longrepr'.
      3) If both present but messages are too different, optionally check longrepr too.
    """
    prog_msg = _normalize_reason_text((program_reason or {}).get("message"))
    mut_msg = _normalize_reason_text((mutant_reason or {}).get("message"))

    if prog_msg or mut_msg:
        if _reason_similarity(prog_msg, mut_msg) >= threshold:
            return True
        # If messages exist but are "too different", still give longrepr a chance
        prog_lr = _normalize_reason_text((program_reason or {}).get("longrepr"))
        mut_lr = _normalize_reason_text((mutant_reason or {}).get("longrepr"))
        if prog_lr or mut_lr:
            return _reason_similarity(prog_lr, mut_lr) >= threshold
        return False

    # No messages available, compare longrepr
    prog_lr = _normalize_reason_text((program_reason or {}).get("longrepr"))
    mut_lr = _normalize_reason_text((mutant_reason or {}).get("longrepr"))
    return _reason_similarity(prog_lr, mut_lr) >= threshold


class ProgramMode:
    def __init__(self):
        self.config = None
        self.results = []

    def pytest_configure(self, config):
        self.config = config

    def pytest_collection_modifyitems(self, session, config, items):
        # apply a simple min-timeout to all tests, but only if > 0
        try:
            m = float(config.getoption(TIMEOUT_MIN))
        except (TypeError, ValueError):
            m = 0.0

        if m <= 0:
            return  # user implicitly disabled timeouts for program run

        for item in items:
            item.add_marker(pytest.mark.timeout(m))

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return

        failure_reason = None
        if report.outcome != "passed":
            failure_reason = _extract_failure_reason(report)

        self.results.append(
            {
                "id": report.nodeid,
                "outcome": report.outcome,
                "duration": getattr(report, "duration", None),
                "failure_reason": failure_reason,
            }
        )

    def pytest_sessionfinish(self, session, exitstatus):
        results_path = Path(self.config.getoption(PROG_RES))
        results_path.write_text(json.dumps(self.results, indent=2))
        print(f"\n Wrote results to {results_path.resolve()}")


class MutantMode:
    def __init__(self):
        self.config = None
        self.reference_data = None
        self.mutant_result = {}
        self.mutant_results_path = None
        self.disable_incompetent = False
        self.mutant_killed = False

    def pytest_configure(self, config):
        self.config = config
        self.disable_incompetent = config.getoption(INCOMP_DISABLE)

        results_path = config.getoption(PROG_RES)
        path = Path(results_path)
        if not path.exists():
            raise pytest.UsageError(f"--program-results '{results_path}' not found for mutant mode")

        # load reference results
        data = json.loads(path.read_text())
        ref_map = {
            entry["id"]: {
                "outcome": entry.get("outcome"),
                "duration": entry.get("duration"),
                "failure_reason": entry.get("failure_reason"),
            }
            for entry in data
            if "id" in entry
        }
        self.reference_data = ref_map

        # richer per-test mutant record, while keeping "pending/survived/killed" semantics explicit
        self.mutant_result = {}
        for entry in data:
            tid = entry.get("id")
            if not tid:
                continue
            base = ref_map.get(tid, {})
            self.mutant_result[tid] = {
                "status": "pending",
                "program_outcome": base.get("outcome"),
                "program_failure_reason": base.get("failure_reason"),
                "mutant_outcome": None,
                "mutant_failure_reason": None,
            }

        # require explicit --mutant-results
        mut_path_opt = config.getoption(MUT_RES)
        if not mut_path_opt:
            raise pytest.UsageError("--mutant-results must be provided in mutant mode")
        self.mutant_results_path = Path(mut_path_opt)

    def pytest_collection_modifyitems(self, session, config, items):
        def _get_float(opt, default=0.0):
            try:
                return float(config.getoption(opt))
            except (TypeError, ValueError):
                return default

        tmin = _get_float(TIMEOUT_MIN)
        tmul = _get_float(TIMEOUT_MUL)

        # both off → no timeouts in mutant run
        if tmin <= 0 and tmul <= 0:
            return

        for item in items:
            base = self.reference_data.get(item.nodeid)
            if not base:
                continue

            dur = base.get("duration")

            # try to use per-test duration if it's a positive number
            try:
                dur_f = float(dur)
            except (TypeError, ValueError):
                dur_f = 0.0

            if dur_f > 0:
                timeout = max(tmin, tmul * dur_f)
            else:
                timeout = tmin  # fall back to flat min if available

            if timeout and timeout > 0:
                item.add_marker(pytest.mark.timeout(timeout))

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return

        nodeid = report.nodeid
        if nodeid not in self.reference_data:
            return

        rec = self.mutant_result.get(nodeid)
        if not isinstance(rec, dict):
            rec = {"status": "pending"}
            self.mutant_result[nodeid] = rec

        current_status = rec.get("status", "pending")

        # Always record what happened in mutant run
        rec["mutant_outcome"] = report.outcome
        rec["mutant_failure_reason"] = _extract_failure_reason(report) if report.outcome != "passed" else None

        # Detect incompetent first (unless disabled)
        if (
            not self.disable_incompetent
            and current_status in ("pending", "survived")
            and self._is_incompetent(report)
        ):
            rec["status"] = "incompetent"
            return

        orig_outcome = self.reference_data[nodeid].get("outcome")
        orig_reason = self.reference_data[nodeid].get("failure_reason")

        mutant_outcome = report.outcome
        mutant_reason = rec.get("mutant_failure_reason")

        killed = False
        killed_why = None

        # 1) Outcome mismatch => killed
        if orig_outcome is not None and mutant_outcome != orig_outcome:
            killed = True
            killed_why = f"outcome mismatch (expected {orig_outcome}, got {mutant_outcome})"

        # 2) Same outcome, both non-passed, but reasoning differs (by similarity) => killed
        elif orig_outcome is not None and mutant_outcome == orig_outcome and mutant_outcome != "passed":
            if not _same_failure_reason(orig_reason, mutant_reason):
                killed = True
                killed_why = f"failure reason mismatch (<{_REASON_SIM_THRESHOLD:.2f} similarity)"

        if killed:
            rec["status"] = "killed"
            self.mutant_killed = True
            print(f"\n Mutant killed by {nodeid} ({killed_why})")

            if self.config.getoption(STOP_ON_KILL):
                pytest.exit("Mutant killed", returncode=0)
        else:
            # Test ran and did NOT kill the mutant → survived (unless already incompetent/killed)
            if current_status == "pending":
                rec["status"] = "survived"

    def _is_incompetent(self, report):
        # According to: https://github.com/mutpy/mutpy/blob/master/mutpy/test_runners/pytest_runner.py#L29
        return (
            report.longrepr is not None
            and hasattr(report.longrepr, "reprcrash")
            and "TypeError" in report.longrepr.reprcrash.message
        )

    def _is_timeout(self, report):
        return (
            report.longrepr is not None
            and hasattr(report.longrepr, "reprcrash")
            and "timeout" in report.longrepr.reprcrash.message.lower()
        )

    def pytest_sessionfinish(self, session, exitstatus):
        self.mutant_results_path.write_text(json.dumps(self.mutant_result, indent=2))
        print(f"\n Wrote mutant results to {self.mutant_results_path.resolve()}")


def pytest_addoption(parser):
    parser.addoption(
        PROG_RES,
        action="store",
        required=False,
        default=None,
        help="Path to JSON file with per-test results (read or write depending on --mode).",
    )
    parser.addoption(
        MUT_RES,
        action="store",
        default=None,
        help=(
            "Path to JSON file where mutant execution results will be written "
            "(only valid and required in --mode=mutant)."
        ),
    )
    parser.addoption(
        MODE,
        choices=["program", "mutant"],
        required=False,
        default=None,
        help="Execution mode: 'program' to record original results, 'mutant' to compare outcomes.",
    )
    parser.addoption(
        STOP_ON_KILL,
        action="store_true",
        help="Stop immediately when a mutant is killed (only valid in mutant mode).",
    )

    parser.addoption(
        TIMEOUT_MIN,
        action="store",
        default="2.0",
        help=(
            "Minimum timeout in seconds. "
            "Program mode: applied to every test if > 0. "
            "Mutant mode: used as a lower bound for per-test timeouts. "
            "Set to 0 or negative to disable in that mode."
        ),
    )
    parser.addoption(
        TIMEOUT_MUL,
        action="store",
        default="3.0",
        help=(
            "Mutant mode only: multiply each test's baseline duration by this value "
            "to derive a per-test timeout. Final timeout is max(--timeout-min, multiplier * duration). "
            "Set to 0 (and --timeout-min <= 0) to disable timeouts."
        ),
    )
    parser.addoption(
        INCOMP_DISABLE,
        action="store_true",
        help=(
            "Disable incompetent mutant labeling. "
            "By default, mutants that cause TypeErrors are labeled as 'incompetent'. "
            "Use this flag to skip that classification."
        ),
    )
    parser.addoption(
        DROP_MISSING,
        action="store_true",
        help="Ignore missing test path arguments (files/dirs) instead of erroring.",
    )


def pytest_configure(config):
    mode = config.getoption(MODE)

    # If no mode is given, completely disable this plugin’s functionality
    if not mode:
        return

    # If mode IS set, enforce that --program-results is provided
    prog_res = config.getoption(PROG_RES)
    if not prog_res:
        raise pytest.UsageError(
            f"{PROG_RES} must be provided when {MODE} is set "
            "(e.g., --mode=program --program-results=path.json)"
        )

    if mode == "mutant":
        plugin = MutantMode()
    elif mode == "program":
        plugin = ProgramMode()
    else:
        raise pytest.UsageError("Unknown --mode value")

    config.pluginmanager.register(plugin, "mutation_plugin")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection(session):
    """
    If pytest fails during collection with a UsageError, and --drop-missing-tests
    is enabled, drop missing filesystem paths (files/dirs) from the positional
    selectors and retry collection.

    Note: explicit nodeid selectors containing '::' are ignored entirely (we only
    support file/dir selectors here).
    """
    outcome = yield
    config = session.config

    # Only act when plugin mode is enabled and user opted in.
    if not config.getoption(MODE) or not config.getoption(DROP_MISSING):
        return

    excinfo = outcome.excinfo
    if not excinfo or excinfo[0] is not pytest.UsageError:
        return

    orig = list(getattr(config, "args", []) or [])
    if not orig:
        return

    kept = []
    for a in orig:
        # We don't support explicit nodeid selectors; drop them to avoid collection errors.
        if "::" in a:
            continue

        p = Path(a)
        if p.exists():
            kept.append(a)

    # Only handle if we actually dropped something
    if kept == orig:
        return

    print("\n[mutation-runner] dropped test selectors:")
    for a in orig:
        if a not in kept:
            print("  ", a)

    config.args = kept

    # Retry collection (if kept is empty, fall back to normal discovery)
    if kept:
        session.perform_collect(args=kept, genitems=True)
    else:
        session.perform_collect(genitems=True)

    outcome.force_result(None)
