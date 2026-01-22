# pytjson/plugin.py
# mypy: allow-untyped-defs
"""
A minimal "JUnit-XML equivalent, but JSON" reporter for pytest.

Compatibility goals:
- Works across a wide range of pytest + pluggy versions (incl. older ones)
- Works on Python 3.7+ (no PEP604 unions, no runtime-evaluated typing that breaks)
- Avoids hook signature mismatches by using *args/**kwargs where pytest changed hooks
- Avoids hard dependencies on internal pytest modules that moved over time

CLI:
  pytest --junit-json=report.json

Optional:
  --junit-prefix=STR             (reuses pytest's built-in option; we DO NOT re-add it)
  --junit-json-meta k=v          (repeatable; values can be typed with basic heuristics)
  --junit-json-ndjson            (write as NDJSON: one JSON object per line)
  --junit-json-append            (append to existing file; intended for NDJSON)

INI (reuses junitxml names for familiarity):
  junit_suite_name = pytest
  junit_logging = no|log|system-out|system-err|out-err|all
  junit_log_passing_tests = true|false
  junit_duration_report = total|call
  junit_family = legacy|xunit1|xunit2   (default xunit2)
"""

from __future__ import absolute_import, division, print_function

import datetime
import functools
import json
import os
import platform
import re
import time
from dataclasses import dataclass

# typing: keep safe for py3.7 + avoid runtime issues
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

# -----------------------------------------------------------------------------
# Best-effort imports from pytest internals (these moved across versions).
# We always provide fallbacks so the plugin still loads on older pytest.
# -----------------------------------------------------------------------------
try:
    from _pytest.config import Config  # type: ignore
except Exception:  # pragma: no cover
    Config = Any  # type: ignore

try:
    from _pytest.config.argparsing import Parser  # type: ignore
except Exception:  # pragma: no cover
    Parser = Any  # type: ignore

try:
    # filename_arg is a helper used by pytest's junitxml plugin; might not exist.
    from _pytest.config import filename_arg  # type: ignore
except Exception:  # pragma: no cover
    filename_arg = None  # type: ignore

try:
    from _pytest.reports import TestReport  # type: ignore
except Exception:  # pragma: no cover
    TestReport = Any  # type: ignore

try:
    from _pytest.fixtures import FixtureRequest  # type: ignore
except Exception:  # pragma: no cover
    FixtureRequest = Any  # type: ignore

try:
    from _pytest.terminal import TerminalReporter  # type: ignore
except Exception:  # pragma: no cover
    TerminalReporter = Any  # type: ignore

# ExceptionRepr/ReprFileLocation moved around; treat as Any if unavailable.
try:
    from _pytest._code.code import ExceptionRepr, ReprFileLocation  # type: ignore
except Exception:  # pragma: no cover
    ExceptionRepr = Any  # type: ignore
    ReprFileLocation = Any  # type: ignore

# -----------------------------------------------------------------------------
# Compat: nodes.SEP moved around; fall back to "/" so we can replace with "."
# -----------------------------------------------------------------------------
try:
    from _pytest import nodes as _nodes  # type: ignore

    _NODE_SEP = getattr(_nodes, "SEP", "/")
except Exception:
    _NODE_SEP = "/"


# -----------------------------------------------------------------------------
# Compat: timing.Instant exists only on newer pytest
# -----------------------------------------------------------------------------
class _CompatInstant(object):
    def __init__(self) -> None:
        self._start = time.time()

    def elapsed(self) -> Any:
        class _Elapsed(object):
            seconds = time.time() - self._start

        return _Elapsed()

    def as_utc(self) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(self._start)


try:
    from _pytest import timing as _timing  # type: ignore
except Exception:
    _timing = None  # type: ignore


def _new_instant() -> Any:
    if _timing is not None and hasattr(_timing, "Instant"):
        try:
            return _timing.Instant()  # type: ignore[attr-defined]
        except Exception:
            return _CompatInstant()
    return _CompatInstant()


# -----------------------------------------------------------------------------
# Compat: stash only exists on newer pytest. Use it if available, else config attr.
# -----------------------------------------------------------------------------
json_key = None
try:
    from _pytest.stash import StashKey  # type: ignore

    json_key = StashKey("LogJUnitJSON")  # older typing-friendly usage
except Exception:
    StashKey = None  # type: ignore
    json_key = None

_ACTIVE = None  # type: Optional["LogJUnitJSON"]


def _set_active(config: Any, plugin: "LogJUnitJSON") -> None:
    global _ACTIVE
    _ACTIVE = plugin

    # prefer stash (pytest>=7), but don't assume it's present
    try:
        if hasattr(config, "stash") and json_key is not None:
            config.stash[json_key] = plugin
            return
    except Exception:
        pass

    # fallback: attach to config
    try:
        setattr(config, "_junitjson_plugin", plugin)
    except Exception:
        pass


def _get_active_from_config(config: Any) -> Optional["LogJUnitJSON"]:
    try:
        if hasattr(config, "stash") and json_key is not None:
            return config.stash.get(json_key, None)
    except Exception:
        pass
    try:
        return getattr(config, "_junitjson_plugin", None)
    except Exception:
        return None


def _get_active() -> Optional["LogJUnitJSON"]:
    return _ACTIVE


def _del_active(config: Any) -> None:
    global _ACTIVE
    try:
        if hasattr(config, "stash") and json_key is not None:
            try:
                del config.stash[json_key]
            except Exception:
                pass
    except Exception:
        pass

    try:
        if hasattr(config, "_junitjson_plugin"):
            try:
                delattr(config, "_junitjson_plugin")
            except Exception:
                pass
    except Exception:
        pass

    _ACTIVE = None


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def bin_text_escape(arg: object) -> str:
    """
    JSON doesn't need XML escaping, but reports can include odd control chars.
    Keep junitxml's "visual escape" behavior for stability/readability.
    """

    def repl(matchobj: Any) -> str:
        i = ord(matchobj.group())
        if i <= 0xFF:
            return "#x%02X" % i
        return "#x%04X" % i

    illegal_re = (
        "[^\u0009\u000a\u000d\u0020-\u007e\u0080-\ud7ff\ue000-\ufffd\u10000-\u10ffff]"
    )
    return re.sub(illegal_re, repl, str(arg))


def mangle_test_address(address: str) -> List[str]:
    path, possible_open_bracket, params = address.partition("[")
    names = path.split("::")
    names[0] = names[0].replace(_NODE_SEP, ".")
    names[0] = re.sub(r"\.py$", "", names[0])
    names[-1] += possible_open_bracket + params
    return names


def _check_record_param_type(param: str, v: Any) -> None:
    __tracebackhide__ = True
    if not isinstance(v, str):
        raise TypeError(
            "%s parameter needs to be a string, but %s given" % (param, type(v).__name__)
        )


def _check_meta_value(key: str, value: Any) -> None:
    __tracebackhide__ = True
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise TypeError(
            "meta value for %r must be str|int|float|bool|None, got %s"
            % (key, type(value).__name__)
        )


def _parse_meta_scalar(v: str) -> Any:
    s = v.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None

    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s, 10)
        except Exception:
            return s

    if re.fullmatch(r"[+-]?(\d+\.\d*|\d*\.\d+)([eE][+-]?\d+)?", s) or re.fullmatch(
        r"[+-]?\d+[eE][+-]?\d+", s
    ):
        try:
            return float(s)
        except Exception:
            return s

    return s


def _safe_location(report: Any) -> Tuple[Optional[str], Optional[int]]:
    # report.location is usually (path, lineno, domain) but not guaranteed
    try:
        loc = getattr(report, "location", None)
        if not loc:
            return (None, None)
        file_ = loc[0] if len(loc) > 0 else None
        line_ = loc[1] if len(loc) > 1 else None
        if line_ is not None:
            try:
                line_ = int(line_)
            except Exception:
                line_ = None
        return (file_, line_)
    except Exception:
        return (None, None)


def _safe_longrepr_text(longrepr: Any) -> str:
    # longrepr sometimes is a string, sometimes rich object
    try:
        return bin_text_escape(longrepr)
    except Exception:
        try:
            return bin_text_escape(str(longrepr))
        except Exception:
            return ""


def _reprcrash_message(longrepr: Any) -> Optional[str]:
    # longrepr.reprcrash.message exists on many versions, but guard it.
    try:
        reprcrash = getattr(longrepr, "reprcrash", None)
        if reprcrash is None:
            return None
        msg = getattr(reprcrash, "message", None)
        if msg is None:
            return None
        return str(msg)
    except Exception:
        return None


def _get_config_verbosity(config: Any) -> int:
    """
    Compat: verbosity moved around over time.
    - Newer: config.get_verbosity()
    - Older: config.option.verbose (int)
    """
    try:
        getv = getattr(config, "get_verbosity", None)
        if callable(getv):
            try:
                return int(getv())
            except Exception:
                pass
    except Exception:
        pass

    try:
        opt = getattr(config, "option", None)
        v = getattr(opt, "verbose", 0) if opt is not None else 0
        return int(v or 0)
    except Exception:
        return 0


def _safe_open_file(path: str, append: bool) -> Any:
    mode = "a" if append else "w"
    # Ensure directory exists
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        try:
            os.makedirs(dirname, exist_ok=True)  # py3.7 ok
        except TypeError:
            # py<3.2 compatibility not needed, but keep defensive.
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
    return open(path, mode, encoding="utf-8")


# -----------------------------------------------------------------------------
# Public fixtures (work with stash or attribute)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def record_junitjson_testsuite_property(request: FixtureRequest) -> Callable[[str, object], None]:
    __tracebackhide__ = True

    def noop(name: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("name", name)

    plugin = _get_active_from_config(getattr(request, "config", None))
    if plugin is None:
        return noop

    def record(name: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("name", name)
        plugin.add_global_property(name, value)

    return record


@pytest.fixture(scope="session")
def record_junitjson_meta(request: FixtureRequest) -> Callable[[str, object], None]:
    __tracebackhide__ = True

    def noop(key: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("key", key)

    plugin = _get_active_from_config(getattr(request, "config", None))
    if plugin is None:
        return noop

    def record(key: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("key", key)
        plugin.add_meta(key, value)

    return record


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass
class _CaseData:
    classname: str
    name: str
    file: Optional[str] = None
    line: Optional[int] = None
    url: Optional[str] = None
    time: float = 0.0
    properties: Optional[List[Dict[str, str]]] = None
    skipped: Optional[Dict[str, str]] = None
    failure: Optional[Dict[str, str]] = None
    error: Optional[Dict[str, str]] = None
    system_out: Optional[str] = None
    system_err: Optional[str] = None

    def __post_init__(self) -> None:
        if self.properties is None:
            self.properties = []


# -----------------------------------------------------------------------------
# Core reporter implementation (NO hook names on the class)
# -----------------------------------------------------------------------------
class LogJUnitJSON(object):
    def __init__(
        self,
        logfile: str,
        prefix: Optional[str],
        suite_name: str,
        logging_mode: str,
        report_duration: str,
        family: str,
        log_passing_tests: bool,
        meta_kv: List[Tuple[str, object]],
        ndjson: bool,
        append: bool,
    ) -> None:
        logfile = os.path.expanduser(os.path.expandvars(logfile))
        self.logfile = os.path.normpath(os.path.abspath(logfile))
        self.prefix = prefix
        self.suite_name = suite_name
        self.logging = logging_mode
        self.log_passing_tests = bool(log_passing_tests)
        self.report_duration = report_duration

        self.family = "xunit1" if family == "legacy" else family
        self.ndjson = bool(ndjson)
        self.append = bool(append)

        self.stats = dict.fromkeys(["error", "passed", "failure", "skipped"], 0)  # type: ignore
        self.global_properties = []  # type: List[Tuple[str, str]]

        self.meta = {}  # type: Dict[str, Any]
        for k, v in meta_kv:
            self.add_meta(k, v)

        self._cases_order = []  # type: List[Tuple[str, Optional[str], Optional[int]]]
        self._cases = {}  # type: Dict[Tuple[str, Optional[str], Optional[int]], _CaseData]

        self._open_failed_calls = []  # type: List[Any]
        self._cnt_double_fail_tests = 0

        self._suite_start = None  # type: Any

    # ---- internal API used by module-level hooks ----

    def on_sessionstart(self) -> None:
        self._suite_start = _new_instant()

    def add_global_property(self, name: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("name", name)
        self.global_properties.append((str(name), bin_text_escape(value)))

    def add_meta(self, key: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("key", key)
        _check_meta_value(key, value)
        if isinstance(value, str):
            self.meta[key] = bin_text_escape(value)
        else:
            self.meta[key] = value

    def add_stats(self, key: str) -> None:
        if key in self.stats:
            self.stats[key] += 1

    def _case_key(self, report_or_nodeid: Any) -> Tuple[str, Optional[str], Optional[int]]:
        nodeid = getattr(report_or_nodeid, "nodeid", report_or_nodeid)
        worker_id = getattr(report_or_nodeid, "worker_id", None)
        item_index = getattr(report_or_nodeid, "item_index", None)

        wid = str(worker_id) if worker_id is not None else None
        ii = None
        if item_index is not None:
            try:
                ii = int(item_index)
            except Exception:
                ii = None

        return (str(nodeid), wid, ii)

    def _get_or_create_case(self, report: Any) -> _CaseData:
        key = self._case_key(report)
        existing = self._cases.get(key)
        if existing is not None:
            return existing

        names = mangle_test_address(getattr(report, "nodeid", str(key[0])))
        classnames = names[:-1]
        if self.prefix:
            classnames.insert(0, self.prefix)

        classname = ".".join(classnames)
        name = bin_text_escape(names[-1])

        file_, line_ = _safe_location(report)
        url = getattr(report, "url", None) if hasattr(report, "url") else None

        case = _CaseData(classname=classname, name=name, file=file_, line=line_, url=url)
        self._cases[key] = case
        self._cases_order.append(key)
        return case

    def _append_properties_from_report(self, case: _CaseData, report: Any) -> None:
        try:
            props = getattr(report, "user_properties", None) or []
            for k, v in props:
                case.properties.append({"name": str(k), "value": bin_text_escape(v)})
        except Exception:
            pass

    def _prepare_content(self, content: str, header: str) -> str:
        return "\n".join([header.center(80, "-"), content, ""])

    def _write_captured_output(self, case: _CaseData, report: Any) -> None:
        try:
            if (not self.log_passing_tests) and getattr(report, "passed", False):
                return
        except Exception:
            pass

        if self.logging == "no":
            return

        content_out = getattr(report, "capstdout", "") or ""
        content_err = getattr(report, "capstderr", "") or ""
        content_log = getattr(report, "caplog", "") or ""

        out_buf = ""
        err_buf = ""

        if self.logging in ("log", "all"):
            out_buf += self._prepare_content(content_log, " Captured Log ")
        if self.logging in ("system-out", "out-err", "all"):
            out_buf += self._prepare_content(content_out, " Captured Out ")
        if self.logging in ("system-err", "out-err", "all"):
            err_buf += self._prepare_content(content_err, " Captured Err ")

        if out_buf:
            case.system_out = (case.system_out or "") + bin_text_escape(out_buf)
        if err_buf:
            case.system_err = (case.system_err or "") + bin_text_escape(err_buf)

    def update_testcase_duration(self, report: Any) -> None:
        try:
            when = getattr(report, "when", "")
            if self.report_duration in ("total", when):
                case = self._get_or_create_case(report)
                case.time += float(getattr(report, "duration", 0.0) or 0.0)
        except Exception:
            pass

    def on_collectreport(self, report: Any) -> None:
        try:
            if getattr(report, "passed", False):
                return
        except Exception:
            return

        case = self._get_or_create_case(report)
        self.update_testcase_duration(report)

        try:
            if getattr(report, "failed", False):
                self.add_stats("error")
                case.error = {
                    "message": "collection failure",
                    "text": _safe_longrepr_text(getattr(report, "longrepr", "")),
                }
            else:
                self.add_stats("skipped")
                case.skipped = {
                    "message": "collection skipped",
                    "text": _safe_longrepr_text(getattr(report, "longrepr", "")),
                    "type": "pytest.skip",
                }
        except Exception:
            # If anything goes wrong here, do not crash pytest.
            pass

    def on_internalerror(self, excrepr: Any) -> None:
        key = ("internal", None, None)
        if key not in self._cases:
            case = _CaseData(classname="pytest", name="internal")
            self._cases[key] = case
            self._cases_order.append(key)
        case = self._cases[key]
        self.add_stats("error")
        try:
            case.error = {"message": "internal error", "text": bin_text_escape(excrepr)}
        except Exception:
            case.error = {"message": "internal error", "text": ""}

    def on_runtest_logreport(self, report: Any) -> None:
        try:
            when = getattr(report, "when", "")
            passed = getattr(report, "passed", False)
            failed = getattr(report, "failed", False)
            skipped = getattr(report, "skipped", False)
        except Exception:
            return

        if passed:
            if when == "call":
                self.add_stats("passed")
                case = self._get_or_create_case(report)
                self._append_properties_from_report(case, report)

        elif failed:
            # Handle "double-fail" (call failed + teardown failed)
            close_report = None
            if when == "teardown":
                try:
                    report_wid = getattr(report, "worker_id", None)
                    report_ii = getattr(report, "item_index", None)
                except Exception:
                    report_wid, report_ii = None, None

                for rep in list(self._open_failed_calls):
                    try:
                        if rep.nodeid == report.nodeid and getattr(rep, "item_index", None) == report_ii and getattr(rep, "worker_id", None) == report_wid:
                            close_report = rep
                            break
                    except Exception:
                        continue

                if close_report is not None:
                    self._cnt_double_fail_tests += 1

            case = self._get_or_create_case(report)
            self._append_properties_from_report(case, report)

            if when == "call":
                self.add_stats("failure")
                if hasattr(report, "wasxfail"):
                    # xfail that failed? keep consistent with junitxml-ish semantics
                    self.add_stats("skipped")
                    case.skipped = {
                        "message": "xfail-marked test passes unexpectedly",
                        "type": "pytest.xfail",
                    }
                else:
                    longrepr = getattr(report, "longrepr", None)
                    msg = _reprcrash_message(longrepr) or str(longrepr)
                    case.failure = {
                        "message": bin_text_escape(msg),
                        "text": _safe_longrepr_text(longrepr),
                    }
                self._open_failed_calls.append(report)

                # Capture output for failing tests even if we don't log passing tests
                if not self.log_passing_tests:
                    self._write_captured_output(case, report)
            else:
                # setup/teardown failure
                self.add_stats("error")
                longrepr = getattr(report, "longrepr", None)
                reason = _reprcrash_message(longrepr) or str(longrepr)
                if when == "teardown":
                    msg = 'failed on teardown with "%s"' % reason
                else:
                    msg = 'failed on setup with "%s"' % reason
                case.error = {
                    "message": bin_text_escape(msg),
                    "text": _safe_longrepr_text(longrepr),
                }

        elif skipped:
            self.add_stats("skipped")
            case = self._get_or_create_case(report)
            self._append_properties_from_report(case, report)

            if hasattr(report, "wasxfail"):
                xfailreason = str(report.wasxfail)
                if xfailreason.startswith("reason: "):
                    xfailreason = xfailreason[8:]
                case.skipped = {
                    "type": "pytest.xfail",
                    "message": bin_text_escape(xfailreason),
                }
            else:
                longrepr = getattr(report, "longrepr", None)
                skipreason = ""
                details = ""
                try:
                    filename, lineno, reason = longrepr
                    skipreason = str(reason)
                    if skipreason.startswith("Skipped: "):
                        skipreason = skipreason[9:]
                    details = "%s:%s: %s" % (filename, lineno, skipreason)
                except Exception:
                    skipreason = str(longrepr)
                    details = skipreason

                case.skipped = {
                    "type": "pytest.skip",
                    "message": bin_text_escape(skipreason),
                    "text": bin_text_escape(details),
                }
                self._write_captured_output(case, report)

        # Duration + captured output at teardown
        self.update_testcase_duration(report)
        if when == "teardown":
            case = self._get_or_create_case(report)
            self._write_captured_output(case, report)

            # close open failed call record
            try:
                report_wid = getattr(report, "worker_id", None)
                report_ii = getattr(report, "item_index", None)
                for rep in list(self._open_failed_calls):
                    if rep.nodeid == report.nodeid and getattr(rep, "item_index", None) == report_ii and getattr(rep, "worker_id", None) == report_wid:
                        self._open_failed_calls.remove(rep)
                        break
            except Exception:
                pass

    def on_sessionfinish(self, session: Any, exitstatus: int) -> None:
        try:
            if self._suite_start is None:
                self._suite_start = _new_instant()

            duration = self._suite_start.elapsed()
            # as_utc might not exist on some objects, but ours does
            try:
                timestamp = self._suite_start.as_utc().astimezone().isoformat()
            except Exception:
                timestamp = datetime.datetime.utcnow().isoformat()

            numtests = (
                int(self.stats.get("passed", 0))
                + int(self.stats.get("failure", 0))
                + int(self.stats.get("skipped", 0))
                + int(self.stats.get("error", 0))
                - int(self._cnt_double_fail_tests)
            )

            testcases = []  # type: List[Dict[str, Any]]
            for key in self._cases_order:
                case = self._cases[key]
                tc = {
                    "classname": case.classname,
                    "name": case.name,
                    "time": float("%.3f" % case.time),
                }

                if self.family == "xunit1":
                    if case.file is not None:
                        tc["file"] = case.file
                    if case.line is not None:
                        tc["line"] = case.line
                    if case.url is not None:
                        tc["url"] = case.url

                if case.properties:
                    tc["properties"] = list(case.properties)

                if case.skipped is not None:
                    tc["skipped"] = case.skipped
                if case.failure is not None:
                    tc["failure"] = case.failure
                if case.error is not None:
                    tc["error"] = case.error

                if case.system_out:
                    tc["system-out"] = case.system_out
                if case.system_err:
                    tc["system-err"] = case.system_err

                testcases.append(tc)

            suite = {
                "name": self.suite_name,
                "errors": int(self.stats.get("error", 0)),
                "failures": int(self.stats.get("failure", 0)),
                "skipped": int(self.stats.get("skipped", 0)),
                "tests": int(numtests),
                "time": float("%.3f" % float(getattr(duration, "seconds", 0.0) or 0.0)),
                "timestamp": timestamp,
                "hostname": platform.node(),
            }

            if self.global_properties:
                suite["properties"] = [{"name": k, "value": v} for (k, v) in self.global_properties]

            suite["testcase"] = testcases

            doc = {
                "testsuites": {"name": "pytest tests", "testsuite": suite},
                "meta": {"exitstatus": int(exitstatus)},
            }
            if self.meta:
                doc["meta"]["custom"] = dict(self.meta)

            with _safe_open_file(self.logfile, self.append) as f:
                if self.ndjson:
                    json.dump(doc, f, ensure_ascii=False)
                    f.write("\n")
                else:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
                    f.write("\n")
        except Exception:
            # Never crash pytest at session end because reporting failed.
            return

    def on_terminal_summary(self, terminalreporter: Any, config: Any) -> None:
        try:
            verbosity = _get_config_verbosity(config)
            if verbosity < 0:
                return

            suffix = ""
            if self.ndjson:
                suffix += " (ndjson)"
            if self.append:
                suffix += " (append)"

            try:
                terminalreporter.write_sep("-", "generated json file: %s%s" % (self.logfile, suffix))
            except Exception:
                # very old terminal reporter
                try:
                    terminalreporter.write_line("generated json file: %s%s" % (self.logfile, suffix))
                except Exception:
                    pass

            if self.append and (not self.ndjson):
                try:
                    terminalreporter.write_line(
                        "NOTE: --junit-json-append without --junit-json-ndjson appends multiple "
                        "JSON documents to one file (not valid single-JSON). Prefer NDJSON."
                    )
                except Exception:
                    pass
        except Exception:
            # Never crash pytest because summary printing failed.
            return


# -----------------------------------------------------------------------------
# Options + config
# -----------------------------------------------------------------------------
def _parse_kv(pairs: List[str]) -> List[Tuple[str, object]]:
    out = []  # type: List[Tuple[str, object]]
    for p in pairs:
        if "=" not in p:
            out.append((p, ""))
        else:
            k, v = p.split("=", 1)
            out.append((k, _parse_meta_scalar(v)))
    return out


def pytest_addoption(parser: Any) -> None:
    group = None
    try:
        group = parser.getgroup("terminal reporting")
    except Exception:
        group = parser

    def _filename_type(x: str) -> str:
        # If pytest provides filename_arg use it; else accept raw string.
        if filename_arg is not None:
            try:
                return filename_arg(x, optname="--junit-json")  # type: ignore[misc]
            except Exception:
                return x
        return x

    try:
        group.addoption(
            "--junit-json",
            action="store",
            dest="junitjson_path",
            metavar="path",
            type=functools.partial(_filename_type),
            default=None,
            help="Create junit-like JSON report file at given path",
        )
    except Exception:
        # older parser APIs: try again with minimal args
        try:
            group.addoption("--junit-json", action="store", dest="junitjson_path", default=None)
        except Exception:
            pass

    try:
        group.addoption(
            "--junit-json-meta",
            action="append",
            dest="junitjson_meta",
            default=[],
            metavar="k=v",
            help="Add custom meta key/value to JSON (repeatable). "
            "Values auto-parse true/false/null/int/float; otherwise kept as string.",
        )
    except Exception:
        pass

    try:
        group.addoption(
            "--junit-json-ndjson",
            action="store_true",
            dest="junitjson_ndjson",
            default=False,
            help="Write report as NDJSON (one JSON object per line; one line per run).",
        )
    except Exception:
        pass

    try:
        group.addoption(
            "--junit-json-append",
            action="store_true",
            dest="junitjson_append",
            default=False,
            help="Append to existing --junit-json file (intended for NDJSON).",
        )
    except Exception:
        pass

    # INI options; tolerate missing addini on very old pytest
    try:
        parser.addini("junit_suite_name", "Test suite name", default="pytest")
        parser.addini(
            "junit_logging",
            "Captured logging in report: one of no|log|system-out|system-err|out-err|all",
            default="no",
        )
        parser.addini(
            "junit_log_passing_tests",
            "Capture log information for passing tests",
            type="bool",
            default=True,
        )
        parser.addini(
            "junit_duration_report",
            "Duration time to report: one of total|call",
            default="total",
        )
        parser.addini(
            "junit_family",
            "Emit JSON similar to JUnit family: one of legacy|xunit1|xunit2",
            default="xunit2",
        )
    except Exception:
        pass


def _ini_get(config: Any, name: str, default: Any) -> Any:
    try:
        return config.getini(name)
    except Exception:
        return default


def pytest_configure(config: Any) -> None:
    try:
        path = getattr(getattr(config, "option", None), "junitjson_path", None)
        if not path:
            return

        # Avoid writing from xdist workers: workerinput attribute exists in many versions.
        if hasattr(config, "workerinput"):
            return

        plugin = LogJUnitJSON(
            logfile=path,
            prefix=getattr(getattr(config, "option", None), "junitprefix", None),
            suite_name=_ini_get(config, "junit_suite_name", "pytest"),
            logging_mode=_ini_get(config, "junit_logging", "no"),
            report_duration=_ini_get(config, "junit_duration_report", "total"),
            family=_ini_get(config, "junit_family", "xunit2"),
            log_passing_tests=bool(_ini_get(config, "junit_log_passing_tests", True)),
            meta_kv=_parse_kv(list(getattr(getattr(config, "option", None), "junitjson_meta", []) or [])),
            ndjson=bool(getattr(getattr(config, "option", None), "junitjson_ndjson", False)),
            append=bool(getattr(getattr(config, "option", None), "junitjson_append", False)),
        )
        _set_active(config, plugin)
    except Exception:
        # Never crash pytest during configure; just skip plugin.
        return


def pytest_unconfigure(config: Any) -> None:
    try:
        _del_active(config)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Module-level hook implementations (pluggy-safe across versions)
# -----------------------------------------------------------------------------
def pytest_sessionstart(session: Any) -> None:
    plugin = _get_active()
    if plugin is not None:
        try:
            plugin.on_sessionstart()
        except Exception:
            pass


def pytest_collectreport(report: Any) -> None:
    plugin = _get_active()
    if plugin is not None:
        try:
            plugin.on_collectreport(report)
        except Exception:
            pass


def pytest_internalerror(excrepr: Any) -> None:
    plugin = _get_active()
    if plugin is not None:
        try:
            plugin.on_internalerror(excrepr)
        except Exception:
            pass


def pytest_runtest_logreport(report: Any) -> None:
    plugin = _get_active()
    if plugin is not None:
        try:
            plugin.on_runtest_logreport(report)
        except Exception:
            pass


# pytest_sessionfinish signature varies: sometimes (session, exitstatus),
# sometimes additional args. Accept *args/**kwargs to avoid validation issues.
def pytest_sessionfinish(session: Any, exitstatus: int, *args: Any, **kwargs: Any) -> None:
    plugin = _get_active()
    if plugin is not None:
        try:
            plugin.on_sessionfinish(session, exitstatus)
        except Exception:
            pass


# pytest_terminal_summary signature varies widely across versions.
# Accept *args/**kwargs to avoid pluggy validation mismatch.
def pytest_terminal_summary(terminalreporter: Any, *args: Any, **kwargs: Any) -> None:
    plugin = _get_active()
    if plugin is None:
        return

    # Newer pytest might pass config=...; older has terminalreporter.config
    config = kwargs.get("config", getattr(terminalreporter, "config", None))
    if config is None:
        # Some versions pass "config" positionally as second arg, or pass exitstatus
        if args:
            # try: if it's a Config-ish object, it will have .option or .getini
            cand = args[0]
            if hasattr(cand, "option") or hasattr(cand, "getini") or hasattr(cand, "getini"):
                config = cand

    if config is None:
        return

    try:
        plugin.on_terminal_summary(terminalreporter, config)
    except Exception:
        pass
