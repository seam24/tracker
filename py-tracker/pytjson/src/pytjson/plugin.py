# pytjson/plugin.py
# mypy: allow-untyped-defs
"""
A minimal "JUnit-XML equivalent, but JSON" reporter for pytest.

Goal:
- Similar information level to pytest's built-in junitxml plugin
- JSON output shaped like JUnit: testsuites -> testsuite -> testcase
- By default behaves like junit_family=xunit2 (minimal testcase fields)

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

Notes:
- Avoids writing on xdist workers (master writes single file).
- If junit_family is xunit2, testcase includes only classname/name/time (+ outcome nodes).
- If junit_family is xunit1/legacy, testcase may also include file/line/url (like junitxml legacy attrs).
- Meta supports JSON-native scalar types (str/int/float/bool/null).
- NDJSON option A: one line per *test run* (entire doc on one line).

Compatibility:
- Supports older pytest/pluggy by:
  * exposing module-level hook functions (no class-instance hook registration)
  * storing plugin instance via config.stash when available, else config attribute
  * avoiding _pytest.timing.Instant if missing
  * providing fallback for nodes.SEP import
"""

from __future__ import annotations

from collections.abc import Callable
import datetime
import functools
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from typing import Any

import pytest
from _pytest._code.code import ExceptionRepr, ReprFileLocation
from _pytest.config import Config, filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter


# -----------------------------------------------------------------------------
# Compat: nodes.SEP moved around; fall back to "::" -> "." replacement if needed
# -----------------------------------------------------------------------------
try:
    from _pytest import nodes as _nodes  # type: ignore
    _NODE_SEP = getattr(_nodes, "SEP", "/")
except Exception:
    _NODE_SEP = "/"


# -----------------------------------------------------------------------------
# Compat: timing.Instant exists only on newer pytest
# -----------------------------------------------------------------------------
class _CompatInstant:
    def __init__(self):
        self._start = time.time()

    def elapsed(self):
        class _Elapsed:
            seconds = time.time() - self._start
        return _Elapsed()

    def as_utc(self):
        return datetime.datetime.utcfromtimestamp(self._start)


try:
    from _pytest import timing  # type: ignore
except Exception:
    timing = None  # type: ignore


def _new_instant():
    if timing is not None and hasattr(timing, "Instant"):
        return timing.Instant()  # type: ignore[attr-defined]
    return _CompatInstant()


# -----------------------------------------------------------------------------
# Compat: stash only exists on newer pytest
# -----------------------------------------------------------------------------
try:
    from _pytest.stash import StashKey  # pytest>=7

    json_key = StashKey["LogJUnitJSON"]()
except Exception:
    StashKey = None  # type: ignore
    json_key = None  # type: ignore


_ACTIVE: "LogJUnitJSON | None" = None


def _set_active(config: Config, plugin: "LogJUnitJSON") -> None:
    global _ACTIVE
    _ACTIVE = plugin
    if hasattr(config, "stash") and json_key is not None:
        config.stash[json_key] = plugin  # type: ignore[index]
    else:
        setattr(config, "_junitjson_plugin", plugin)


def _get_active_from_config(config: Config) -> "LogJUnitJSON | None":
    if hasattr(config, "stash") and json_key is not None:
        return config.stash.get(json_key, None)  # type: ignore[union-attr]
    return getattr(config, "_junitjson_plugin", None)


def _get_active() -> "LogJUnitJSON | None":
    return _ACTIVE


def _del_active(config: Config) -> None:
    global _ACTIVE
    if hasattr(config, "stash") and json_key is not None:
        try:
            del config.stash[json_key]  # type: ignore[index]
        except Exception:
            pass
    if hasattr(config, "_junitjson_plugin"):
        try:
            delattr(config, "_junitjson_plugin")
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

    def repl(matchobj: re.Match[str]) -> str:
        i = ord(matchobj.group())
        if i <= 0xFF:
            return f"#x{i:02X}"
        return f"#x{i:04X}"

    illegal_re = (
        "[^\u0009\u000a\u000d\u0020-\u007e\u0080-\ud7ff\ue000-\ufffd\u10000-\u10ffff]"
    )
    return re.sub(illegal_re, repl, str(arg))


def mangle_test_address(address: str) -> list[str]:
    path, possible_open_bracket, params = address.partition("[")
    names = path.split("::")
    names[0] = names[0].replace(_NODE_SEP, ".")
    names[0] = re.sub(r"\.py$", "", names[0])
    names[-1] += possible_open_bracket + params
    return names


def _check_record_param_type(param: str, v: str) -> None:
    __tracebackhide__ = True
    if not isinstance(v, str):
        raise TypeError(
            f"{param} parameter needs to be a string, but {type(v).__name__} given"
        )


def _check_meta_value(key: str, value: object) -> None:
    __tracebackhide__ = True
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise TypeError(
            f"meta value for {key!r} must be str|int|float|bool|None, "
            f"got {type(value).__name__}"
        )


def _parse_meta_scalar(v: str) -> object:
    s = v.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
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


# -----------------------------------------------------------------------------
# Public fixtures (work with stash or attribute)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def record_junitjson_testsuite_property(
    request: FixtureRequest,
) -> Callable[[str, object], None]:
    __tracebackhide__ = True

    def noop(name: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type("name", name)

    plugin = _get_active_from_config(request.config)
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

    plugin = _get_active_from_config(request.config)
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

    file: str | None = None
    line: int | None = None
    url: str | None = None

    time: float = 0.0
    properties: list[dict[str, str]] = None  # type: ignore[assignment]

    skipped: dict[str, str] | None = None
    failure: dict[str, str] | None = None
    error: dict[str, str] | None = None

    system_out: str | None = None
    system_err: str | None = None

    def __post_init__(self) -> None:
        if self.properties is None:
            self.properties = []


# -----------------------------------------------------------------------------
# Core reporter implementation (NO hook names on the class)
# -----------------------------------------------------------------------------
class LogJUnitJSON:
    def __init__(
        self,
        logfile: str,
        prefix: str | None,
        suite_name: str,
        logging_mode: str,
        report_duration: str,
        family: str,
        log_passing_tests: bool,
        meta_kv: list[tuple[str, object]],
        ndjson: bool,
        append: bool,
    ) -> None:
        logfile = os.path.expanduser(os.path.expandvars(logfile))
        self.logfile = os.path.normpath(os.path.abspath(logfile))
        self.prefix = prefix
        self.suite_name = suite_name
        self.logging = logging_mode
        self.log_passing_tests = log_passing_tests
        self.report_duration = report_duration

        self.family = "xunit1" if family == "legacy" else family
        self.ndjson = bool(ndjson)
        self.append = bool(append)

        self.stats: dict[str, int] = dict.fromkeys(
            ["error", "passed", "failure", "skipped"], 0
        )
        self.global_properties: list[tuple[str, str]] = []

        self.meta: dict[str, Any] = {}
        for k, v in meta_kv:
            self.add_meta(k, v)

        self._cases_order: list[tuple[str, str | None, int | None]] = []
        self._cases: dict[tuple[str, str | None, int | None], _CaseData] = {}

        self._open_failed_calls: list[TestReport] = []
        self._cnt_double_fail_tests = 0

        self._suite_start = None  # set in sessionstart

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

    def _case_key(self, report: TestReport | str) -> tuple[str, str | None, int | None]:
        nodeid = getattr(report, "nodeid", report)
        worker_id = getattr(report, "worker_id", None)
        item_index = getattr(report, "item_index", None)
        return (
            str(nodeid),
            (str(worker_id) if worker_id is not None else None),
            (int(item_index) if item_index is not None else None),
        )

    def _get_or_create_case(self, report: TestReport) -> _CaseData:
        key = self._case_key(report)
        existing = self._cases.get(key)
        if existing is not None:
            return existing

        names = mangle_test_address(report.nodeid)
        classnames = names[:-1]
        if self.prefix:
            classnames.insert(0, self.prefix)

        classname = ".".join(classnames)
        name = bin_text_escape(names[-1])

        file = None
        line = None
        url = None
        try:
            file = report.location[0]
            if report.location[1] is not None:
                line = int(report.location[1])
        except Exception:
            pass
        if hasattr(report, "url"):
            url = getattr(report, "url")

        case = _CaseData(
            classname=classname,
            name=name,
            file=file,
            line=line,
            url=url,
        )
        self._cases[key] = case
        self._cases_order.append(key)
        return case

    def _append_properties_from_report(self, case: _CaseData, report: TestReport) -> None:
        for k, v in (getattr(report, "user_properties", None) or []):
            case.properties.append({"name": str(k), "value": bin_text_escape(v)})

    def _prepare_content(self, content: str, header: str) -> str:
        return "\n".join([header.center(80, "-"), content, ""])

    def _write_captured_output(self, case: _CaseData, report: TestReport) -> None:
        if not self.log_passing_tests and report.passed:
            return
        if self.logging == "no":
            return

        content_out = getattr(report, "capstdout", "") or ""
        content_err = getattr(report, "capstderr", "") or ""
        content_log = getattr(report, "caplog", "") or ""

        out_buf = ""
        err_buf = ""

        if self.logging in ["log", "all"]:
            out_buf += self._prepare_content(content_log, " Captured Log ")
        if self.logging in ["system-out", "out-err", "all"]:
            out_buf += self._prepare_content(content_out, " Captured Out ")
        if self.logging in ["system-err", "out-err", "all"]:
            err_buf += self._prepare_content(content_err, " Captured Err ")

        if out_buf:
            case.system_out = (case.system_out or "") + bin_text_escape(out_buf)
        if err_buf:
            case.system_err = (case.system_err or "") + bin_text_escape(err_buf)

    def update_testcase_duration(self, report: TestReport) -> None:
        if self.report_duration in {"total", getattr(report, "when", "")}:
            case = self._get_or_create_case(report)
            case.time += float(getattr(report, "duration", 0.0) or 0.0)

    def on_collectreport(self, report: TestReport) -> None:
        if report.passed:
            return

        case = self._get_or_create_case(report)
        self.update_testcase_duration(report)

        if report.failed:
            self.add_stats("error")
            case.error = {
                "message": "collection failure",
                "text": bin_text_escape(getattr(report, "longrepr", "")),
            }
        else:
            self.add_stats("skipped")
            case.skipped = {
                "message": "collection skipped",
                "text": bin_text_escape(getattr(report, "longrepr", "")),
                "type": "pytest.skip",
            }

    def on_internalerror(self, excrepr: ExceptionRepr) -> None:
        key = ("internal", None, None)
        if key not in self._cases:
            case = _CaseData(classname="pytest", name="internal")
            self._cases[key] = case
            self._cases_order.append(key)
        case = self._cases[key]
        self.add_stats("error")
        case.error = {"message": "internal error", "text": bin_text_escape(excrepr)}

    def on_runtest_logreport(self, report: TestReport) -> None:
        close_report = None

        if report.passed:
            if getattr(report, "when", "") == "call":
                self.add_stats("passed")
                case = self._get_or_create_case(report)
                self._append_properties_from_report(case, report)

        elif report.failed:
            if getattr(report, "when", "") == "teardown":
                report_wid = getattr(report, "worker_id", None)
                report_ii = getattr(report, "item_index", None)
                close_report = next(
                    (
                        rep
                        for rep in self._open_failed_calls
                        if (
                            rep.nodeid == report.nodeid
                            and getattr(rep, "item_index", None) == report_ii
                            and getattr(rep, "worker_id", None) == report_wid
                        )
                    ),
                    None,
                )
                if close_report:
                    self._cnt_double_fail_tests += 1

            case = self._get_or_create_case(report)
            self._append_properties_from_report(case, report)

            if getattr(report, "when", "") == "call":
                self.add_stats("failure")
                if hasattr(report, "wasxfail"):
                    self.add_stats("skipped")
                    case.skipped = {
                        "message": "xfail-marked test passes unexpectedly",
                        "type": "pytest.xfail",
                    }
                else:
                    longrepr = getattr(report, "longrepr", None)
                    reprcrash: ReprFileLocation | None = getattr(longrepr, "reprcrash", None)
                    message = reprcrash.message if reprcrash is not None else str(longrepr)
                    case.failure = {
                        "message": bin_text_escape(message),
                        "text": bin_text_escape(longrepr),
                    }
                self._open_failed_calls.append(report)

                if not self.log_passing_tests:
                    self._write_captured_output(case, report)
            else:
                self.add_stats("error")
                longrepr = getattr(report, "longrepr", None)
                reprcrash = getattr(longrepr, "reprcrash", None)
                reason = reprcrash.message if reprcrash is not None else str(longrepr)
                when = getattr(report, "when", "")
                if when == "teardown":
                    msg = f'failed on teardown with "{reason}"'
                else:
                    msg = f'failed on setup with "{reason}"'
                case.error = {
                    "message": bin_text_escape(msg),
                    "text": bin_text_escape(longrepr),
                }

        elif report.skipped:
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
                details = ""
                skipreason = ""
                try:
                    filename, lineno, reason = longrepr
                    skipreason = str(reason)
                    if skipreason.startswith("Skipped: "):
                        skipreason = skipreason[9:]
                    details = f"{filename}:{lineno}: {skipreason}"
                except Exception:
                    skipreason = str(longrepr)
                    details = skipreason

                case.skipped = {
                    "type": "pytest.skip",
                    "message": bin_text_escape(skipreason),
                    "text": bin_text_escape(details),
                }
                self._write_captured_output(case, report)

        self.update_testcase_duration(report)

        if getattr(report, "when", "") == "teardown":
            case = self._get_or_create_case(report)
            self._write_captured_output(case, report)

            report_wid = getattr(report, "worker_id", None)
            report_ii = getattr(report, "item_index", None)
            close_report = next(
                (
                    rep
                    for rep in self._open_failed_calls
                    if (
                        rep.nodeid == report.nodeid
                        and getattr(rep, "item_index", None) == report_ii
                        and getattr(rep, "worker_id", None) == report_wid
                    )
                ),
                None,
            )
            if close_report:
                self._open_failed_calls.remove(close_report)

    def on_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        dirname = os.path.dirname(os.path.abspath(self.logfile))
        os.makedirs(dirname, exist_ok=True)

        if self._suite_start is None:
            self._suite_start = _new_instant()

        duration = self._suite_start.elapsed()
        timestamp = self._suite_start.as_utc().astimezone().isoformat()

        numtests = (
            self.stats["passed"]
            + self.stats["failure"]
            + self.stats["skipped"]
            + self.stats["error"]
            - self._cnt_double_fail_tests
        )

        testcases: list[dict[str, Any]] = []
        for key in self._cases_order:
            case = self._cases[key]
            tc: dict[str, Any] = {
                "classname": case.classname,
                "name": case.name,
                "time": float(f"{case.time:.3f}"),
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

        suite: dict[str, Any] = {
            "name": self.suite_name,
            "errors": int(self.stats["error"]),
            "failures": int(self.stats["failure"]),
            "skipped": int(self.stats["skipped"]),
            "tests": int(numtests),
            "time": float(f"{duration.seconds:.3f}"),
            "timestamp": timestamp,
            "hostname": platform.node(),
        }

        if self.global_properties:
            suite["properties"] = [
                {"name": k, "value": v} for (k, v) in self.global_properties
            ]

        suite["testcase"] = testcases

        doc: dict[str, Any] = {
            "testsuites": {"name": "pytest tests", "testsuite": suite},
            "meta": {"exitstatus": int(exitstatus)},
        }
        if self.meta:
            doc["meta"]["custom"] = dict(self.meta)

        mode = "a" if self.append else "w"
        with open(self.logfile, mode, encoding="utf-8") as f:
            if self.ndjson:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")
            else:
                json.dump(doc, f, ensure_ascii=False, indent=2)
                f.write("\n")

    def on_terminal_summary(self, terminalreporter: TerminalReporter, config: Config) -> None:
        if config.get_verbosity() >= 0:
            suffix = ""
            if self.ndjson:
                suffix += " (ndjson)"
            if self.append:
                suffix += " (append)"
            terminalreporter.write_sep("-", f"generated json file: {self.logfile}{suffix}")
            if self.append and not self.ndjson:
                terminalreporter.write_line(
                    "NOTE: --junit-json-append without --junit-json-ndjson appends multiple "
                    "JSON documents to one file (not valid single-JSON). Prefer NDJSON."
                )


# -----------------------------------------------------------------------------
# Options + config
# -----------------------------------------------------------------------------
def _parse_kv(pairs: list[str]) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for p in pairs:
        if "=" not in p:
            out.append((p, ""))
        else:
            k, v = p.split("=", 1)
            out.append((k, _parse_meta_scalar(v)))
    return out


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junit-json",
        action="store",
        dest="junitjson_path",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junit-json"),
        default=None,
        help="Create junit-like JSON report file at given path",
    )
    group.addoption(
        "--junit-json-meta",
        action="append",
        dest="junitjson_meta",
        default=[],
        metavar="k=v",
        help="Add custom meta key/value to JSON (repeatable). "
        "Values auto-parse true/false/null/int/float; otherwise kept as string.",
    )
    group.addoption(
        "--junit-json-ndjson",
        action="store_true",
        dest="junitjson_ndjson",
        default=False,
        help="Write report as NDJSON (one JSON object per line; one line per run).",
    )
    group.addoption(
        "--junit-json-append",
        action="store_true",
        dest="junitjson_append",
        default=False,
        help="Append to existing --junit-json file (intended for NDJSON).",
    )

    parser.addini("junit_suite_name", "Test suite name", default="pytest")
    parser.addini(
        "junit_logging",
        "Captured logging in report: one of no|log|system-out|system-err|out-err|all",
        default="no",
    )
    parser.addini(
        "junit_log_passing_tests",
        "Capture log information for passing tests: ",
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


def pytest_configure(config: Config) -> None:
    path = getattr(config.option, "junitjson_path", None)
    if path and not hasattr(config, "workerinput"):
        plugin = LogJUnitJSON(
            logfile=path,
            prefix=getattr(config.option, "junitprefix", None),  # reuse pytest's --junit-prefix
            suite_name=config.getini("junit_suite_name"),
            logging_mode=config.getini("junit_logging"),
            report_duration=config.getini("junit_duration_report"),
            family=config.getini("junit_family"),
            log_passing_tests=bool(config.getini("junit_log_passing_tests")),
            meta_kv=_parse_kv(list(getattr(config.option, "junitjson_meta", []) or [])),
            ndjson=bool(getattr(config.option, "junitjson_ndjson", False)),
            append=bool(getattr(config.option, "junitjson_append", False)),
        )
        _set_active(config, plugin)


def pytest_unconfigure(config: Config) -> None:
    _del_active(config)


# -----------------------------------------------------------------------------
# Module-level hook implementations (pluggy-safe across versions)
# -----------------------------------------------------------------------------
def pytest_sessionstart(session: pytest.Session) -> None:
    plugin = _get_active()
    if plugin is not None:
        plugin.on_sessionstart()


def pytest_collectreport(report: TestReport) -> None:
    plugin = _get_active()
    if plugin is not None:
        plugin.on_collectreport(report)


def pytest_internalerror(excrepr: ExceptionRepr) -> None:
    plugin = _get_active()
    if plugin is not None:
        plugin.on_internalerror(excrepr)


def pytest_runtest_logreport(report: TestReport) -> None:
    plugin = _get_active()
    if plugin is not None:
        plugin.on_runtest_logreport(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    plugin = _get_active()
    if plugin is not None:
        plugin.on_sessionfinish(session, exitstatus)


def pytest_terminal_summary(terminalreporter: TerminalReporter, config: Config) -> None:
    plugin = _get_active()
    if plugin is not None:
        plugin.on_terminal_summary(terminalreporter, config)
