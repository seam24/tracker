# conftest.py (or package it as a plugin)
# NDJSON test report aimed at mutation testing, with JUnit-like labeling:
#
# ASSUMPTION:
# - One execution of each test case is stored, even if the test is run multiple times.
#
# Outcome mapping:
# - failed := report.failed in the CALL phase (includes assertion failures *and* exceptions in test body)
# - error := report.failed in SETUP or TEARDOWN phase
# - skipped := report.skipped (including normal skips)
# - xfailed/xpassed tracked via report.wasxfail (kept distinct, not merged)
#
# Other goals:
# - Broad compatibility: no f-strings, no typing, no __future__ needed (Python 3.4+).
# - Tests are keyed by nodeid (no redundant "id" field)
# - Include message/text for non-pass-ish outcomes
# - Duration mode: call or total
# - Meta: --json-report-meta key=value (value parsed as int/float if possible)
#
# Output:
# - NDJSON: exactly ONE JSON object per pytest invocation (one line).
# - If --json-report-append is used: append that JSON line.

import json
import re
import sys
from pathlib import Path

import pytest

JSON_OUT = "--json-report"
JSON_APPEND = "--json-report-append"
JSON_META = "--json-report-meta"
JSON_DURATION = "--json-report-duration"  # call|total

DURATION_CHOICES = ("call", "total")

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def pytest_addoption(parser):
    g = parser.getgroup("reporting")
    g.addoption(
        JSON_OUT,
        action="store",
        default=None,
        help="Write a NDJSON test report to this path (one JSON object per run).",
    )
    g.addoption(
        JSON_APPEND,
        action="store_true",
        default=False,
        help="Append the JSON object line to the NDJSON file instead of overwriting.",
    )
    g.addoption(
        JSON_META,
        action="append",
        default=[],
        metavar="key=value",
        help="Optional metadata (repeatable). Values parsed as int/float if possible.",
    )
    g.addoption(
        JSON_DURATION,
        action="store",
        default="call",
        help="Duration to store per test: call|total (default: call).",
    )


def pytest_configure(config):
    path = config.getoption(JSON_OUT)
    if not path:
        return

    # xdist safety: only write on master
    if hasattr(config, "workerinput"):
        return

    plugin = SimpleNdjsonReportPlugin(config)
    config.pluginmanager.register(plugin, "simple-ndjson-report")


def _parse_value(v):
    v = (v or "").strip()
    if _INT_RE.match(v):
        try:
            return int(v)
        except Exception:
            return v
    if _FLOAT_RE.match(v):
        try:
            return float(v)
        except Exception:
            return v
    return v


def _parse_kv(s):
    if "=" not in s:
        raise pytest.UsageError("{} expects key=value, got {!r}".format(JSON_META, s))
    k, v = s.split("=", 1)
    k = k.strip()
    if not k:
        raise pytest.UsageError("{} has empty key in {!r}".format(JSON_META, s))
    return k, _parse_value(v)


def _longrepr_to_text_and_message(report):
    lr = getattr(report, "longrepr", None)
    if lr is None:
        return None, None

    msg = None
    try:
        rc = getattr(lr, "reprcrash", None)
        if rc is not None:
            m = getattr(rc, "message", None)
            if m:
                msg = m
    except Exception:
        pass

    try:
        txt = str(lr)
    except Exception:
        txt = repr(lr)

    if (not msg) and txt:
        msg = "<no message>"

    return msg, txt


class SimpleNdjsonReportPlugin(object):
    def __init__(self, config):
        self.config = config
        self.path = Path(config.getoption(JSON_OUT))
        self.append = bool(config.getoption(JSON_APPEND))

        self.duration_mode = (config.getoption(JSON_DURATION) or "call").strip().lower()
        if self.duration_mode not in DURATION_CHOICES:
            raise pytest.UsageError(
                "{} must be one of {}, got {!r}".format(
                    JSON_DURATION, DURATION_CHOICES, self.duration_mode
                )
            )

        meta = {}
        for kv in (config.getoption(JSON_META) or []):
            k, v = _parse_kv(kv)
            meta[k] = v

        impl = getattr(sys, "implementation", None)
        impl_name = getattr(impl, "name", None) if impl is not None else None

        # nodeid -> entry
        self.tests = {}

        # collection info (collection failures are *not* counted as per-test outcomes here)
        self.collection = {"ok": None, "count": None, "problems": []}

        self.session = {
            "pytest_version": getattr(pytest, "__version__", None),
            "platform": {
                "python": "{}.{}.{}".format(
                    sys.version_info.major, sys.version_info.minor, sys.version_info.micro
                ),
                "implementation": impl_name,
                "os": sys.platform,
            },
            "meta": meta,
            "exitstatus": None,
            "summary": {
                "passed": 0,
                "failed": 0,
                "error": 0,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
            },
            "duration_mode": self.duration_mode,
        }

    def pytest_collection_finish(self, session):
        items = getattr(session, "items", None)
        if items is not None:
            self.collection["count"] = len(items)
            if not self.collection["problems"]:
                self.collection["ok"] = True

    def pytest_collectreport(self, report):
        # Mirrors junitxml: collection problems are recorded separately as "error"/"skipped"
        if getattr(report, "passed", False):
            return
        msg, txt = _longrepr_to_text_and_message(report)
        outcome = "failed" if getattr(report, "failed", False) else "skipped"
        self.collection["problems"].append(
            {
                "nodeid": getattr(report, "nodeid", None),
                "outcome": outcome,
                "message": msg,
                "text": txt,
            }
        )
        self.collection["ok"] = False

    def pytest_runtest_logreport(self, report):
        nodeid = getattr(report, "nodeid", None)
        if not nodeid:
            return

        entry = self.tests.get(nodeid)
        if entry is None:
            loc = getattr(report, "location", None)
            location = None
            if loc and isinstance(loc, (tuple, list)) and len(loc) >= 3:
                location = {"file": loc[0], "line": loc[1], "name": loc[2]}

            entry = {
                "location": location,
                # internal aggregation
                "_phases": {},  # when -> (passed/failed/skipped)
                "_durations": {"setup": 0.0, "call": 0.0, "teardown": 0.0},
                "_wasxfail": None,
                # output fields
                "outcome": None,
                "duration": None,
                "message": None,
                "text": None,
                "xfail": False,
                "xfail_reason": None,
                "phase": None,  # the phase that provided message/text (or determined outcome)
            }
            self.tests[nodeid] = entry

        when = getattr(report, "when", None)  # setup/call/teardown
        outcome = getattr(report, "outcome", None)  # passed/failed/skipped
        dur = float(getattr(report, "duration", 0.0) or 0.0)

        if when in entry["_durations"]:
            entry["_durations"][when] += dur
        if when:
            entry["_phases"][when] = outcome

        wasxfail = getattr(report, "wasxfail", None)
        if wasxfail is not None and entry["_wasxfail"] is None:
            entry["_wasxfail"] = wasxfail

        # Prefer details from the phase that actually failed (and prefer setup/teardown over call).
        if outcome != "passed":
            msg, txt = _longrepr_to_text_and_message(report)

            if entry["message"] is None and entry["text"] is None:
                entry["message"] = msg
                entry["text"] = txt
                entry["phase"] = when
            else:
                if when in ("setup", "teardown") and entry.get("phase") not in ("setup", "teardown"):
                    entry["message"] = msg
                    entry["text"] = txt
                    entry["phase"] = when

    def pytest_sessionfinish(self, session, exitstatus):
        self.session["exitstatus"] = exitstatus

        for nodeid, entry in self.tests.items():
            phases = entry.get("_phases", {})
            durs = entry.get("_durations", {})

            call_out = phases.get("call")
            setup_out = phases.get("setup")
            teardown_out = phases.get("teardown")

            base = "passed"
            base_phase = None

            # JUnit-like base classification:
            if call_out == "failed":
                base = "failed"
                base_phase = "call"
            elif setup_out == "failed":
                base = "error"
                base_phase = "setup"
            elif teardown_out == "failed":
                base = "error"
                base_phase = "teardown"
            else:
                if "skipped" in list(phases.values()):
                    base = "skipped"
                    if setup_out == "skipped":
                        base_phase = "setup"
                    elif call_out == "skipped":
                        base_phase = "call"
                    elif teardown_out == "skipped":
                        base_phase = "teardown"

            wasxfail = entry.get("_wasxfail", None)
            if wasxfail is not None:
                entry["xfail"] = True
                try:
                    entry["xfail_reason"] = str(wasxfail)
                except Exception:
                    entry["xfail_reason"] = "xfail"

                if base == "passed":
                    final = "xpassed"
                else:
                    final = "xfailed"
            else:
                final = base

            entry["outcome"] = final

            if entry.get("phase") is None:
                entry["phase"] = base_phase

            if final in self.session["summary"]:
                self.session["summary"][final] += 1
            else:
                self.session["summary"].setdefault(final, 0)
                self.session["summary"][final] += 1

            # Duration
            if self.duration_mode == "call":
                entry["duration"] = float(durs.get("call", 0.0) or 0.0)
            else:
                entry["duration"] = float(
                    (durs.get("setup", 0.0) or 0.0)
                    + (durs.get("call", 0.0) or 0.0)
                    + (durs.get("teardown", 0.0) or 0.0)
                )

            # drop internals
            entry.pop("_phases", None)
            entry.pop("_durations", None)
            entry.pop("_wasxfail", None)

            # shrink: drop details for clean passes only
            if entry["outcome"] == "passed":
                entry.pop("message", None)
                entry.pop("text", None)
                entry.pop("xfail", None)
                entry.pop("xfail_reason", None)
                entry.pop("phase", None)
            else:
                if entry.get("message") is None:
                    entry.pop("message", None)
                if entry.get("text") is None:
                    entry.pop("text", None)

        if self.collection["ok"] is None:
            self.collection["ok"] = not bool(self.collection["problems"])

        payload = {
            "session": self.session,
            "collection": self.collection,
            "tests": self.tests,
        }

        self._write_ndjson_line(payload)

    def _write_ndjson_line(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # NDJSON: one JSON object per line
        line = json.dumps(payload, ensure_ascii=False) + "\n"

        mode = "a" if self.append else "w"
        with self.path.open(mode, encoding="utf-8") as f:
            f.write(line)

        print("\n[simple-json-report] wrote NDJSON line to {}".format(self.path.resolve()))
