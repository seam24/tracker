# src/pytest_mutation_runner/plugin.py

import json
from pathlib import Path
import pytest

MODE="--mode"
PROG_RES="--program-results"
MUT_RES = "--mutant-results"
STOP_ON_KILL="--stop-on-kill"
TIMEOUT_MIN="--timeout-min"
TIMEOUT_MUL="--timeout-multiplier"
INCOMP_DISABLE = "--disable-incompetent"
DROP_MISSING = "--drop-missing-tests"

class ProgramMode:

    def __init__(self):
        self.config=None
        self.results=[]

    def pytest_configure(self, config):
        self.config=config

    def pytest_collection_modifyitems(self,session, config, items):
        # apply a simple min-timeout to all tests, but only if > 0
        try:
            m = float(config.getoption(TIMEOUT_MIN))
        except (TypeError, ValueError):
            m = 0.0

        if m <= 0:
            return # user implicitly disabled timeouts for program run

        for item in items:
            item.add_marker(pytest.mark.timeout(m))

    def pytest_runtest_logreport(self,report):
        if report.when != "call":
            return
        self.results.append({
            "id": report.nodeid,
            "outcome": report.outcome,
            "duration": getattr(report, "duration", None),
        })

    def pytest_sessionfinish(self,session, exitstatus):
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
        self.config=config
        self.disable_incompetent = config.getoption(INCOMP_DISABLE)
        results_path = config.getoption(PROG_RES)
        path = Path(results_path)
        if not path.exists():
            raise pytest.UsageError(f"--program-results '{results_path}' not found for mutant mode")

        # load reference results
        data = json.loads(path.read_text())
        ref_map = {
            entry["id"]: {"outcome": entry.get("outcome"), "duration": entry.get("duration")}
            for entry in data if "id" in entry
        }
        self.reference_data = ref_map

        self.mutant_result = {entry["id"]: "pending" for entry in data if "id" in entry}

        # require explicit --mutant-results
        mut_path_opt = config.getoption(MUT_RES)
        if not mut_path_opt:
            raise pytest.UsageError("--mutant-results must be provided in mutant mode")
        self.mutant_results_path = Path(mut_path_opt)

    def pytest_collection_modifyitems(self,session, config, items):

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

        current = self.mutant_result.get(nodeid, "pending")

        # Detect incompetent first
        if (not self.disable_incompetent
                and current in ("pending", "survived")
                and self._is_incompetent(report)):
            self.mutant_result[nodeid] = "incompetent"
            return

        orig_outcome = self.reference_data[nodeid].get("outcome")

        if orig_outcome is not None and report.outcome != orig_outcome:
            # KILLED by this test
            self.mutant_result[nodeid] = "killed"
            self.mutant_killed = True
            print(
                f"\n Mutant killed by {nodeid} "
                f"(expected {orig_outcome}, got {report.outcome})"
            )

            if self.config.getoption(STOP_ON_KILL):
                pytest.exit("Mutant killed", returncode=0)
        else:
            # Test ran and did NOT kill the mutant → survived (unless already incompetent/killed)
            if current == "pending":
                self.mutant_result[nodeid] = "survived"

    def _is_incompetent(self,report):
        # According to: https://github.com/mutpy/mutpy/blob/master/mutpy/test_runners/pytest_runner.py#L29
        return (report.longrepr is not None and
                    'TypeError' in report.longrepr.reprcrash.message)

    def _is_timeout(self,report):
        return (report.longrepr is not None
                and "timeout" in report.longrepr.reprcrash.message.lower())

    def pytest_sessionfinish(self,session, exitstatus):
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
        # Do nothing: pytest behaves as if this plugin is not there
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

