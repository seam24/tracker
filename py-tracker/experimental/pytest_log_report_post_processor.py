#!/usr/bin/env python3
import re
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any


# Sections
FAILURES_SECTION = "=================================== FAILURES ==================================="
ERRORS_SECTION   = "==================================== ERRORS ===================================="

# Headers
FAIL_HEADER_RE = re.compile(r"^_{10,}\s*(.*?)\s*_{10,}\s*$")
ERROR_HEADER_RE = re.compile(r"^_{10,}\s*ERROR collecting\s+(.*?)\s*_{10,}\s*$")

# Traceback parsing
EXC_LINE_RE = re.compile(r"^E\s+([A-Za-z_]\w*(?:Error|Exception)):\s*(.*)$")
FRAME_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):\s+in\s+(?P<func>.+)$")

# Totals (pytest end line)
TOTALS_RE = re.compile(
    r"=+\s*"
    r"(?:(\d+)\s+failed,\s*)?"
    r"(?:(\d+)\s+passed,\s*)?"
    r"(?:(\d+)\s+skipped,\s*)?"
    r"(?:(\d+)\s+warnings\s+)?"
    r"in\s+([\d.]+)\s+seconds\s*=+"
)

# Collection interruption
COLLECTION_INTERRUPTED_RE = re.compile(r"Interrupted:\s+(\d+)\s+errors?\s+during\s+collection", re.IGNORECASE)

# Optional metadata
CONDA_ERROR_RE = re.compile(r"^ERROR\s+conda\..*?:\s+`(.+?)`\s+failed\.", re.IGNORECASE)
COLLECTED_RE = re.compile(r"^collected\s+(\d+)\s+items", re.IGNORECASE)
PYTEST_META_RE = re.compile(r"^platform\s+(.+?)\s+--\s+Python\s+(.+?),\s+pytest-([^,]+),", re.IGNORECASE)

# Optional: skip reasons only if pytest printed them (e.g. -rs)
SHORT_SUMMARY_HDR_RE = re.compile(r"^=+\s*short test summary info\s*=+", re.IGNORECASE)
SKIP_SUMMARY_LINE_RE = re.compile(r"^SKIPPED\s+\[\d+\]\s+(.+?)(?::\s*(.*))?$")


@dataclass
class Entry:
    where: str
    exception_type: Optional[str] = None
    message: Optional[str] = None
    text: str = ""
    # frames is optional and will be omitted unless requested
    frames: Optional[List[Dict[str, Any]]] = None


@dataclass
class Stats:
    conda_command: Optional[str] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None
    pytest_version: Optional[str] = None
    collected: Optional[int] = None
    totals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    failures: List[Entry] = field(default_factory=list)
    errors: List[Entry] = field(default_factory=list)
    skipped: List[Entry] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)


def _parse_entry(where: str, block_lines: List[str], include_frames: bool) -> Entry:
    e = Entry(where=where, text="\n".join(block_lines).strip())

    frames: List[Dict[str, Any]] = []
    for ln in block_lines:
        fm = FRAME_RE.match(ln)
        if fm:
            frames.append({
                "path": fm.group("path"),
                "line": int(fm.group("line")),
                "func": fm.group("func").strip(),
            })

        em = EXC_LINE_RE.match(ln)
        if em:
            e.exception_type = em.group(1)
            e.message = em.group(2)

    # fallback message if no "E   Type: msg" line
    if not e.message:
        for ln in reversed(block_lines):
            s = ln.strip()
            if s:
                e.message = s
                break

    if include_frames and frames:
        e.frames = frames

    return e


def _omit_none(obj):
    """Recursively drop None fields so 'frames' disappears when not requested."""
    if isinstance(obj, dict):
        return {k: _omit_none(v) for k, v in obj.items() if v is not None and _omit_none(v) is not None}
    if isinstance(obj, list):
        out = [_omit_none(x) for x in obj]
        out = [x for x in out if x is not None]
        return out
    return obj


def parse_log(text: str, include_frames: bool) -> Report:
    r = Report()
    lines = text.splitlines()

    # --- quick metadata scans (stats) ---
    for ln in lines:
        m = CONDA_ERROR_RE.match(ln)
        if m and not r.stats.conda_command:
            r.stats.conda_command = m.group(1)

        m = PYTEST_META_RE.match(ln)
        if m and not r.stats.platform:
            r.stats.platform, r.stats.python_version, r.stats.pytest_version = m.group(1), m.group(2), m.group(3)

        m = COLLECTED_RE.match(ln)
        if m and r.stats.collected is None:
            r.stats.collected = int(m.group(1))

    # --- parse FAILURES / ERRORS blocks ---
    section: Optional[str] = None  # None / "FAILURES" / "ERRORS"
    current_where: Optional[str] = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_where, current_lines, section
        if not current_where:
            current_lines = []
            return
        entry = _parse_entry(current_where, current_lines, include_frames=include_frames)
        if section == "FAILURES":
            r.failures.append(entry)
        elif section == "ERRORS":
            r.errors.append(entry)
        current_where = None
        current_lines = []

    for ln in lines:
        if ln.strip() == FAILURES_SECTION:
            flush()
            section = "FAILURES"
            continue
        if ln.strip() == ERRORS_SECTION:
            flush()
            section = "ERRORS"
            continue

        if section == "FAILURES":
            m = FAIL_HEADER_RE.match(ln)
            if m:
                flush()
                current_where = m.group(1).strip()
                current_lines = []
                continue
            if current_where:
                current_lines.append(ln)
            continue

        if section == "ERRORS":
            m = ERROR_HEADER_RE.match(ln)
            if m:
                flush()
                current_where = m.group(1).strip()
                current_lines = []
                continue
            if current_where:
                current_lines.append(ln)
            continue

    flush()

    # --- parse "Interrupted: N errors during collection" as an ERROR entry ---
    for ln in reversed(lines[-800:]):
        m = COLLECTION_INTERRUPTED_RE.search(ln)
        if m:
            n = int(m.group(1))
            r.errors.append(Entry(where="collection", message=f"{n} errors during collection", text=ln.strip()))
            break

    # --- totals/stats ---
    for ln in reversed(lines[-300:]):
        m = TOTALS_RE.match(ln.strip())
        if m:
            failed = int(m.group(1)) if m.group(1) else 0
            passed = int(m.group(2)) if m.group(2) else 0
            skipped = int(m.group(3)) if m.group(3) else 0
            warnings = int(m.group(4)) if m.group(4) else 0
            seconds = float(m.group(5))

            r.stats.totals = {
                "failed": failed,
                "passed": passed,
                "skipped": skipped,
                "warnings": warnings,
                "seconds": seconds,
            }
            # also put skipped count into skipped list (count-only entry)
            r.skipped.append(Entry(where="totals", message=str(skipped), text=f"skipped={skipped}"))
            break

    # --- optional per-skip reasons if present ---
    in_short = False
    for ln in lines:
        if SHORT_SUMMARY_HDR_RE.match(ln):
            in_short = True
            continue
        if in_short:
            if ln.strip().startswith("==="):
                in_short = False
                continue
            sm = SKIP_SUMMARY_LINE_RE.match(ln.strip())
            if sm:
                where = sm.group(1).strip()
                reason = (sm.group(2) or "").strip() or None
                r.skipped.append(Entry(where=where, message=reason, text=ln.strip()))

    return r


def print_summary(rep: Report):
    totals = rep.stats.totals or {}
    print("Stats:")
    if rep.stats.conda_command:
        print(f"  Conda: {rep.stats.conda_command}")
    if rep.stats.platform:
        print(f"  Env: {rep.stats.platform} | Python {rep.stats.python_version} | pytest {rep.stats.pytest_version}")
    if rep.stats.collected is not None:
        print(f"  Collected: {rep.stats.collected}")
    if totals:
        print(f"  Totals: {totals.get('failed',0)} failed, {totals.get('passed',0)} passed, "
              f"{totals.get('skipped',0)} skipped, {totals.get('warnings',0)} warnings in {totals.get('seconds',0)}s")

    print()
    print(f"Failures: {len(rep.failures)}")
    for i, e in enumerate(rep.failures[:10], 1):
        msg = f"{e.exception_type}: {e.message}" if e.exception_type else (e.message or "")
        print(f"  {i}. {e.where} -> {msg}")

    print()
    print(f"Errors: {len(rep.errors)}")
    for i, e in enumerate(rep.errors[:10], 1):
        msg = f"{e.exception_type}: {e.message}" if e.exception_type else (e.message or "")
        print(f"  {i}. {e.where} -> {msg}")

    # skipped: show count from totals if available
    sc = None
    if totals:
        sc = totals.get("skipped")
    if sc is not None:
        print()
        print(f"Skipped (count): {sc}")


def main():
    ap = argparse.ArgumentParser(description="Parse pytest/conda logs into JSON with failures/errors/skipped + stats.")
    ap.add_argument("logfile", help="Path to captured stdout/stderr log")
    ap.add_argument("--json-out", default=None, help="Write JSON report to this path (default: stdout)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print a human-readable summary")
    ap.add_argument("--include-frames", action="store_true", help="Include full parsed frames in JSON output")
    args = ap.parse_args()

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    rep = parse_log(text, include_frames=args.include_frames)

    if args.do_print:
        print_summary(rep)

    data = _omit_none(asdict(rep))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
