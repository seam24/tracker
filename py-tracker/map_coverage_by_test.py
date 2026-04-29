#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: map_coverage_by_test.py coverage.json test_to_covered_lines.json"
    )

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

coverage = json.loads(input_path.read_text(encoding="utf-8"))

mapping = defaultdict(lambda: defaultdict(list))

for filename, filedata in coverage.get("files", {}).items():
    contexts_by_line = filedata.get("contexts", {})

    for line_number, contexts in contexts_by_line.items():
        for context in contexts:
            if not context:
                continue

            test_id = context.split("|", 1)[0]
            mapping[test_id][filename].append(int(line_number))

result = {
    test_id: {
        filename: sorted(set(lines))
        for filename, lines in files.items()
    }
    for test_id, files in sorted(mapping.items())
}

output_path.write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
)

print(f"Wrote {output_path}")
print(f"Mapped {len(result)} tests")
