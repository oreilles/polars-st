"""Print the results of bench_sjoin.py for several builds side by side, as a Markdown table.

Usage: python compare.py LABEL=RESULT.json [LABEL=RESULT.json ...]

The last column says whether every build returned the same pairs for the case.
"""

import json
import sys
from pathlib import Path

builds = [arg.split("=", 1) for arg in sys.argv[1:]]
runs = {
    label: {r["case"]: r for r in json.loads(Path(path).read_text())["results"]}
    for label, path in builds
}
labels = [label for label, _ in builds]
print("| Case | " + " | ".join(labels) + " | Same pairs |")
print("|---|" + "---|" * (len(labels) + 1))
for case in runs[labels[0]]:
    rows = [runs[label].get(case) for label in labels]
    times = " | ".join(f"{r['seconds']:.3f} s" if r else "-" for r in rows)
    same = "yes" if all(r and r["sig"] == rows[0]["sig"] for r in rows) else "**no**"
    print(f"| {case} | {times} | {same} |")
