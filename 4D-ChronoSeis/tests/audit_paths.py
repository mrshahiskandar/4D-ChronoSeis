"""Audit: after the GUI's run-time substitutions, does any path in the notebook still point
outside the chosen output folder?  Usage: python tests/audit_paths.py <repo_dir> [new_output_folder]
Only INPUT files (your Baseline/Monitor/markers/platform/wells) should be listed."""
import collections, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nbrunner
repo = sys.argv[1]; ws = sys.argv[2] if len(sys.argv) > 2 else "/home/apgce_team_08/chronoseis_output"
here = Path(__file__).resolve().parent.parent
R = nbrunner.Runner(repo, here / "pipeline.json", "/tmp/_w", "/tmp/_o")
nb = R.nbs["4d"]
p = dict(base="<INPUT>/Baseline.sgy", monitor="<INPUT>/Monitor.sgy", markers_csv="<INPUT>/markers.csv",
         platform_csv="<INPUT>/platform.csv", wells_dir="<INPUT>/wells", workdir=ws,
         root_from="/home/apgce_team_08", root_to=ws)
outside = collections.Counter(); log = []
for i, c in enumerate(R._cells("4d")):
    if c["cell_type"] != "code": continue
    src = R._prepare(nb, i, "".join(c["source"]), p, log)
    for m in re.finditer(r'["\'](/[A-Za-z0-9_./\-{}]+)["\']', src):
        if not m.group(1).startswith(ws): outside[(i, m.group(1))] += 1
    for ln in src.splitlines():
        if re.search(r'(to_csv|savefig|segyio\.create|write_text|open)\(\s*["\'][^/"\']', ln): print("RELATIVE WRITE", i, ln.strip()[:100])
for k in sorted(outside): print("outside output folder:", k)
print("done:", len(outside), "paths outside the output folder (should be inputs only)")
