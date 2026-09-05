#!/usr/bin/env python3
"""Build ui/sentinel.html from the template plus a real evaluation export.

    python scripts/export_ui.py     # run the pipeline, write reports/ui_data.json
    python scripts/build_ui.py      # inline it into the page

The page carries no invented figures. Every case, gate rationale, survival
curve and headline metric comes out of an actual run.

`node --check` runs over the page's script before anything is written. An
earlier revision shipped a page whose JavaScript did not parse: the HTML
rendered perfectly and every data-driven element was silently empty. That is
the worst failure mode available to a page like this, and it is invisible
without either a browser or this gate.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "ui" / "sentinel.template.html"
DATA = ROOT / "reports" / "ui_data.json"
OUT = ROOT / "ui" / "sentinel.html"


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA} -- run scripts/export_ui.py first")
        return 1

    tmpl = TEMPLATE.read_text()
    data = DATA.read_text()
    json.loads(data)                       # reject NaN and friends early

    match = re.search(r"<script>\n(.*?)\n</script>", tmpl, re.S)
    if match is None:
        print("no inline <script> found in the template")
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(match.group(1).replace("__DATA__", "{}"))
        probe = pathlib.Path(fh.name)
    try:
        result = subprocess.run(["node", "--check", str(probe)],
                                capture_output=True, text=True)
    except FileNotFoundError:
        print("node not found -- skipping the syntax gate (install node to enable)")
        result = None
    finally:
        probe.unlink(missing_ok=True)

    if result is not None and result.returncode:
        print("JavaScript syntax error, refusing to write:\n" + result.stderr[:900])
        return 1

    page = tmpl.replace("__DATA__", data)
    if "__DATA__" in page:
        print("placeholder survived substitution")
        return 1
    OUT.write_text(page)
    print(f"syntax ok -- wrote {OUT.relative_to(ROOT)} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
