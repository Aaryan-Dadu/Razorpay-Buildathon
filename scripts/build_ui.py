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
import shutil
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
    return smoke_test(page)


def smoke_test(page: str) -> int:
    """Load the page in a real browser and check it actually populated.

    The syntax gate catches a page that will not parse. It cannot catch a
    page that parses and then throws: a timing constant named `T` was
    shadowed by the chart's own `T`, every animation delay became NaN, and
    `render` died partway leaving the event list silently empty. Valid
    syntax, blank output. So the build also renders the page and asserts
    that the data-driven regions have content.
    """
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        print("no chrome found -- skipping the render check")
        return 0

    probe = ('<!doctype html><meta charset="utf-8">'
             '<div id="__err"></div><script>'
             'addEventListener("error",function(e){'
             'document.getElementById("__err").textContent="ERR "+e.message;});'
             '</script>') + page
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "probe.html"
        f.write_text(probe)
        r = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=8000", "--dump-dom", f"file://{f}"],
            capture_output=True, text=True, timeout=120)
    dom = r.stdout

    err = re.search(r'<div id="__err">(.*?)</div>', dom, re.S)
    if err and err.group(1).strip():
        print("runtime error on load: " + err.group(1).strip()[:300])
        return 1

    checks = {
        "case list": 'class="chip"',
        "lane markers": 'class="mark"',
        "event detail": 'class="ev-amt"',
        "scoreboard": 'class="ours"',
        "cohort rows": "insufficient",
    }
    missing = [name for name, needle in checks.items() if needle not in dom]
    if missing:
        print("page rendered but these regions are empty: " + ", ".join(missing))
        return 1
    print("render check ok -- every data region populated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
