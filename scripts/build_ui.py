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
DATA = ROOT / "reports" / "ui_data.json"

#: Where the landing page's button points. Set by --instrument, else left as
#: a relative link so the file works when opened straight off disk.
INSTRUMENT_DEFAULT = "sentinel.html"

TARGETS = {
    "instrument": {
        "template": ROOT / "ui" / "sentinel.template.html",
        "data": ROOT / "reports" / "ui_data.json",
        "out": ROOT / "ui" / "sentinel.html",
        "checks": {
            "case list": 'class="chip"',
            "lane markers": 'class="mark"',
            "event detail": 'class="ev-amt"',
            "scoreboard": 'class="ours"',
            "cohort rows": "insufficient",
            "puzzle options": 'class="opt"',
            "stream tracks": 'class="trk"',
            "threshold dial": 'id="d-net"',
        },
        "interactions": True,
    },
    "landing": {
        "template": ROOT / "ui" / "landing.template.html",
        "data": ROOT / "reports" / "landing_data.json",
        "out": ROOT / "ui" / "landing.html",
        "checks": {
            "lane legend": "Goodwill credit",
            "headline figure": 'class="huge"',
            "figure cards": 'class="fig"',
            "money band": 'class="b-save"',
            "scoreboard": 'class="ours"',
        },
        "interactions": False,
    },
}


def build(name: str, spec: dict, instrument_url: str) -> int:
    if not spec["data"].exists():
        print(f"{name}: missing {spec['data']} -- run scripts/export_ui.py first")
        return 1

    tmpl = spec["template"].read_text()
    data = spec["data"].read_text()
    json.loads(data)                       # reject NaN and friends early
    OUT = spec["out"]

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

    page = tmpl.replace("__DATA__", data).replace("__INSTRUMENT__", instrument_url)
    if "__DATA__" in page or "__INSTRUMENT__" in page:
        print(f"{name}: a placeholder survived substitution")
        return 1
    OUT.write_text(page)
    print(f"{name}: syntax ok, wrote {OUT.relative_to(ROOT)} ({len(page):,} bytes)")
    rc = smoke_test(page, spec["checks"])
    if rc:
        return rc
    return interaction_test(page) if spec["interactions"] else 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=INSTRUMENT_DEFAULT,
                    help="URL the landing page's button points at")
    ap.add_argument("--only", choices=sorted(TARGETS))
    a = ap.parse_args()
    for name, spec in TARGETS.items():
        if a.only and name != a.only:
            continue
        rc = build(name, spec, a.instrument)
        if rc:
            return rc
    return 0


def smoke_test(page: str, checks: dict) -> int:
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

    missing = [name for name, needle in checks.items() if needle not in dom]
    if missing:
        print("page rendered but these regions are empty: " + ", ".join(missing))
        return 1
    print("render check ok -- every data region populated")
    return 0


INTERACTIONS = """
setTimeout(function(){
  var L=[];
  function t(n,c){ L.push((c?"PASS  ":"FAIL  ")+n); }
  var opts=document.querySelectorAll("#try-opts .opt");
  t("puzzle offers four options", opts.length===4);
  opts[0].click();
  var rev=document.getElementById("try-reveal");
  t("puzzle reveals on answer", !rev.hidden && rev.textContent.length>60);
  t("answering locks the options", opts[0].disabled===true);
  t("exactly one option marked correct",
    document.querySelectorAll("#try-opts .right").length===1);
  t("stream builds every track",
    document.querySelectorAll("#tracks .trk").length===STREAM_ORDERS);
  t("deck opens mid-run rather than empty",
    +document.getElementById("t-ev").textContent > 0);
  var sc=document.getElementById("scrub");
  sc.value=100; sc.dispatchEvent(new Event("input"));
  t("scrubbing to the end settles every event",
    document.getElementById("t-ev").textContent===String(STREAM_EVENTS));
  t("scrubbing to the end catches every duplicate",
    document.getElementById("t-dup").textContent===String(STREAM_DUPS));
  t("each catch raises an alert",
    document.querySelectorAll("#feed .alert").length===STREAM_DUPS);
  t("money stopped is non-zero", document.getElementById("t-money").textContent!=="0");
  sc.value=0; sc.dispatchEvent(new Event("input"));
  t("scrubbing back to zero clears state",
    document.getElementById("t-ev").textContent==="0" &&
    document.querySelectorAll("#tracks .trk.over").length===0);
  var d=document.getElementById("thr");
  d.value=0; d.dispatchEvent(new Event("input"));
  t("the lowest threshold reports a net loss",
    document.getElementById("d-net").classList.contains("loss"));
  d.value=d.max; d.dispatchEvent(new Event("input"));
  t("the highest threshold catches nothing",
    document.getElementById("d-r").textContent==="0.0%");
  document.getElementById("__out").textContent=L.join("\\n");
}, 900);
"""


def interaction_test(page: str) -> int:
    """Drive the page's controls and assert what they did.

    A page can render every region and still have controls that do nothing.
    These run the three interactive pieces the way a visitor would: answer
    the puzzle, scrub the stream to the end and back, push the threshold to
    both extremes.
    """
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        return 0
    blob = json.loads(DATA.read_text())
    script = (INTERACTIONS
              .replace("STREAM_ORDERS", str(len(blob["stream_orders"])))
              .replace("STREAM_EVENTS", str(len(blob["stream"])))
              .replace("STREAM_DUPS", str(sum(1 for o in blob["stream_orders"]
                                              if o["dup"]))))
    probe = ('<!doctype html><meta charset="utf-8"><div id="__err"></div>'
             '<div id="__out"></div><script>addEventListener("error",'
             'function(e){document.getElementById("__err").textContent='
             '"ERR "+e.message;});</script>' + page
             + "<script>" + script + "</script>")
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "itest.html"
        f.write_text(probe)
        r = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=9000", "--dump-dom", f"file://{f}"],
            capture_output=True, text=True, timeout=120)
    out = re.search(r'<div id="__out">(.*?)</div>', r.stdout, re.S)
    lines = (out.group(1) if out else "").strip()
    if not lines:
        print("interaction test produced no result")
        return 1
    for line in lines.split("\n"):
        print("  " + line.strip())
    fails = lines.count("FAIL")
    print(f"interactions: {lines.count('PASS')} pass, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
