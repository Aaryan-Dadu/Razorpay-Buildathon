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
#: Relative by default so the built site works from any host, and from
#: the filesystem. `cleanUrls` on Vercel serves it at /instrument too.
INSTRUMENT_DEFAULT = "./instrument.html"

TARGETS = {
    "instrument": {
        "template": ROOT / "ui" / "sentinel.template.html",
        "data": ROOT / "reports" / "ui_data.json",
        "out": ROOT / "ui" / "sentinel.html",
        "deploy": ROOT / "web" / "instrument.html",
        "desc": "Try the join a merchant cannot make, run 52 days of "
                "remediation forward, and move the duplicate line until the "
                "false-positive bill overtakes what it recovers.",
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
        "deploy": ROOT / "web" / "index.html",
        "desc": "Four systems pay the same customer back, none of them share "
                "a key, and the second payout leaves unseen.",
        "checks": {
            "lane legend": "Goodwill credit",
            "headline figure": 'class="huge"',
            "figure cards": 'class="fig"',
            "scoreboard": 'class="ours"',
            "threshold grid": 'class="dot',
            "money rail": 'class="r-pos"',
            "scrub copy": "held-out orders",
        },
        "interactions": "landing",
    },
}


#: Written into web/ as well as living at the repo root. Vercel reads
#: vercel.json from whatever is configured as the project's Root Directory,
#: so a project pointed at web/ never sees the root one and silently loses
#: the redirects. Shipping both copies makes the site behave the same under
#: either setting.
SITE_CONFIG = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "cleanUrls": True,
    "trailingSlash": False,
    "redirects": [
        {"source": "/index.html", "destination": "/", "permanent": False},
        {"source": "/instrument.html", "destination": "/instrument",
         "permanent": False},
    ],
    "headers": [{
        "source": "/(.*)",
        "headers": [
            {"key": "X-Content-Type-Options", "value": "nosniff"},
            {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
            {"key": "X-Frame-Options", "value": "SAMEORIGIN"},
            {"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"},
        ],
    }],
}


def write_site_config() -> None:
    out = ROOT / "web" / "vercel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(SITE_CONFIG, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} (root-directory-agnostic)")


def check_internal_links() -> int:
    """Every relative link in the deployed pages must resolve to a real file.

    The landing page linked to ./instrument.html, which exists on disk and
    404s in production, because `cleanUrls` serves /instrument and removes
    the .html form rather than redirecting to it. The file check below would
    not have caught that on its own, so vercel.json now redirects the .html
    forms back, and this asserts the target file is actually there.
    """
    web = ROOT / "web"
    bad = []
    for page in sorted(web.glob("*.html")):
        for m in re.finditer(r'href="\.\/([^"#?]+)"', page.read_text()):
            if not (web / m.group(1)).exists():
                bad.append(f"{page.name} -> ./{m.group(1)}")
    if bad:
        print("broken internal links: " + ", ".join(bad))
        return 1
    print(f"internal links ok -- {len(list(web.glob('*.html')))} pages cross-link cleanly")
    return 0


def build(name: str, spec: dict, instrument_url: str) -> int:
    if not spec["data"].exists():
        print(f"{name}: missing {spec['data']} -- run scripts/export_ui.py first")
        return 1

    tmpl = spec["template"].read_text()
    data = spec["data"].read_text()
    json.loads(data)                       # reject NaN and friends early
    OUT = spec["out"]
    OUT.parent.mkdir(parents=True, exist_ok=True)

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

    # The two copies point at different places on purpose. A published
    # artifact has no sibling file to link to, so it needs the instrument's
    # own URL; the deployed site does have one, and a relative link there
    # survives any host and works straight off the filesystem.
    page = tmpl.replace("__DATA__", data).replace("__INSTRUMENT__", instrument_url)
    deploy_page = tmpl.replace("__DATA__", data).replace("__INSTRUMENT__",
                                                         "./instrument.html")
    if "__DATA__" in page or "__INSTRUMENT__" in page or "__INSTRUMENT__" in deploy_page:
        print(f"{name}: a placeholder survived substitution")
        return 1
    OUT.write_text(page)
    deployed = ""
    if spec.get("deploy"):
        dep = spec["deploy"]
        dep.parent.mkdir(parents=True, exist_ok=True)
        doc = wrap_document(deploy_page, spec.get("desc", ""))
        dep.write_text(doc)
        deployed = f" + {dep.relative_to(ROOT)} ({len(doc):,} bytes)"
    print(f"{name}: syntax ok, wrote {OUT.relative_to(ROOT)} "
          f"({len(page):,} bytes){deployed}")
    rc = smoke_test(page, spec["checks"])
    if rc:
        return rc
    if spec["interactions"] == "landing":
        return scroll_test(page)
    return interaction_test(page) if spec["interactions"] else 0


SCROLL_PROBE = """
// The page throttles scroll work into requestAnimationFrame, and
// IntersectionObserver fires asynchronously too. Reading the DOM in the
// same tick as the scroll therefore measures the previous frame, which
// makes a working page look inert. Each step yields before it asserts.
(function(){
  var L=[], w=document.querySelector(".pin-wrap"), read=document.getElementById("thr-read");
  function t(n,c){ L.push((c?"PASS  ":"FAIL  ")+n); }
  function q(sel){ return document.querySelectorAll(sel).length; }
  function at(frac){
    var travel=w.offsetHeight-window.innerHeight;
    window.scrollTo(0, w.offsetTop + travel*frac);
    window.dispatchEvent(new Event("scroll"));
  }
  var lo, hi, steps=[
    function(){ t("dot grid built", q("#grid .dot")===NEAR_N); at(0); },
    function(){
      lo=parseFloat(read.textContent);
      t("top of the pin gives the lowest line", lo<1.1);
      t("the lowest line reports a net loss",
        document.getElementById("pf-net").classList.contains("loss"));
      t("honest refunds are held there", q("#grid .dot.alarm")>0);
      at(1);
    },
    function(){
      hi=parseFloat(read.textContent);
      t("bottom of the pin raises the line", hi>lo);
      t("nothing is caught at the top of the range", q("#grid .dot.caught")===0);
      t("every duplicate is missed there", q("#grid .dot.missed")===NEAR_DUPS);
      at(0.5);
    },
    function(){
      var mid=parseFloat(read.textContent);
      t("mid-scrub lands between the ends", mid>lo && mid<hi);
      t("mid-scrub catches duplicates", q("#grid .dot.caught")>0);
      t("mid-scrub is back in profit",
        document.getElementById("pf-net").classList.contains("gain"));
      window.scrollTo(0, document.body.scrollHeight);
      window.dispatchEvent(new Event("scroll"));
    },
    function(){
      t("progress rule advances with the page",
        /scale|matrix/.test(getComputedStyle(document.getElementById("rule-progress")).transform));
      t("sections rule themselves in", q("section.in")>0);
      document.getElementById("__out").textContent=L.join("\\n");
    }
  ];
  (function run(i){
    if(i>=steps.length) return;
    steps[i]();
    setTimeout(function(){ run(i+1); }, 180);
  })(0);
})();
"""


def scroll_test(page: str) -> int:
    """Drive the pinned section by scroll position and check it responds.

    A scroll-scrubbed section is the easiest thing on the page to break
    without noticing: the markup renders, the copy reads fine, and the
    control silently does nothing because a rect calculation went to zero.
    So this scrolls to the top, middle and bottom of the pin and asserts the
    threshold, the dot states and the sign of the net all move with it.
    """
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        return 0
    blob = json.loads(TARGETS["landing"]["data"].read_text())
    near = blob.get("near", [])
    script = (SCROLL_PROBE
              .replace("NEAR_DUPS", str(sum(1 for n in near if n[1] == 1)))
              .replace("NEAR_N", str(len(near))))
    probe = ('<!doctype html><meta charset="utf-8"><div id="__err"></div>'
             '<div id="__out"></div><script>addEventListener("error",'
             'function(e){document.getElementById("__err").textContent='
             '"ERR "+e.message;});</script>' + page
             + "<script>" + script + "</script>")
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "scroll.html"
        f.write_text(probe)
        r = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--window-size=1280,900", "--virtual-time-budget=9000",
             "--dump-dom", f"file://{f}"],
            capture_output=True, text=True, timeout=120)
    out = re.search(r'<div id="__out">(.*?)</div>', r.stdout, re.S)
    lines = (out.group(1) if out else "").strip()
    if not lines:
        print("scroll test produced no result")
        return 1
    for line in lines.split("\n"):
        print("  " + line.strip())
    fails = lines.count("FAIL")
    print(f"scroll: {lines.count('PASS')} pass, {fails} fail")
    return 1 if fails else 0


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
    write_site_config()
    return check_internal_links()


#: The Artifact runtime supplies a document shell and a small reset. A file
#: served straight off a static host gets neither, so the deployable copy is
#: wrapped here. The reset is reproduced deliberately rather than assumed:
#: without `body{margin:0}` the full-bleed hero canvas sits inside an 8px
#: gutter on every edge, which is exactly the kind of difference that only
#: shows up in production.
SHELL_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="{desc}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="website">
<link rel="icon" href="data:image/svg+xml,{icon}">
<style>
  *{{box-sizing:border-box}}
  html{{-webkit-text-size-adjust:100%}}
  body{{margin:0}}
  img{{max-width:100%}}
  [hidden]{{display:none!important}}
</style>
{head}
</head>
<body>
{body}
</body>
</html>
"""

ICON = ("%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
        "%3Ctext y='26' font-size='26'%3E%F0%9F%A7%BE%3C/text%3E%3C/svg%3E")


def wrap_document(page: str, desc: str) -> str:
    """Split the artifact-format page into head and body and wrap it.

    Everything up to the end of the last <style> block is head material
    (title, font links, the page's own CSS); the rest is markup.
    """
    cut = page.rfind("</style>")
    if cut == -1:
        head, body = "", page
    else:
        cut += len("</style>")
        head, body = page[:cut], page[cut:]
    m = re.search(r"<title>(.*?)</title>", head, re.S)
    title = m.group(1).strip() if m else "Double-Dip Sentinel"
    return SHELL_HEAD.format(desc=desc, title=title, icon=ICON,
                             head=head.strip(), body=body.strip())


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

  // gate deck
  var gp=document.getElementById("g-prev"), gn=document.getElementById("g-next");
  t("gate opens on the first decision",
    /^1 of /.test(document.getElementById("g-pos").textContent));
  t("cannot step back from the first", gp.disabled===true);
  var first=document.querySelector("#gate .mono").textContent;
  gn.click();
  t("stepping forward changes the decision",
    document.querySelector("#gate .mono").textContent!==first);
  t("the deck says whether the gate was right",
    document.querySelectorAll("#gate .gate-truth").length===1);
  while(!gn.disabled) gn.click();
  t("the deck reaches its last decision", gn.disabled===true);
  t("the deck includes at least one call the gate got wrong",
    GATE_WRONG>0);

  // track detail
  var trk=document.querySelector("#tracks .trk");
  trk.click();
  var det=document.getElementById("detail");
  t("clicking a track opens its history", !det.hidden);
  t("the history names the order", /ord_/.test(det.textContent));
  det.querySelector("#d-close").click();
  t("the history closes again", det.hidden===true);

  // observation horizon
  var hz=document.getElementById("hz");
  var lastNaive=document.getElementById("hz-naive").textContent;
  hz.value=0; hz.dispatchEvent(new Event("input"));
  var earlyNaive=parseFloat(document.getElementById("hz-naive").textContent);
  t("pulling the horizon back lowers the reported rate",
    earlyNaive<parseFloat(lastNaive));
  t("the early horizon is flagged as understated",
    document.getElementById("hz-under").classList.contains("warn"));
  t("almost the whole early book is still inside the window",
    parseFloat(document.getElementById("hz-open").textContent)>90);
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
                                              if o["dup"])))
              .replace("GATE_WRONG", str(sum(
                  1 for g in blob["decisions"]
                  if (g["verdict"] == "block") != g["was_duplicate"]))))
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
