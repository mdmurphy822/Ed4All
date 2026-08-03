"""Screenshot assembled end-user pages + exemplars in light/dark schemes.

RAM GUARD: Chromium typesetting ~2,800 MathJax equations per page can spike
multiple GB — the 2026-07-04 13:09 WSL OOM was this stacked on a live
synthesis run (9GB consumed in <2 min per runtime/state/qa/ram_watch.log). Refuses to
launch under 6 GB MemAvailable and never runs alongside ed4all synthesis.

Usage: python scripts/harness/shoot_pages.py <name=path.html> [...]  (shots land in runtime/shots/)
"""
import subprocess
import sys
from pathlib import Path

avail_kb = int(next(l for l in open("/proc/meminfo") if "MemAvailable" in l).split()[1])
if avail_kb < 6 * 1024 * 1024:
    sys.exit(f"RAM guard: {avail_kb//1024} MB available (<6GB) — refusing chromium launch")
if subprocess.run(["pgrep", "-f", "ed4all run textbook"], capture_output=True).stdout.strip():
    sys.exit("RAM guard: ed4all synthesis running — screenshots serialize after it")

from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "runtime" / "shots"
OUT.mkdir(exist_ok=True)
pages = [a.split("=", 1) for a in sys.argv[1:]]
if not pages:
    sys.exit("usage: shoot_pages.py name=path.html [...]")

with sync_playwright() as pw:
    for scheme in ("light", "dark"):
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 950},
                                  color_scheme=scheme)
        pg = ctx.new_page()
        for name, f in pages:
            pg.goto(f"file://{Path(f).resolve()}")
            try:
                pg.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            pg.wait_for_timeout(3500)
            h = pg.evaluate("document.body.scrollHeight")
            for fr in ([0, 0.18, 0.45, 0.75] if scheme == "light" else [0, 0.45]):
                pg.evaluate(f"window.scrollTo(0, {int(h*fr)})")
                pg.wait_for_timeout(400)
                shot = OUT / f"{name}_{scheme}_{int(fr*100):02d}.png"
                pg.screenshot(path=str(shot))
                print(shot.name)
        ctx.close()
        browser.close()  # fresh browser per scheme bounds peak RSS
