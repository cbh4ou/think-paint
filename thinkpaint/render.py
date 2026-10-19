"""Headless render of a p5.brush sketch to PNG.

Writes an HTML page next to the output PNG (sketch inlined, libs via file://
URLs), loads it in headless Chromium, waits for the painting to settle,
screenshots the canvas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO / "templates" / "p5_brush_canvas.html"
DEFAULT_P5 = REPO / "lib" / "p5.min.js"
DEFAULT_BRUSH = REPO / "lib" / "p5.brush.js"
def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def build_page(sketch_js: str, template: Path = DEFAULT_TEMPLATE) -> str:
    html = template.read_text()
    return (
        html.replace("__P5_URL__", _file_url(DEFAULT_P5))
        .replace("__BRUSH_URL__", _file_url(DEFAULT_BRUSH))
        .replace("__SKETCH_JS__", sketch_js)
    )

def render_code(code: str, out_png: Path, timeout_ms: int = 20000) -> dict:
    """Render an in-memory sketch string (used by the reward function)."""
    out_png = out_png.resolve()
    sketch = out_png.with_suffix(".js")
    sketch.write_text(code)
    return render(sketch, out_png, timeout_ms)


def render(sketch: Path, out_png: Path, timeout_ms: int = 20000) -> dict:
    from playwright.sync_api import sync_playwright

    page_html = build_page(sketch.read_text())
    out_png = out_png.resolve()
    tmp_html = out_png.with_suffix(".html")
    tmp_html.write_text(page_html)

    result: dict = {"png": str(out_png), "errors": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 900, "height": 1200})
        page.on("pageerror", lambda e: result["errors"].append(f"pageerror: {e}"))
        try:
            page.goto(tmp_html.as_uri())
            page.wait_for_function("window.__done === true", timeout=timeout_ms)
            cv = page.query_selector("canvas")
            if cv is None:
                result["errors"].append("no canvas element rendered")
            else:
                cv.screenshot(path=str(out_png))
        except Exception as e:  # timeout counts as an error for the gate
            result["errors"].append(f"render: {e.__class__.__name__}: {e}")
        result["errors"].extend(page.evaluate("window.__errors") or [])
        browser.close()
    result["ok"] = bool(result["errors"]) is False and out_png.exists()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout-ms", type=int, default=20000)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(a.sketch, a.out, a.timeout_ms), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
