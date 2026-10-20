"""Self-check for the anti-cheat gate (the reward-hack firewall).

Run: uv run python -m tests.test_gate  (or uv run pytest tests/)
"""

from __future__ import annotations

from pathlib import Path

from thinkpaint.gate import run_gate, syntax_ok

REPO = Path(__file__).resolve().parent.parent
WATERCOLOR = {
    "allowed": ["fill", "bleed", "polygon", "circle", "pick", "stroke",
                "strokeWeight", "line", "flowLine", "spline"],
    "banned": ["text(", "image(", "drawingContext"],
    "min_coverage": 0.08,
}
SKETCH = """function setup() {
  createCanvas(800, 1000, WEBGL);
  background("#f8f5f0");
  brush.load();
  brush.fill("#e8b4a6", 110);
  brush.bleed(0.4);
  brush.polygon([[-380, -480], [330, -430], [360, 430], [-350, 470]]);
  brush.pick("cpencil");
  brush.stroke("#4a2511");
  brush.strokeWeight(1.2);
  brush.line(0, 0, 90, 40);
}
function draw() {}"""


def test_allowlisted_sketch_passes() -> None:
    assert run_gate(SKETCH, None, WATERCOLOR)["pass"]


def test_syntax_error_fails() -> None:
    ok, err = syntax_ok("function setup() {")
    assert not ok and err


def test_banned_token_fails() -> None:
    res = run_gate('text("Beautiful Watercolor Hibiscus");', None, WATERCOLOR)
    assert not res["pass"] and any("banned" in r for r in res["reasons"])


def test_off_allowlist_brush_method_fails() -> None:
    res = run_gate("brush.hatch(5, 45);", None, WATERCOLOR)
    assert not res["pass"] and any("hatch" in r for r in res["reasons"])


def test_hallucinated_brush_method_fails() -> None:
    res = run_gate("brush.translate(10, 20);", None, WATERCOLOR)
    assert not res["pass"]


def test_raw_p5_shapes_fail() -> None:
    res = run_gate("ellipse(0, 0, 50, 60);", None, WATERCOLOR)
    assert not res["pass"] and any("ellipse" in r for r in res["reasons"])

def test_math_helpers_allowed() -> None:
    code = ("randomSeed(7); let a = random(0, 10); let n = noise(a, 3); "
            "let m = map(n, 0, 1, 5, 50); let c = constrain(m, 0, 100); "
            "let l = lerp(c, 200, 0.5); let col = color('#d96b52'); "
            "brush.fill(col, 120); brush.circle(a, l, m);")
    assert run_gate(code, None, WATERCOLOR)["pass"]


def test_local_helper_functions_allowed() -> None:
    code = "function petal(x, y) { brush.circle(x, y, 30); }\npetal(5, 5);"
    assert run_gate(code, None, WATERCOLOR)["pass"]


def test_commented_calls_ignored() -> None:
    assert run_gate("// text('hi') and brush.hatch(1, 2)\nbrush.circle(0,0,10);",
                    None, WATERCOLOR)["pass"]


def test_pixel_coverage_detects_blank() -> None:
    from PIL import Image

    from thinkpaint.gate import pixel_coverage

    blank = REPO / "gallery" / ".test-blank.png"
    Image.new("RGB", (80, 100), "#ffffff").save(blank)
    try:
        assert pixel_coverage(blank) < 0.02
    finally:
        blank.unlink(missing_ok=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
