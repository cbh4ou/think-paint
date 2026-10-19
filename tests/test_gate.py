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
SMOKE = REPO / "gallery" / "smoke" / "hibiscus.js"


def test_allowlisted_sketch_passes() -> None:
    assert run_gate(SMOKE.read_text(), None, WATERCOLOR)["pass"]


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


def test_local_helper_functions_allowed() -> None:
    code = "function petal(x, y) { brush.circle(x, y, 30); }\npetal(5, 5);"
    assert run_gate(code, None, WATERCOLOR)["pass"]


def test_commented_calls_ignored() -> None:
    assert run_gate("// text('hi') and brush.hatch(1, 2)\nbrush.circle(0,0,10);",
                    None, WATERCOLOR)["pass"]


def test_pixel_coverage_detects_blank() -> None:
    from thinkpaint.gate import pixel_coverage

    blank = REPO / "gallery" / "smoke" / "hibiscus.png"
    assert pixel_coverage(blank) > 0.08


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
