"""Anti-cheat gate: syntax check, API allowlist, banned tokens, pixel coverage.

Run before any VLM judge call so reward hackers (raw p5 shapes, canvas text,
blank canvases) score 0.0 without paying for a judge.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# p5/JS helpers a sketch may use without going through brush.*.
SAFE_GLOBALS = {
    "createCanvas", "background", "WEBGL", "frameRate", "noLoop", "redraw",
    "push", "pop", "translate", "rotate", "scale", "sin", "cos", "tan", "abs",
    "map", "random", "randomSeed", "noise", "noiseSeed", "constrain", "lerp",
    "color", "red", "green", "blue", "alpha",
    "min", "max", "dist", "floor", "ceil", "round", "print", "console",
    "radians", "degrees", "angleMode", "sq", "sqrt", "pow", "noiseDetail",
    "PI", "TWO_PI", "HALF_PI", "QUARTER_PI", "width", "height",
    "setup", "draw", "windowWidth", "windowHeight",
}

CALL_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*\(")
BRUSH_RE = re.compile(r"\bbrush\s*\.\s*([A-Za-z_$][\w$]*)\s*\(")


KEYWORDS = {
    "for", "while", "if", "else", "switch", "case", "catch", "try", "finally",
    "return", "function", "new", "typeof", "delete", "void", "throw", "await",
    "yield", "class", "super", "do", "in", "of", "instanceof",
}
ALWAYS_ALLOWED_BRUSH = {"load"}  # harness bootstrap, not an art method

def syntax_ok(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["node", "--check", path], capture_output=True, text=True, timeout=30
        )
        return r.returncode == 0, next(
            (ln for ln in r.stderr.splitlines() if ln.strip()
             and not ln.startswith("Node.js")),
            "syntax check failed",
        )
    finally:
        Path(path).unlink(missing_ok=True)


def defined_functions(code: str) -> set[str]:
    names = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", code))
    names.update(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:\(|function|async)", code))
    return names


def api_ok(code: str, allowed: list[str], banned: list[str]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    # strip comments so commented-out calls don't trip the gate
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    local = defined_functions(code)
    for tok in banned:
        if tok in code:
            problems.append(f"banned token: {tok}")
    for m in BRUSH_RE.finditer(code):
        if m.group(1) not in allowed and m.group(1) not in ALWAYS_ALLOWED_BRUSH:
            problems.append(f"brush.{m.group(1)}() not in allowlist")
    for m in CALL_RE.finditer(code):
        name = m.group(1)
        if name in KEYWORDS:
            continue
        start, _ = m.span(1)
        if re.search(r"[.\w$]\s*$", code[:start]):
            continue  # member call: handled by BRUSH_RE / object internals
        if name in SAFE_GLOBALS or name in local:
            continue
        problems.append(f"raw call {name}() not allowed (use brush.*)")
    return not problems, problems


def pixel_coverage(png: Path, tol: int = 18) -> float:
    from PIL import Image

    img = Image.open(png).convert("RGB")
    small = img.resize((160, 200))
    px = list(small.getdata())
    bg = max(set(px), key=px.count)  # dominant color = paper
    different = sum(
        1
        for (r, g, b) in px
        if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) > tol
    )
    return different / len(px)


def run_gate(code: str, png: Path | None, cfg: dict) -> dict:
    allowed: list[str] = cfg.get("allowed", [])
    banned: list[str] = cfg.get("banned", [])
    min_cov: float = cfg.get("min_coverage", 0.08)
    reasons: list[str] = []

    ok, err = syntax_ok(code)
    if not ok:
        reasons.append(f"syntax: {err}")
    ok, problems = api_ok(code, allowed, banned)
    reasons.extend(problems)
    cov = None
    if png is not None and png.exists():
        cov = round(pixel_coverage(png), 4)
        if cov < min_cov:
            reasons.append(f"pixel coverage {cov} < {min_cov}")
    elif png is not None:
        reasons.append("render produced no png")
    return {"pass": not reasons, "reasons": reasons, "coverage": cov}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch", type=Path)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--allowed", nargs="*", default=[])
    ap.add_argument("--banned", nargs="*", default=[])
    ap.add_argument("--min-coverage", type=float, default=0.08)
    ap.add_argument("--config", type=Path, help="variant config yaml")
    a = ap.parse_args()
    cfg = {
        "allowed": a.allowed,
        "banned": a.banned,
        "min_coverage": a.min_coverage,
    }
    if a.config:
        import yaml

        v = yaml.safe_load(a.config.read_text())
        cfg = {
            "allowed": v.get("allowed", []),
            "banned": v.get("banned", []),
            "min_coverage": v.get("gate", {}).get("min_coverage", 0.08),
        }
    print(json.dumps(run_gate(a.sketch.read_text(), a.png, cfg), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
