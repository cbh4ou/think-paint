"""Variant config: one YAML per painting style (subject/medium swap point).

A variant pins the canvas, the 10-function brush allowlist, banned tokens,
gate thresholds, the judge question, and the golden reference pool. The
trainer/loop code never changes between variants.

Vendored stack (matches the original project): p5 1.11.10 + p5.brush 1.1.4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

API_DOC = {
    "scaleBrushes": "brush.scaleBrushes(f)  // scale all default brush params by f",
    "noStroke": "brush.noStroke()  // disable stroke for subsequent shapes",
    "fill": 'brush.fill("#d96b52", 120)  // fill color + opacity 0-255 for subsequent shapes',
    "noFill": "brush.noFill()  // disable fill",
    "bleed": "brush.bleed(0.4)  // watercolor edge bleed strength, 0 to 0.5",
    "fillTexture": "brush.fillTexture(0.6, 0.4)  // paper-grain texture 0-1, edge-darkening 0-1",
    "beginShape": "brush.beginShape()  // start custom polygon",
    "vertex": "brush.vertex(x, y)  // add vertex (inside beginShape/endShape)",
    "endShape": "brush.endShape(CLOSE)  // finish shape; CLOSE joins last to first vertex",
    "circle": "brush.circle(x, y, radius)  // filled circle with current fill",
    "polygon": "brush.polygon([[x,y],[x,y],...])  // filled polygon",
    "pick": 'brush.pick("cpencil")  // switch brush: 2B HB 2H cpencil pen rotring spray marker marker2 charcoal',
    "set": 'brush.set("rotring", "#1f4b99", 0.8)  // pick brush + color + weight, activates stroke mode',
    "stroke": 'brush.stroke("#4a2511")  // stroke color (activates stroke mode)',
    "strokeWeight": "brush.strokeWeight(1.5)  // stroke weight multiplier",
    "line": "brush.line(x1, y1, x2, y2)",
    "flowLine": "brush.flowLine(x, y, length, dirDegrees)  // stroke following the active vector field",
    "spline": "brush.spline([[x,y],[x,y],...], 0.5)  // smooth path through points",
    "rect": "brush.rect(x, y, w, h, CENTER)  // stroked/filled rectangle",
    "hatch": "brush.hatch(dist, angle, {rand: 0.1})  // hatch subsequent shapes' interiors",
    "setHatch": 'brush.setHatch("pen", "#333333", 0.5)  // brush/color/weight used for hatching',
}


@dataclass
class Variant:
    name: str
    medium: str
    canvas: list[int]
    allowed: list[str]
    banned: list[str]
    min_coverage: float
    judge_question: str
    refs_per_candidate: int
    style_hint: str
    dir: Path
    prompts: list[str] = field(default_factory=list)

    @property
    def golden_pool(self) -> Path:
        return self.dir / "golden_pool"

    def system_prompt(self) -> str:
        api = "\n".join(f"- {API_DOC[m]}" for m in self.allowed if m in API_DOC)
        w, h = self.canvas
        return f"""You are a generative painter. You write one p5.brush JavaScript
sketch that renders a single {self.medium}. {self.style_hint}

Canvas: createCanvas({w}, {h}, WEBGL). Origin (0,0) is the CENTER of the
canvas; x grows right, y grows down. Keep all paint inside the canvas.

Structure (mandatory):
function setup() {{
  createCanvas({w}, {h}, WEBGL);
  background("#f8f5f0"); // warm paper
  brush.load();
  // ... all painting happens here, directly ...
}}
function draw() {{}}

You may ONLY call these brush.* methods:
{api}

Any other p5 drawing (text, image, ellipse, triangle, drawingContext, raw
fill/stroke/shape calls) is BANNED and will be rejected. brush has NO
translate/rotate — wrap shapes in bare p5 globals push(); translate(x,y);
rotate(a); ... pop(); instead. Math helpers (random, sin, cos, map, lerp,
PI...) and your own helper functions are fine.

Craft rules:
- Work in layers: pale background wash first, then large shapes, mid forms,
  finally small accents. Overlap translucent fills so colors mix and deepen
  like real pigment.
- Vary geometry: rotate/translate between shapes; never tile identical shapes.
- Compose for the whole canvas: the subject fills most of the page with
  breathing room; asymmetric balance beats a centered stamp.
- Stay under ~150 lines. No loop with more than 200 iterations.

Respond with a single ```javascript code block and nothing else."""


def load_variant(path: Path) -> Variant:
    raw = yaml.safe_load(path.read_text())
    d = path.parent
    prompts: list[str] = []
    pf = d / "prompts.jsonl"
    if pf.exists():
        prompts = [
            json.loads(line)["prompt"]
            for line in pf.read_text().splitlines()
            if line.strip()
        ]
    return Variant(
        name=raw["name"],
        medium=raw["medium"],
        canvas=raw.get("canvas", [800, 1000]),
        allowed=raw["allowed"],
        banned=raw.get("banned", []),
        min_coverage=raw.get("gate", {}).get("min_coverage", 0.08),
        judge_question=raw["judge"]["question"],
        refs_per_candidate=raw.get("judge", {}).get("refs_per_candidate", 3),
        style_hint=raw.get("style_hint", ""),
        dir=d,
        prompts=prompts,
    )
