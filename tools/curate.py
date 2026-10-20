"""Level-2 curation UI: a local sorting viewer for stamping the golden pool.

    uv run python tools/curate.py            # http://localhost:8416

Keys: 1 = love, 2 = okay, 3 = skip, arrows = navigate. Decisions persist in
gallery/.curation.json; stamps land in variants/<variant>/golden_pool/ with
the same auto-naming as `paint.py stamp`. Candidate images are discovered
recursively under gallery/ (score.json rewards shown when present).
"""

from __future__ import annotations

import json
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GALLERY = REPO / "gallery"
DECISIONS = GALLERY / ".curation.json"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>think-paint curate</title>
<style>
 body{margin:0;font-family:system-ui;background:#181818;color:#eee;display:flex;flex-direction:column;height:100vh}
 header{display:flex;gap:16px;padding:10px 16px;background:#222;align-items:center}
 header b{color:#ffd}
 #stage{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#111}
 #stage img{max-width:100%;max-height:100%;object-fit:contain}
 kbd{background:#333;border:1px solid #555;border-radius:4px;padding:2px 8px}
 button.v{font-size:16px;padding:10px 22px;border:0;border-radius:8px;cursor:pointer}
 .love{background:#7a2c2c}.okay{background:#4a4a26}.skip{background:#333}
 button.v.active{outline:3px solid #ffd;outline-offset:-3px}
 select{background:#333;color:#eee;border:1px solid #555;padding:4px 8px}
</style></head><body>
<header><b>think-paint curate</b>
 <select id="variant"></select>
 <span id="meta"></span><span id="pool" style="margin-left:auto"></span></header>
<div id="stage"><img id="img"></div>
<footer>
 <button class="v love" id="b-love" onclick="stamp('love')"><kbd>1</kbd> Love</button>
 <button class="v okay" id="b-okay" onclick="stamp('okay')"><kbd>2</kbd> Okay</button>
 <button class="v skip" id="b-skip" onclick="stamp('skip')"><kbd>3</kbd> Skip</button>
 <button class="v skip" onclick="stamp('undo')"><kbd>Z</kbd> Undo</button>
 <button class="v skip" onclick="prev()">&#9664;</button>
 <button class="v skip" onclick="next()">&#9654;</button>
</footer>
<script>
let imgs=[], i=0, variant='watercolor';
async function api(p, body){const r=await fetch('/api/'+p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json()}
async function load(){
  const d=await fetch('/api/images?variant='+variant).then(r=>r.json());
  imgs=d.images; i=0; render();
  document.getElementById('pool').textContent=`pool: ${d.pool_love} love / ${d.pool_okay} okay`;
}
function render(){
  if(!imgs.length){document.getElementById('stage').innerHTML='<p style="color:#888">no candidate images found</p>';return}
  const it=imgs[i];
  document.getElementById('img').src='/img/'+it.path;
  const badge=it.decision?` [${it.decision.toUpperCase()}]`:'';
  document.getElementById('meta').textContent=
    `${i+1}/${imgs.length}${badge}  r=${it.reward==null?'-':it.reward.toFixed(3)}  ${it.prompt||''}`;
  ['love','okay','skip'].forEach(t=>{
    document.getElementById('b-'+t).classList.toggle('active', it.decision===t)});
}
async function stamp(tier){
  if(!imgs.length)return;
  const it=imgs[i];
  await api('stamp',{png:it.path,tier,variant});
  it.decision = tier==='skip'||tier==='undo' ? '' : tier;
  if(tier==='undo'){load_pool();render()}else{next(); load_pool()}
}
async function load_pool(){
  const d=await fetch('/api/images?variant='+variant).then(r=>r.json());
  document.getElementById('pool').textContent=`pool: ${d.pool_love} love / ${d.pool_okay} okay`;
}
function next(){i=(i+1)%imgs.length;render()}
function prev(){i=(i-1+imgs.length)%imgs.length;render()}
document.addEventListener('keydown',e=>{
  if(e.target && e.target.tagName==='SELECT')return;  // arrows flip the dropdown
  if(['1','2','3','z','Z','ArrowRight','ArrowLeft'].includes(e.key))e.preventDefault();
  if(e.key==='1')stamp('love');else if(e.key==='2')stamp('okay');
  else if(e.key==='3')stamp('skip');else if(e.key==='z'||e.key==='Z')stamp('undo');
  else if(e.key==='ArrowRight')next();
  else if(e.key==='ArrowLeft')prev();});
(async()=>{
  const vs=await fetch('/api/variants').then(r=>r.json());
  const sel=document.getElementById('variant');
  vs.forEach(v=>{const o=document.createElement('option');o.value=o.textContent=v;sel.appendChild(o)});
  variant=vs[0]||'watercolor';
  sel.onchange=()=>{variant=sel.value;sel.blur();load()};
  load();
})();
</script></body></html>"""


def variants() -> list[str]:
    return sorted(p.parent.name for p in (REPO / "variants").glob("*/config.yaml"))


def candidates(variant: str) -> list[dict]:
    """Images that belong to ONE variant: under gallery/<variant>/, or whose
    score.json names that variant. Everything else (other variants' runs,
    smoke tests, contact sheets) is excluded."""
    out = []
    decisions = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    for png in sorted(GALLERY.rglob("*.png")):
        rel = png.relative_to(REPO)
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "gallery":
            continue
        if "contact-sheet" in png.stem or "golden_pool" in parts:
            continue
        if "candidates" in parts or "smoke" in parts:
            continue
        if parts[1] == variant:
            run_variant = variant
        else:
            score = png.parent / "score.json"
            if not score.exists():
                continue
            try:
                run_variant = json.loads(score.read_text()).get("variant")
            except json.JSONDecodeError:
                continue
            if run_variant != variant:
                continue
        reward, prompt = None, ""
        score = png.parent / "score.json"
        if score.exists():
            try:
                s = json.loads(score.read_text())
                best = s.get("best", {})
                reward = best.get("reward")
                if reward is not None and reward < 0:
                    reward = None
                prompt = s.get("prompt", "")[:40]
            except json.JSONDecodeError:
                pass
        out.append({"path": str(rel), "reward": reward, "prompt": prompt,
                    "decision": decisions.get(str(rel), "")})
    return out


def pool_copies(png: Path, variant: str) -> list[Path]:
    """Both possible auto-named copies of a source painting in a pool."""
    pool = REPO / "variants" / variant / "golden_pool"
    stem = f"{{tier}}_{png.parent.name}-{png.stem}{png.suffix}"
    return [pool / stem.format(tier="love"), pool / stem.format(tier="okay")]



def pool_counts(variant: str) -> tuple[int, int]:
    pool = REPO / "variants" / variant / "golden_pool"
    love = okay = 0
    for p in pool.glob("*.png"):
        (love, okay) = (love + 1, okay) if p.stem.startswith("love") else (love, okay + 1)
    return love, okay


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode(), "text/html")
        elif self.path.startswith("/img/"):
            f = REPO / self.path[5:]
            if f.exists():
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, b"{}")
        elif self.path.startswith("/api/images"):
            variant = self.path.split("variant=")[-1].split("&")[0] or "watercolor"
            love, okay = pool_counts(variant)
            self._send(200, json.dumps(
                {"images": candidates(variant), "pool_love": love,
                 "pool_okay": okay}).encode())
        elif self.path.startswith("/api/variants"):
            self._send(200, json.dumps(variants()).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        png = REPO / str(req.get("png", ""))
        tier = req.get("tier", "skip")
        variant = req.get("variant", "watercolor")
        decisions = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
        decisions[str(req.get("png", ""))] = tier
        DECISIONS.write_text(json.dumps(decisions, indent=1))
        if tier == "undo":
            for copy in pool_copies(png, variant):
                copy.unlink(missing_ok=True)
            decisions.pop(str(req.get("png", "")), None)
            DECISIONS.write_text(json.dumps(decisions, indent=1))
            love, okay = pool_counts(variant)
            self._send(200, json.dumps({"undone": True, "pool_love": love,
                                        "pool_okay": okay}).encode())
        elif tier in ("love", "okay") and png.exists():
            # re-stamping replaces: drop the other tier's copy of this source
            for copy in pool_copies(png, variant):
                copy.unlink(missing_ok=True)
            pool = REPO / "variants" / variant / "golden_pool"
            pool.mkdir(parents=True, exist_ok=True)
            dst = pool / f"{tier}_{png.parent.name}-{png.stem}{png.suffix}"
            shutil.copy(png, dst)
            love, okay = pool_counts(variant)
            self._send(200, json.dumps({"stamped": str(dst.relative_to(REPO)),
                                        "pool_love": love, "pool_okay": okay}).encode())
        else:
            self._send(200, json.dumps({"skipped": True}).encode())

    def log_message(self, *a: object) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8416
    print(f"curate UI: http://localhost:{port}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
