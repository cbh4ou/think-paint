"""The Level-1 agentic painting loop:

generate (LLM writes p5.brush JS) -> render (headless Chromium) -> gate
(anti-cheat) -> critique (VLM sees the PNG, rewrites) -> repeat N rounds ->
score (pairwise win-rate vs the variant's golden pool).

Every stage is also a CLI subcommand so a workflow engine (AWO) can wire the
same steps together itself.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from . import gate as gate_mod
from . import render as render_mod
from .config import Variant, load_variant
from .llm import LLM

REPO = Path(__file__).resolve().parent.parent


def extract_js(text: str | None) -> str:
    if not text:
        return ""
    m = re.findall(r"```(?:javascript|js)?\s*\n(.*?)```", text, flags=re.S)
    return (m[-1] if m else text).strip()


def draw_references(pool: Path, n: int) -> list[str]:
    """Half love / half okay tiers when both exist (weak policies still get
    beatable rivals); otherwise any files."""
    all_refs = sorted(str(p) for p in pool.glob("*.png"))
    love = [r for r in all_refs if Path(r).stem.startswith("love")]
    okay = [r for r in all_refs if not Path(r).stem.startswith("love")]
    if love and okay:
        k = min(n, len(love) + len(okay))
        half = (k + 1) // 2
        return random.sample(love, min(half, len(love))) + random.sample(
            okay, min(k - min(half, len(love)), len(okay))
        )
    return random.sample(all_refs, min(n, len(all_refs)))


def score_png(llm: LLM, v: Variant, png: Path) -> dict:
    """Pairwise win-rate vs the pool: N references, each judged in BOTH
    presentation orders (position-bias control). Score = wins / 2N.
    Tolerates concurrent curation: vanished files just drop that comparison."""
    if not png.exists():
        return {"reward": None, "reason": "candidate missing"}
    refs = [r for r in draw_references(v.golden_pool, v.refs_per_candidate)
            if Path(r).exists()]
    if not refs:
        return {"reward": None, "reason": "golden pool empty"}
    judge_llm = llm.vision()
    wins = 0
    for ref in refs:
        for a_first in (True, False):
            x, y = (str(png), ref) if a_first else (ref, str(png))
            try:
                pick = judge_llm.pairwise(v.judge_question, x, y)
            except FileNotFoundError:
                continue  # file vanished mid-judging (human curating in parallel)
            wins += (pick == "A") == a_first
    return {"reward": round(wins / (2 * len(refs)), 3),
            "wins": wins, "refs": len(refs)}
def cmd_judge(a: argparse.Namespace) -> None:
    v = load_variant(a.variant)
    print(json.dumps(score_png(LLM(), v, a.png)))


def cmd_generate(a: argparse.Namespace) -> None:
    v = load_variant(a.variant)
    feedback = Path(a.feedback_file).read_text() if a.feedback_file else ""
    text = LLM().generate(v.system_prompt(), a.prompt, feedback)
    a.out.write_text(extract_js(text))
    print(json.dumps({"sketch": str(a.out), "chars": a.out.stat().st_size}))


def cmd_render(a: argparse.Namespace) -> None:
    res = render_mod.render(a.sketch, a.out, a.timeout_ms)
    if a.json_out:
        a.json_out.write_text(json.dumps(res))
    print(json.dumps(res))
    if not res["ok"]:
        sys.exit(2)


def cmd_gate(a: argparse.Namespace) -> None:
    v = load_variant(a.variant)
    render_errors: list = []
    if a.render_json and Path(a.render_json).exists():
        render_errors = json.loads(Path(a.render_json).read_text()).get("errors", [])
    res = gate_mod.run_gate(
        a.sketch.read_text(),
        a.png,
        {"allowed": v.allowed, "banned": v.banned, "min_coverage": v.min_coverage},
    )
    if render_errors:
        res["pass"] = False
        res["reasons"] = [f"render: {e}" for e in render_errors] + res["reasons"]
    if not res["pass"] and a.feedback_out:
        a.feedback_out.write_text(
            "Your last sketch failed the automated gate:\n- "
            + "\n- ".join(res["reasons"])
            + "\nFix the code and output the full sketch again."
        )
    print(json.dumps(res))
    if not res["pass"]:
        sys.exit(3)


def cmd_status(a: argparse.Namespace) -> None:
    """Round counter for workflow loops: append a tally, report doneness."""
    log = a.dir / "rounds.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    if log.exists():
        n = len(log.read_text().splitlines())
    log.write_text(log.read_text() + f"round {n}\n" if log.exists() else f"round {n}\n")
    png = a.dir / "round.png"
    print(json.dumps({"done": n + 1 >= a.rounds, "rounds_done": n + 1, "png": str(png.resolve())}))


def cmd_critique(a: argparse.Namespace) -> None:
    v = load_variant(a.variant)
    text = LLM().vision().critique(v.system_prompt(), a.prompt, str(a.png))
    a.out.write_text(text)
    print(json.dumps({"feedback_file": str(a.out), "chars": len(text)}))


def cmd_batch(a: argparse.Namespace) -> None:
    """Pool generation: run the full loop over every prompt in prompts.jsonl.
    Resumable — dirs with a score.json are skipped. Output defaults to
    gallery/<variant>/ so variants never collide."""
    v = load_variant(a.variant)
    prefix = a.out_prefix or Path("gallery") / v.name
    prompts = v.prompts[: a.limit] if a.limit else v.prompts
    total = len(prompts)
    for i, p in enumerate(prompts, 1):
        out = prefix / f"{i:03d}"
        if (out / "score.json").exists():
            print(f"[{i}/{total}] skip {out} (done)")
            continue
        print(f"[{i}/{total}] {p}", flush=True)
        cmd_run(argparse.Namespace(variant=a.variant, prompt=p, rounds=a.rounds,
                                   max_attempts=a.max_attempts, out=out))


def cmd_shortlist(a: argparse.Namespace) -> None:
    """Rank finished runs under a gallery prefix by best judge reward."""
    rows = []
    for f in sorted(a.prefix.rglob("score.json")):
        s = json.loads(f.read_text())
        best = s.get("best", {})
        if best.get("reward") is not None and best.get("reward", -1) >= 0:
            png = Path(best["png"])
            if not png.exists():
                png = f.parent / png.name
            rows.append((best["reward"], str(png), s["prompt"][:48]))
    rows.sort(reverse=True)
    if not rows:
        print(f"no finished runs under {a.prefix}/ yet")
        return
    for rank, (reward, png, prompt) in enumerate(rows, 1):
        print(f"{rank:2d}.  reward {reward:.3f}  {png}   <- {prompt}")
    print("\nstamp keepers: uv run python -m thinkpaint.paint stamp "
          "--variant <config> --png <path> --tier love|okay --name <label>")


def cmd_stamp(a: argparse.Namespace) -> None:
    """Human taste step: copy a painting into the variant's golden pool.
    Tier 'love' (name must start love_) is the top tier; 'okay' is the
    beatable tier the judge mixes in for early-policy signal."""
    v = load_variant(a.variant)
    src = Path(a.png)
    if not src.exists():
        raise SystemExit(f"no such painting: {src}")
    tier = "love" if a.tier == "love" else "okay"
    name = a.name or f"{src.parent.name}-{src.stem}"  # pool/003/round0.png -> 003-round0
    dst = v.golden_pool / f"{tier}_{name}{src.suffix}"
    v.golden_pool.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(src, dst)
    n = len(list(v.golden_pool.glob("*.png")))
    print(f"stamped {dst}  (pool now {n} paintings)")


def cmd_doctor(a: argparse.Namespace) -> None:
    """Readiness check for non-coders: everything needed to paint, train,
    and judge — with a plain-language fix for each failure."""
    import os as _os
    import shutil as _sh
    import tomllib as _toml

    def check(label: str, ok: bool, fix: str) -> None:
        print(f"{'PASS' if ok else 'FAIL':4s}  {label}" + ("" if ok else f"\n      fix: {fix}"))

    check("node (syntax gate)", _sh.which("node") is not None,
          "install Node.js (https://nodejs.org)")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        check("headless chromium (renderer)", True, "")
    except Exception:
        check("headless chromium (renderer)", False,
              "uv run playwright install chromium")

    providers_path = Path(_os.environ.get("THINKPAINT_PROVIDERS")
                          or REPO / "providers.toml")
    try:
        profiles = _toml.loads(providers_path.read_text())
        check(f"providers.toml ({len(profiles)} profiles: {', '.join(sorted(profiles))})",
              True, "")
    except Exception as e:
        check("providers.toml", False, f"fix {providers_path}: {e}")
        profiles = {}

    for name in sorted(profiles):
        p = profiles[name]
        key = p.get("key") or _os.environ.get(p.get("key_env", ""), "")
        check(f"profile [{name}] key ({p.get('key_env', 'inline')})",
              bool(key), f"export ${p.get('key_env', '?')} or add key = \"...\" in {providers_path}")

    for vdir in sorted((REPO / "variants").glob("*/config.yaml")):
        v = load_variant(vdir)
        pool = len(list(v.golden_pool.glob("*.png")))
        gens = vdir.parent / "generators.toml"
        gcount = len(_toml.loads(gens.read_text()).get("generator", [])) if gens.exists() else 0
        check(f"variant {v.name}: {len(v.prompts)} prompts, {gcount} generators, {pool} pool images",
              pool >= 3, "stamp more loves with tools/curate.py (need >= 3 to judge)")

    if a.ping and profiles:
        from thinkpaint.llm import _load_profile

        import urllib.request as _u

        for name in sorted(profiles):
            if name != a.ping and a.ping != "all":
                continue
            try:
                p = _load_profile(name)
                body = json.dumps({"model": p.get("model"),
                                   "messages": [{"role": "user", "content": "Reply: OK"}],
                                   "max_tokens": 512}).encode()
                req = _u.Request(p["base_url"].rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {p.get('key', '')}"})
                with _u.urlopen(req, timeout=60) as r:
                    json.loads(r.read())
                check(f"profile [{name}] endpoint reachable", True, "")
            except Exception as e:
                check(f"profile [{name}] endpoint reachable", False, str(e)[:120])


def cmd_run(a: argparse.Namespace) -> None:
    v = load_variant(a.variant)
    llm = LLM()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    feedback = ""
    best: dict = {"reward": -1.0}
    history: list[dict] = []

    for rnd in range(a.rounds):
        sketch = out / f"round{rnd}.js"
        png = out / f"round{rnd}.png"

        for _attempt in range(a.max_attempts):
            code = extract_js(llm.generate(v.system_prompt(), a.prompt, feedback))
            sketch.write_text(code)
            r = render_mod.render(sketch, png)
            g = gate_mod.run_gate(
                code,
                png if r["ok"] else None,
                {"allowed": v.allowed, "banned": v.banned, "min_coverage": v.min_coverage},
            )
            feedback = (
                "Your last sketch failed the automated gate:\n"
                f"- render errors: {r['errors']}\n- gate: {g['reasons']}\n"
                "Fix the code and output the full sketch again."
            )
            if g["pass"]:
                break
        else:
            history.append({"round": rnd, "result": "gate_failed_after_retries"})
            continue

        if rnd < a.rounds - 1:
            feedback = llm.vision().critique(v.system_prompt(), a.prompt, str(png))
        history.append({"round": rnd, "result": "ok", "coverage": g["coverage"]})

    scored = []
    for rnd in range(a.rounds):
        png = out / f"round{rnd}.png"
        if not png.exists():
            continue
        s = score_png(llm, v, png)
        if s["reward"] is None:
            break
        scored.append({"round": rnd, **s})
        if s["reward"] > best["reward"]:
            best = {
                "round": rnd,
                "reward": s["reward"],
                "png": str(png),
                "sketch": str(out / f"round{rnd}.js"),
            }

    summary = {
        "prompt": a.prompt,
        "variant": v.name,
        "rounds": a.rounds,
        "history": history,
        "scored": scored,
        "best": best,
    }
    (out / "score.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))  # compact: one line, stdio drivers can parse


def main() -> None:
    ap = argparse.ArgumentParser(prog="thinkpaint")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--variant", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--feedback-file", type=Path)
    g.add_argument("--out", type=Path, required=True)
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("render")
    r.add_argument("--sketch", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--timeout-ms", type=int, default=20000)
    r.add_argument("--json-out", type=Path)
    r.set_defaults(fn=cmd_render)

    sl = sub.add_parser("shortlist")
    sl.add_argument("--prefix", type=Path, default=Path("gallery"))
    sl.set_defaults(fn=cmd_shortlist)

    sm = sub.add_parser("stamp")
    sm.add_argument("--variant", type=Path, required=True)
    sm.add_argument("--png", required=True)
    sm.add_argument("--tier", choices=["love", "okay"], default="love")
    sm.add_argument("--name", default=None, help="optional; auto-named from source path")
    sm.set_defaults(fn=cmd_stamp)

    d = sub.add_parser("doctor", help="readiness check (--ping <profile> to live-test)")
    d.add_argument("--ping", default="", help="profile name to live-ping, or 'all'")
    d.set_defaults(fn=cmd_doctor)


    gt = sub.add_parser("gate")
    gt.add_argument("--sketch", type=Path, required=True)
    gt.add_argument("--variant", type=Path, required=True)
    gt.add_argument("--png", type=Path)
    gt.add_argument("--render-json", type=Path)
    gt.add_argument("--feedback-out", type=Path)
    gt.set_defaults(fn=cmd_gate)
    st = sub.add_parser("status")
    st.add_argument("--dir", type=Path, required=True)
    st.add_argument("--rounds", type=int, required=True)
    st.set_defaults(fn=cmd_status)

    b = sub.add_parser("batch")
    b.add_argument("--variant", type=Path, required=True)
    b.add_argument("--rounds", type=int, default=3)
    b.add_argument("--max-attempts", type=int, default=3)
    b.add_argument("--out-prefix", type=Path, default=None,
                   help="default: gallery/pool/<variant>/")
    b.add_argument("--limit", type=int, default=0,
                   help="only the first N prompts (0 = all)")
    b.set_defaults(fn=cmd_batch)


    c = sub.add_parser("critique")
    c.add_argument("--variant", type=Path, required=True)
    c.add_argument("--prompt", required=True)
    c.add_argument("--png", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(fn=cmd_critique)

    j = sub.add_parser("judge")
    j.add_argument("--variant", type=Path, required=True)
    j.add_argument("--png", type=Path, required=True)
    j.set_defaults(fn=cmd_judge)

    run = sub.add_parser("run")
    run.add_argument("--variant", type=Path, required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(fn=cmd_run)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
