"""The reward function artifact: grade one generated sketch exactly the way
the agentic loop does, but return a single scalar for GRPO.

Signal mix (Surya's rubric, HPSv3 dropped, pairwise re-weighted):
  0.05 * gate (syntax + allowlist + render + pixel coverage, binary)
  0.05 * length (concise code; ramps to 0 past max_tokens)
  0.90 * pairwise win-rate vs the variant's golden pool (VLM judge)

A completion that fails the gate scores 0.0 and never pays for a judge call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import gate as gate_mod
from . import render as render_mod
from .config import Variant
from .llm import LLM
from .paint import extract_js, score_png

# ~4 chars per token is the standard heuristic; the exact tokenizer doesn't
# matter for a shaping signal.
CHARS_PER_TOKEN = 4


@dataclass
class RewardScorer:
    variant: Variant
    llm: LLM
    weights: tuple[float, float, float] = (0.05, 0.05, 0.90)
    max_tokens: int = 1500
    log_file: Path | None = None

    def _length_score(self, code: str) -> float:
        tokens = len(code) / CHARS_PER_TOKEN
        if tokens <= self.max_tokens:
            return 1.0
        return max(0.0, 1.0 - (tokens - self.max_tokens) / 1000.0)

    def score(self, prompt: str, completion: str, tmp_dir: Path | None = None) -> dict:
        t0 = time.time()
        out: dict = {"reward": 0.0, "gate_pass": False, "length": 0.0,
                     "pairwise": None, "coverage": None, "reasons": []}
        code = extract_js(completion)

        ok, err = gate_mod.syntax_ok(code)
        if not ok:
            out["reasons"].append(f"syntax: {err}")
        if ok:
            ok2, problems = gate_mod.api_ok(code, self.variant.allowed, self.variant.banned)
            out["reasons"].extend(problems)
            ok = ok2
        if ok:
            tmp = tmp_dir or Path("/tmp/thinkpaint-reward")
            tmp.mkdir(parents=True, exist_ok=True)
            png = tmp / f"r_{int(time.time() * 1000) % 10**10}.png"
            r = render_mod.render_code(code, png)
            # infra failures (timeout, renderer crash) return reward None so the
            # rollout is EXCLUDED from the group; model-caused failures score 0.0
            infra = [e for e in r["errors"] if e.startswith("render:")]
            if infra:
                out["reward"] = None
                out["reasons"].extend(infra[:2])
            elif not r["ok"]:
                out["reasons"].extend(r["errors"][:2])  # pageerror: model's bug
            else:
                res = gate_mod.run_gate(
                    code, png,
                    {"allowed": self.variant.allowed, "banned": self.variant.banned,
                     "min_coverage": self.variant.min_coverage},
                )
                out["coverage"] = res["coverage"]
                if res["pass"]:
                    out["gate_pass"] = True
                    try:
                        s = score_png(self.llm, self.variant, png)
                    except Exception as e:  # judge unreachable -> not the model's fault
                        out["reward"] = None
                        out["reasons"].append(f"judge: {e.__class__.__name__}")
                        s = {}
                    out["pairwise"] = s.get("reward")
                    if out["gate_pass"] and out["pairwise"] is None and out["reward"] == 0.0:
                        out["reasons"].append(s.get("reason", "no pool"))
                else:
                    out["reasons"].extend(res["reasons"])
            png.unlink(missing_ok=True)
            png.with_suffix(".js").unlink(missing_ok=True)
        if out["reward"] is None:
            out["seconds"] = round(time.time() - t0, 1)
            self._log(prompt, out)
            return out

        w_gate, w_len, w_pair = self.weights
        if out["gate_pass"]:
            out["length"] = self._length_score(code)
            pair = out["pairwise"] if out["pairwise"] is not None else 0.0
            out["reward"] = round(w_gate + w_len * out["length"] + w_pair * pair, 4)
        out["seconds"] = round(time.time() - t0, 1)
        self._log(prompt, out)
        return out

    def _log(self, prompt: str, out: dict) -> None:
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a") as f:
                f.write(json.dumps({"prompt": prompt[:80], **out}) + "\n")
