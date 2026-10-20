"""LLM transport for the paint loop.

Two interchangeable backends, selected by env/config:
- api:   OpenAI-compatible chat completions (default endpoint: z.ai, model
         glm-5.3-flash — it writes the JS and can see images for judging).
- stdio: prints one JSON request per line to stdout, reads one JSON response
         line from stdin — lets any agent harness drive the loop.

Request/response contract (same for both backends):
  {"type": "generate"|"critique"|"pairwise", ...} -> {"text": str} | {"choice": "A"|"B"}
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-5.3-flash"


def _load_profile(name: str) -> dict:
    """Resolve a provider profile from providers.toml (repo root or
    THINKPAINT_PROVIDERS). Keys may be inline (`key`) or env refs (`key_env`)."""
    import os
    import tomllib
    from pathlib import Path

    path = Path(os.environ.get("THINKPAINT_PROVIDERS")
                or Path(__file__).resolve().parent.parent / "providers.toml")
    if not path.exists():
        raise SystemExit(f"provider profile '{name}' requested but {path} not found")
    profiles = tomllib.loads(path.read_text())
    if name not in profiles:
        raise SystemExit(
            f"no profile '{name}' in {path}. Available: {sorted(profiles)}")
    p = profiles[name]
    if "key_env" in p:
        p = {**p, "key": os.environ.get(p["key_env"], "")}
    return p


def _apply_profile(target: dict, name: str) -> None:
    """Fill missing endpoint fields of `target` (base_url/api_key/model) from
    profile `name`. Explicit THINKPAINT_* env vars always win."""
    p = _load_profile(name)
    if not target.get("base_url"):
        target["base_url"] = p.get("base_url", "")
    if not target.get("api_key"):
        target["api_key"] = p.get("key", "")
    if not target.get("model"):
        target["model"] = p.get("model", DEFAULT_MODEL)
    if not target.get("vision_model"):
        target["vision_model"] = p.get("vision_model") or target.get("model", DEFAULT_MODEL)


class LLM:
    """One transport, two roles: the generator (text model that writes code)
    and the vision model (critique + pairwise judge) may differ. Endpoints
    resolve from explicit THINKPAINT_* env vars first, then a named profile
    in providers.toml (THINKPAINT_PROVIDER / THINKPAINT_VISION_PROVIDER)."""

    def __init__(self) -> None:
        self.mode = os.environ.get("THINKPAINT_LLM", "api")
        cfg = {
            "base_url": os.environ.get("THINKPAINT_BASE_URL", ""),
            "api_key": os.environ.get("THINKPAINT_API_KEY", ""),
            "model": os.environ.get("THINKPAINT_MODEL", ""),
        }
        if os.environ.get("THINKPAINT_PROVIDER"):
            _apply_profile(cfg, os.environ["THINKPAINT_PROVIDER"])
        self.base_url = cfg["base_url"] or DEFAULT_BASE_URL
        self.api_key = cfg["api_key"]
        self.model = cfg["model"] or DEFAULT_MODEL

        vcfg = {
            "base_url": os.environ.get("THINKPAINT_VISION_BASE_URL", ""),
            "api_key": os.environ.get("THINKPAINT_VISION_API_KEY", ""),
            "model": os.environ.get("THINKPAINT_VISION_MODEL", ""),
            "vision_model": os.environ.get("THINKPAINT_VISION_MODEL", ""),
        }
        if os.environ.get("THINKPAINT_VISION_PROVIDER"):
            _apply_profile(vcfg, os.environ["THINKPAINT_VISION_PROVIDER"])
        self.vision_base_url = vcfg["base_url"] or self.base_url
        self.vision_api_key = vcfg["api_key"] or self.api_key
        self.vision_model = vcfg["model"] or vcfg.get("vision_model") or self.model
        if self.mode == "api" and not self.api_key:
            raise SystemExit(
                "THINKPAINT_API_KEY is not set and no provider profile "
 "supplied a key. Export THINKPAINT_API_KEY or set THINKPAINT_PROVIDER."
            )

    def vision(self) -> "LLM":
        """A view of this transport pointing at the vision endpoint."""
        other = LLM.__new__(LLM)
        other.__dict__.update(self.__dict__)
        other.base_url, other.api_key, other.model = (
            self.vision_base_url, self.vision_api_key, self.vision_model)
        return other

    @staticmethod
    def _data_url(png_path: str) -> str:
        with open(png_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()

    # -- backends ---------------------------------------------------------
    def _api_chat(self, system: str, user: str, images: list[str]) -> str:
        content: list[dict] = [{"type": "text", "text": user}]
        for p in images:
            content.append(
                {"type": "image_url", "image_url": {"url": self._data_url(p)}}
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.8,
                "max_tokens": 8192,
                # glm-5.3-flash is a reasoning model; without this the thinking
                # channel eats the token budget and content comes back empty
                "thinking": {"type": "disabled"},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(4):  # upstream rate limits happen at 4-way fan-out
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    m = json.loads(r.read())["choices"][0]["message"]
                    return (m.get("content") or m.get("reasoning_content")
                            or m.get("reasoning") or "")
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")
                if e.code == 429 and attempt < 3:
                    time.sleep(20 * (attempt + 1))
                    continue
                raise SystemExit(
                    f"{self.model} API error {e.code} from {self.base_url}: "
                    f"{detail[:300]}"
                ) from e
        return ""

    def _stdio(self, req: dict) -> dict:
        sys.stdout.write(json.dumps(req) + "\n")
        sys.stdout.flush()
        return json.loads(sys.stdin.readline())

    # -- public API --------------------------------------------------------
    def generate(self, system: str, prompt: str, feedback: str = "") -> str:
        if self.mode == "stdio":
            return self._stdio(
                {"type": "generate", "system": system, "prompt": prompt, "feedback": feedback}
            )["text"]
        user = prompt if not feedback else f"{prompt}\n\nPrevious attempt feedback:\n{feedback}"
        return self._api_chat(system, user, [])

    def critique(self, system: str, prompt: str, png: str) -> str:
        if self.mode == "stdio":
            return self._stdio(
                {"type": "critique", "system": system, "prompt": prompt, "image": png}
            )["text"]
        return self._api_chat(
            system,
            f"Here is what your code rendered for: {prompt}\n\n"
            "Critique the composition and pigment layering in 3-5 concrete bullets, "
            "then rewrite the JavaScript to fix them. Respond with a single "
            "```javascript code block.",
            [png],
        )

    def pairwise(self, question: str, png_a: str, png_b: str) -> str:
        """Ask 'which is better, A or B?' -> 'A' | 'B' (order randomized by caller)."""
        if self.mode == "stdio":
            return self._stdio(
                {"type": "pairwise", "question": question, "image_a": png_a, "image_b": png_b}
            )["choice"]
        text = self._api_chat(
            "You are a precise art judge. Answer with exactly one letter: A or B.",
            f"{question}\n\nImage A is the first image. Image B is the second image. "
            "Which is better, A or B? Reply with exactly: A or B.",
            [png_a, png_b],
        )
        t = text.strip().upper()
        return "A" if t.startswith("A") else "B"
