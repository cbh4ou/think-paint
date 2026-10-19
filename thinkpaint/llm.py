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
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-5.3-flash"


class LLM:
    def __init__(self) -> None:
        self.mode = os.environ.get("THINKPAINT_LLM", "api")
        self.base_url = os.environ.get("THINKPAINT_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = os.environ.get("THINKPAINT_API_KEY", "")
        self.model = os.environ.get("THINKPAINT_MODEL", DEFAULT_MODEL)
        if self.mode == "api" and not self.api_key:
            raise SystemExit(
                "THINKPAINT_API_KEY is not set (export THINKPAINT_API_KEY=...). "
                "Note: environment variable names are case-sensitive."
            )

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
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise SystemExit(
                f"{self.model} API error {e.code} from {self.base_url}: {detail}"
            ) from e

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
