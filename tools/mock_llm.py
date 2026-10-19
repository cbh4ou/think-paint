"""Minimal OpenAI-compatible mock for proof-running the think-paint workflow.

Serves POST /v1/chat/completions from stdlib only:
- code-generation/critique requests -> a verified gate-passing sketch,
  variant-aware (watercolor or ink, picked from the system prompt)
- pairwise judge requests -> always "A"

Not a general mock: it exists so `awo composer` proof-runs can execute the
full generate->render->gate->critique->judge->curate loop headlessly and
cheaply. Real runs use THINKPAINT_BASE_URL (default z.ai glm-5.3-flash).
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

GALLERY = Path(__file__).resolve().parent.parent / "gallery"
SKETCH_WATERCOLOR = GALLERY / "poppy-run" / "round1.js"
SKETCH_INK = GALLERY / "smoke" / "pine.js"


def reply(content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode()


def pick_sketch(system: str) -> str:
    path = SKETCH_INK if "ink pen sketch" in system else SKETCH_WATERCOLOR
    return path.read_text() if path.exists() else (
        "function setup() { createCanvas(800, 1000, WEBGL); background('#f8f5f0');"
        " brush.load(); brush.fill('#d96b52', 150); brush.circle(0, 0, 300); }"
        "function draw() {}"
    )


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        system = str(messages[0].get("content", "")) if messages else ""
        user = str(messages[-1].get("content", "")) if messages else ""
        text_user = user if isinstance(user, str) else json.dumps(user)
        if "art judge" in system or "A or B" in text_user:
            content = "A"
        elif "critique" in text_user.lower():
            content = (
                "The rendering reads well; push the mark-making further.\n"
                "```javascript\n" + pick_sketch(system) + "\n```"
            )
        else:
            content = "```javascript\n" + pick_sketch(system) + "\n```"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply(content))

    def log_message(self, *args: object) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8907
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
