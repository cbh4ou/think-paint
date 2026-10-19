"""Level 3 inference: generate a sketch with a trained think-paint adapter.

Runs in the training venv (needs torch/peft):

  uv run --project training python paint_adapter.py \
    --adapter adapters/watercolor-grpo/final --prompt "a peach hibiscus" \
    --out runs/adapter.js

Then render + gate it with the normal CLI:
  uv run python -m thinkpaint.paint render --sketch runs/adapter.js --out runs/adapter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkpaint.config import load_variant
from thinkpaint.paint import extract_js


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=Path, default=Path("variants/watercolor/config.yaml"))
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/adapter.js"))
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    a = ap.parse_args()

    v = load_variant(a.variant)
    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    messages = [
        {"role": "system", "content": v.system_prompt()},
        {"role": "user", "content": f"Paint: {a.prompt}. Respond with a single "
                                    "```javascript code block."},
    ]
    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc.input_ids.to(model.device)
    out = model.generate(
        input_ids=ids, max_new_tokens=a.max_new_tokens, temperature=0.8,
        top_p=0.95, do_sample=True,
    )
    completion = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(extract_js(completion))
    print(f"sketch written: {a.out} ({a.out.stat().st_size} chars)")


if __name__ == "__main__":
    main()
