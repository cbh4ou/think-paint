# Think Paint

Think Paint turns prompts into code-based paintings. An LLM writes
[p5.brush](https://github.com/acamposuribe/p5.brush) JavaScript, Playwright
renders it, safety gates reject invalid output, and a vision model critiques
and scores each painting against a curated reference pool.

Painting styles are reusable variants containing configuration, prompts, and
references. Optional GRPO training produces a small QLoRA adapter per variant
while leaving the base model unchanged.

## Run

Requires Python 3.14+, [`uv`](https://docs.astral.sh/uv/), and an
OpenAI-compatible vision API key.

```bash
uv sync
uv run playwright install chromium
export THINKPAINT_API_KEY=...
uv run python -m thinkpaint.paint run \
  --variant variants/watercolor/config.yaml \
  --prompt "a crimson poppy bending in the wind" \
  --rounds 3 \
  --out runs/poppy
```

Set `THINKPAINT_BASE_URL` and `THINKPAINT_MODEL` to use a different compatible
provider.

## Train a variant

Training uses 4-bit QLoRA and is sized for an 8 GB GPU.

```bash
uv sync --project training
uv run --project training python train_grpo.py \
  --variant variants/watercolor/config.yaml
```
