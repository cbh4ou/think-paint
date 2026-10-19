"""Level 3: GRPO training for think-paint variants (QLoRA, RTX 4060 8GB).

Compiles the agentic loop into weights: the model samples sketches, the exact
grading pipeline (syntax -> allowlist -> headless render -> pixel coverage ->
pairwise VLM judge vs the golden pool) scores each sample, and GRPO pushes
probability mass toward the above-average paintings. Base weights stay
frozen; the output is a ~100MB LoRA adapter per variant.

Run from the repo root (thinkpaint imports rely on it):

  cd training && uv sync
  cd .. && THINKPAINT_API_KEY=... uv run --project training python train_grpo.py \
      --variant variants/watercolor/config.yaml

The reward calls the same OpenAI-compatible judge the agentic loop uses
(THINKPAINT_MODEL, default glm-5.3-flash). Rendering happens in-process via
playwright; budget ~7s per completion — it dominates step time
(ponytail: sequential renders; parallelize with a browser pool if it hurts).
"""

from __future__ import annotations

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from thinkpaint.config import load_variant
from thinkpaint.llm import LLM
from thinkpaint.reward import RewardScorer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def build_dataset(variant) -> Dataset:
    rows = []
    for subject in variant.prompts:
        rows.append({
            "prompt": [
                {"role": "system", "content": variant.system_prompt()},
                {"role": "user",
                 "content": f"Paint: {subject}. Respond with a single "
                            "```javascript code block."},
            ]
        })
    if not rows:
        raise SystemExit(f"no prompts found next to {variant.dir / 'prompts.jsonl'}")
    return Dataset.from_list(rows)


def make_reward(scorer: RewardScorer):
    def thinkpaint_reward(prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            text = (completion[0]["content"]
                    if isinstance(completion, list) else str(completion))
            # None (infra failure) excludes the rollout from the group
            rewards.append(scorer.score(prompt="", completion=text)["reward"])
        return rewards

    thinkpaint_reward.__name__ = "thinkpaint_reward"
    return thinkpaint_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=Path, default=Path("variants/watercolor/config.yaml"))
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=None,
                    help="adapter output dir (default adapters/<variant>-grpo)")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--max-completion-length", type=int, default=1536)
    ap.add_argument("--save-steps", type=int, default=50)
    args = ap.parse_args()

    variant = load_variant(args.variant)
    out: Path = args.out or Path("adapters") / f"{variant.name}-grpo"
    out.parent.mkdir(parents=True, exist_ok=True)
    scorer = RewardScorer(
        variant, LLM(), log_file=out / "reward_log.jsonl"
    )

    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",  # linear decay starved mid-run
        warmup_steps=5,
        scale_rewards="none",  # one gate rejection shrank every other advantage
        optim="adamw_8bit",
        bf16=True,
        mask_truncated_completions=True,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=make_reward(scorer),
        args=config,
        train_dataset=build_dataset(variant),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        peft_config=LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05,
            target_modules="all-linear",  # hand lists miss MoE projections
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"reward log: {scorer.log_file}")
    print(f"trained variant artifact saved: {out / 'final'}")


if __name__ == "__main__":
    main()
