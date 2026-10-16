"""ToolTok three-stage training entry point."""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .train.sft import run_sft
from .train.sft2 import run_sft_screenspot
from .train.sft3 import run_sft_screenspot_pro
from .utils.action import ACTION_BASE_EMBEDDING
from .utils.parameters import HYPERPARAMS as HP
from .utils.tokenizer import add_new_tokens, save_model, validate_action_tokenizer


STAGES = ("all", "init", "1", "2", "3")


def validate_checkpoint(path: str) -> None:
    """Validate lightweight ToolTok invariants without loading model weights."""
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    validate_action_tokenizer(processor.tokenizer)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_tied = bool(
        getattr(getattr(config, "text_config", None), "tie_word_embeddings", False)
    )
    if config.tie_word_embeddings or text_tied:
        raise ValueError("tie_word_embeddings is still enabled")


def checkpoint_exists(path: str, phase_name: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        validate_checkpoint(path)
    except Exception as error:
        raise RuntimeError(
            f"{phase_name} checkpoint exists but is invalid: {path}. "
            "Move the incomplete checkpoint aside before retrying."
        ) from error
    print(f"[{phase_name}] Valid checkpoint found at {path}; skipping.")
    return True


def run_initialization() -> None:
    if checkpoint_exists(HP.INIT_MODEL_PATH, "Phase 0"):
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "Phase 0 must run once in a single process. Run "
            "`python -m src --stage init` before Accelerate/DDP training."
        )
    if not os.path.exists(HP.BASE_MODEL_PATH):
        raise FileNotFoundError(f"base model not found: {HP.BASE_MODEL_PATH}")

    print(f"[Phase 0] Loading base model from {HP.BASE_MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(HP.BASE_MODEL_PATH, trust_remote_code=True)
    model, processor = add_new_tokens(model, processor, ACTION_BASE_EMBEDDING)
    save_model(model, processor, HP.INIT_MODEL_PATH)
    print(f"[Phase 0] Complete: {HP.INIT_MODEL_PATH}")


def run_stage_one() -> None:
    if not checkpoint_exists(HP.SFT_OUTPUT_PATH, "Stage I"):
        run_sft()


def run_stage_two() -> None:
    if not checkpoint_exists(HP.SFT_2_OUTPUT_PATH, "Stage II"):
        run_sft_screenspot()


def run_stage_three() -> None:
    if not checkpoint_exists(HP.SFT_3_OUTPUT_PATH, "Stage III"):
        run_sft_screenspot_pro()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ToolTok SFT pipeline")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="run one phase, or the complete three-stage pipeline",
    )
    return parser


def main() -> None:
    stage = build_parser().parse_args().stage
    if stage in ("all", "init"):
        run_initialization()
    if stage in ("all", "1"):
        run_stage_one()
    if stage in ("all", "2"):
        run_stage_two()
    if stage in ("all", "3"):
        run_stage_three()
    print(f"[ToolTok] Requested stage {stage!r} completed.")


if __name__ == "__main__":
    main()
