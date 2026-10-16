"""Stage III SFT on ScreenSpot-Pro trajectories."""

from __future__ import annotations

import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrainingArguments

from ..tools.visual_utils import draw_cursor
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.pathing import bbox_center, build_action_path, clamp_bbox, sample_start_position
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.sft_screenspot_pro import ScreenSpotDataManager
from ..utils.qa_data import mix_with_squad
from .action_trainer import WeightedActionTrainer, resolve_gradient_accumulation
from .sft2 import SFTDataCollator, generate_cot_for_step


class ScreenSpotProSFTDataset(Dataset):
    def __init__(self):
        self.manager = ScreenSpotDataManager()
        self.data = self.manager.raw_train
        print(f"[SFT-3 Dataset] Train: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._process_sample(self.data[index])

    def _process_sample(self, sample):
        image_value = sample.get("image", sample.get("image_path"))
        try:
            if isinstance(image_value, Image.Image):
                image = image_value.convert("RGB")
            else:
                image = Image.open(image_value).convert("RGB")
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError("cannot load ScreenSpot-Pro sample image") from error

        original_width, original_height = image.size
        scale = min(
            1.0,
            HP.SFT_MAX_IMAGE_SIDE / max(original_width, original_height),
        )
        if scale < 1.0:
            image = image.resize(
                (int(original_width * scale), int(original_height * scale)),
                Image.Resampling.LANCZOS,
            )
        width, height = image.size

        raw_bbox = sample.get("bbox")
        if raw_bbox is None or len(raw_bbox) != 4:
            raise ValueError(f"invalid ScreenSpot-Pro bbox: {raw_bbox!r}")
        if all(0.0 <= coordinate <= 1.0 for coordinate in raw_bbox):
            absolute_bbox = [
                raw_bbox[0] * width,
                raw_bbox[1] * height,
                raw_bbox[2] * width,
                raw_bbox[3] * height,
            ]
        else:
            absolute_bbox = [coordinate * scale for coordinate in raw_bbox]
        absolute_bbox = clamp_bbox(absolute_bbox, (width, height))
        target_center = bbox_center(absolute_bbox)

        start_pos = sample_start_position(target_center, (width, height))
        full_path = build_action_path(start_pos, absolute_bbox, (width, height))
        if not full_path:
            # A tiny target can be unreachable on the fixed movement lattice. Use a
            # valid on-target click instead of teaching an incorrect off-box click.
            start_pos = target_center
            full_path = [("<CLICK_SHORT>", start_pos)]

        if random.random() < 0.33:
            step_index = len(full_path) - 1
        else:
            step_index = random.randint(0, max(0, len(full_path) - 2))

        action_token = full_path[step_index][0]
        history_tokens = [token for token, _ in full_path[:step_index]]
        current_pos = start_pos if step_index == 0 else full_path[step_index - 1][1]
        history = " -> ".join(history_tokens) if history_tokens else "None (Start)"
        user_content = (
            f"[Action] Task: {sample['instruction']}\nPrevious Actions: {history}"
        )
        response = generate_cot_for_step(
            cursor_pos=current_pos,
            target_bbox=absolute_bbox,
            instruction=sample["instruction"],
            next_action_token=action_token,
            img_size=(width, height),
        )
        return {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ],
            "image": draw_cursor(image, current_pos[0], current_pos[1]),
            "requires_action": True,
        }


def run_sft_screenspot_pro():
    input_path = HP.SFT_2_OUTPUT_PATH
    output_path = HP.SFT_3_OUTPUT_PATH
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Stage II checkpoint not found: {input_path}")

    print(f"[SFT-3] Loading Stage II model from {input_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        input_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(input_path, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "visual" not in name

    action_train_dataset = ScreenSpotProSFTDataset()
    train_dataset = mix_with_squad(
        action_train_dataset,
        HP.QA_DATA_PATH,
        qa_fraction=HP.QA_MIX_RATIO,
        seed=HP.SFT_SEED,
    )
    collator = SFTDataCollator(processor)
    accumulation, world_size, global_batch = resolve_gradient_accumulation(
        HP.SFT_3_BATCH_SIZE
    )
    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=HP.SFT_3_EPOCHS,
        per_device_train_batch_size=HP.SFT_3_BATCH_SIZE,
        gradient_accumulation_steps=accumulation,
        learning_rate=HP.SFT_3_LEARN_RATE,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )
    print(
        f"[SFT-3] world_size={world_size}, global_batch={global_batch}, "
        f"grad_accum={accumulation}"
    )
    trainer = WeightedActionTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    print(f"[SFT-3] Training complete: {output_path}")


if __name__ == "__main__":
    run_sft_screenspot_pro()
