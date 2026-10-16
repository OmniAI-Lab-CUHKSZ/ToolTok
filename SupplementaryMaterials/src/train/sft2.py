# src/train/sft2.py

import torch
import random
import os
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image

from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments
)

# Project imports
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.pathing import bbox_center, build_action_path, clamp_bbox, sample_start_position
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.spatial_cot import format_spatial_cot
from ..tools.visual_utils import draw_cursor
from ..utils.sft_screenspot import ScreenSpotDataManager
from ..utils.qa_data import mix_with_squad
from .action_trainer import WeightedActionTrainer, resolve_gradient_accumulation

# =============================================================================
# 1. Shared concise CoT
# =============================================================================

def generate_cot_for_step(
    cursor_pos, target_bbox, instruction, next_action_token, img_size
):
    return format_spatial_cot(
        task=instruction,
        cursor_pos=cursor_pos,
        target_bbox=target_bbox,
        image_size=img_size,
        action_token=next_action_token,
    )

# =============================================================================
# 3. Dataset & Collator
# =============================================================================

@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images, requires_action = [], [], []
        for f in features:
            msgs = f["messages"]
            fmt = []
            for message in msgs:
                if message["role"] == "user":
                    content = [
                        {"type": "image"},
                        {"type": "text", "text": message["content"]},
                    ]
                    fmt.append({"role": "user", "content": content})
                elif message["role"] == "assistant":
                    fmt.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": message["content"]}],
                        }
                    )
                else:
                    fmt.append(message)
            txt = self.processor.apply_chat_template(fmt, tokenize=False, add_generation_prompt=False)
            texts.append(txt)
            images.append(f["image"])
            requires_action.append(bool(f["requires_action"]))
            
        batch = self.processor(
            text=texts, images=images, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i in range(len(labels)):
            starts = (batch["input_ids"][i] == im_start).nonzero(as_tuple=True)[0]
            if len(starts) >= 2:
                labels[i, :starts[-1] + 2] = -100
        
        batch["labels"] = labels
        batch["requires_action"] = torch.tensor(requires_action, dtype=torch.bool)
        return batch

class ScreenSpotSFTDataset(Dataset):
    def __init__(self):
        self.ss_manager = ScreenSpotDataManager()
        self.data = self.ss_manager.raw_train
        print(f"[SFT-2 Dataset] Train: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._process_sample(self.data[idx])

    def _process_sample(self, sample):
        raw_img = sample['image'].convert("RGB")
        
        # --- Random Resizing Logic ---
        orig_w, orig_h = raw_img.size
        curr_max_dim = max(orig_w, orig_h)
        
        # Decide target size based on probability
        target_max_side = HP.SFT_MAX_IMAGE_SIDE

        # Calculate scale and resize if needed
        scale = 1.0
        if curr_max_dim > target_max_side:
            scale = target_max_side / curr_max_dim
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            raw_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Update width/height for coordinate calculation
        w, h = raw_img.size
        
        # --- BBox & Coordinates ---
        bbox = sample.get('bbox', sample.get('point', [0, 0]))
        if bbox is None or len(bbox) not in (2, 4):
            raise ValueError(f"invalid ScreenSpot bbox: {bbox!r}")
        if len(bbox) == 2: bbox = [bbox[0], bbox[1], bbox[0], bbox[1]]
        
        if all(0.0 <= c <= 1.0 for c in bbox):
            absolute_bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
        else:
            absolute_bbox = [coordinate * scale for coordinate in bbox]
        absolute_bbox = clamp_bbox(absolute_bbox, (w, h))
        tx, ty = bbox_center(absolute_bbox)
        start_x, start_y = sample_start_position((tx, ty), (w, h))

        # --- Path Generation ---
        # Use the randomized start position
        full_path = build_action_path((start_x, start_y), absolute_bbox, (w, h))
        
        # --- Step Selection ---
        if not full_path:
            start_x, start_y = tx, ty
            action_token = "<CLICK_SHORT>"
            history_tokens = []
            curr_pos = (start_x, start_y)
        else:
            if random.random() < 0.2:
                step_idx = len(full_path) - 1
            else:
                step_idx = random.randint(0, max(0, len(full_path) - 2))
            
            target_step = full_path[step_idx]
            action_token = target_step[0]
            
            history_tokens = []
            # Initialize loop from randomized start
            curr_cx, curr_cy = start_x, start_y
            
            for i in range(step_idx):
                h_token, (nx, ny) = full_path[i]
                history_tokens.append(h_token)
                curr_cx, curr_cy = nx, ny
            
            curr_pos = (curr_cx, curr_cy)

        # --- Prompt Formatting ---
        if not history_tokens:
            history_str = "None (Start)"
        else:
            history_str = " -> ".join(history_tokens)
        
        user_content = f"[Action] Task: {sample['instruction']}\nPrevious Actions: {history_str}"

        # --- Vis & CoT ---
        image = draw_cursor(raw_img, curr_pos[0], curr_pos[1])
        
        cot = generate_cot_for_step(
            cursor_pos=curr_pos,
            target_bbox=absolute_bbox,
            instruction=sample['instruction'],
            next_action_token=action_token,
            img_size=(w, h)
        )
        
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}, 
            {"role": "assistant", "content": cot}
        ]
        
        return {"messages": msgs, "image": image, "requires_action": True}

# =============================================================================
# 4. Execution
# =============================================================================

def run_sft_screenspot():
    if not os.path.exists(HP.SFT_2_INPUT_PATH):
        raise FileNotFoundError(f"Stage I checkpoint not found: {HP.SFT_2_INPUT_PATH}")

    print(f"[SFT-2] Loading model from {HP.SFT_2_INPUT_PATH}...")
    # NOTE: device_map="auto" REMOVED for DDP/Accelerate compatibility
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.SFT_2_INPUT_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(HP.SFT_2_INPUT_PATH, trust_remote_code=True)
    
    # Gradient Checkpointing (Save VRAM)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Freeze Visual, Train LLM + Embeddings
    model.get_input_embeddings().weight.requires_grad = True
    for name, param in model.named_parameters():
        if "visual" in name: param.requires_grad = False
        else: param.requires_grad = True
        
    action_train_ds = ScreenSpotSFTDataset()
    train_ds = mix_with_squad(
        action_train_ds,
        HP.QA_DATA_PATH,
        qa_fraction=HP.QA_MIX_RATIO,
        seed=HP.SFT_SEED,
    )
    collator = SFTDataCollator(processor)
    accumulation, world_size, global_batch = resolve_gradient_accumulation(
        HP.SFT_2_BATCH_SIZE
    )
    
    args = TrainingArguments(
        output_dir=HP.SFT_2_OUTPUT_PATH,
        num_train_epochs=HP.SFT_2_EPOCHS,
        per_device_train_batch_size=HP.SFT_2_BATCH_SIZE,
        gradient_accumulation_steps=accumulation,
        learning_rate=HP.SFT_2_LEARN_RATE,
        bf16=True,
        logging_steps=5, # Logs will appear every 5 steps
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        prediction_loss_only=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )
    print(f"[SFT-2] world_size={world_size}, global_batch={global_batch}, grad_accum={accumulation}")
    
    trainer = WeightedActionTrainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=collator, processing_class=processor
    )
    
    print("[SFT-2] Starting Training with Detailed Loss Logging...")
    trainer.train()
    
    trainer.save_model(HP.SFT_2_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_2_OUTPUT_PATH)

if __name__ == "__main__":
    run_sft_screenspot()
