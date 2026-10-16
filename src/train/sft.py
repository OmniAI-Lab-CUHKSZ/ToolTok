# src/train/sft.py

import json
import torch
import os
import random
import math
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments
)
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import List, Dict, Any

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import atomic_action_token_ids
from ..utils.pathing import build_action_path, point_in_bbox
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.qa_data import mix_with_squad
from ..utils.spatial_cot import format_simple_action_cot, format_spatial_cot
from .action_trainer import (
    WeightedActionTrainer,
    resolve_gradient_accumulation,
)

# =============================================================================
# 1. Shared concise CoT and action labeling
# =============================================================================

def get_spatial_label_with_cot(
    cursor_pos, target_bbox, intent, image_size, task
):
    """Choose one valid action and render the shared three-part concise CoT."""
    if intent["type"] == "nav":
        if intent["mode"] == "back":
            return format_simple_action_cot(
                task,
                "This is a system-navigation request rather than an on-screen target.",
                "I should return to the previous view.",
                "<GO_BACK>",
            )
        return format_simple_action_cot(
            task,
            "This is a system-navigation request rather than an on-screen target.",
            "I should return to the home screen.",
            "<GO_HOME>",
        )

    if point_in_bbox(cursor_pos, target_bbox):
        if intent["type"] == "click":
            return format_spatial_cot(
                task, cursor_pos, target_bbox, image_size, "<CLICK_SHORT>"
            )
        if intent["type"] == "long_click":
            return format_simple_action_cot(
                task,
                "The cursor is inside the target box.",
                "The requested interaction is a long press.",
                "<CLICK_LONG>",
            )
        if intent["type"] == "text":
            return format_simple_action_cot(
                task,
                "The cursor is inside the input target.",
                "I should begin text entry.",
                "<TEXT_START>",
            )
        return format_simple_action_cot(
            task,
            "The cursor has reached the requested target.",
            "No further movement is needed.",
            "<END_ACTION>",
        )

    path = build_action_path(cursor_pos, target_bbox, image_size)
    if not path or not path[0][0].startswith("<MOVE_"):
        raise RuntimeError(
            f"cannot construct a movement label from {cursor_pos} to {target_bbox}"
        )
    return format_spatial_cot(
        task, cursor_pos, target_bbox, image_size, path[0][0]
    )

# =============================================================================
# 2. Dataset
# =============================================================================
class SFTDataset(Dataset):
    def __init__(self, data_path):
        # 1. Load Semantic Data
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Stage I data not found: {data_path}. Run "
                "`python -m src.utils.generate_data` first."
            )
        with open(data_path, 'r', encoding='utf-8') as f:
            self.jsonl_data = [json.loads(line) for line in f]
        
        self.num_spatial = HP.SFT_SPATIAL_SAMPLES
        self.total_len = len(self.jsonl_data) + self.num_spatial
        
        print(f"[Dataset] Total Samples: {self.total_len}")
        
        self.ui_names = ["Submit", "Cancel", "Search", "Menu", "Settings", "Back", "Profile", "Login", "Options", "Button", "Application", "Game", "Function"]
        self.text_contents = ["hello", "test", "123456", "user", "admin"]
        
        self.prompts_click = [
            "Click {name}", "Select {name}", "Tap on {name}", "Hit the {name} button", "Choose {name}"
        ]
        self.prompts_move = [
            "Move to {name}", "Hover over {name}", "Point at {name}", "Go to {name}", "Find {name}"
        ]
        self.prompts_text = [
            "Type '{content}'", "Enter '{content}' ", "Fill {name} with '{content}'", "Input '{content}'"
        ]
        self.prompts_back = [
            "Go back", "Return to previous page", "Back", "Navigate back", "Backspace", "Backwards"
        ]
        self.prompts_home = [
            "Go home", "Return to main screen", "Home", "Exit to desktop", "Quit", "Homepage"
        ]

    def get_random_background(self):
        shade = random.randint(20, 235)
        return Image.new(
            "RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (shade, shade, shade)
        )

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # === TYPE 1: SEMANTIC (JSONL) ===
        if idx < len(self.jsonl_data):
            item = self.jsonl_data[idx]
            data_type = item.get("data_type", "action")
            if data_type == "explanation":
                image = Image.new('RGB', (HP.IMAGE_SIZE, HP.IMAGE_SIZE), color=(0, 0, 0))
            else:
                random_pixels = np.random.randint(0, 255, (HP.IMAGE_SIZE, HP.IMAGE_SIZE, 3), dtype=np.uint8)
                image = Image.fromarray(random_pixels)
            msgs = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + item["messages"]
            return {"messages": msgs, "image": image, "requires_action": True}

        # === TYPE 2: SPATIAL & NAVIGATION (Synthetic) ===
        else:
            image = self.get_random_background()
            draw = ImageDraw.Draw(image)
            margin = 50
            
            sample_rng = random.random()
            
            tx = random.randint(margin, HP.IMAGE_SIZE - margin)
            ty = random.randint(margin, HP.IMAGE_SIZE - margin)
            target_name = random.choice(self.ui_names)
            
            # CASE A: Moving (70%)
            if sample_rng < 0.7:
                while True:
                    cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                    cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                    if math.hypot(cx-tx, cy-ty) > 50: break
                
                sub_type = random.choice(["click", "move", "text"])
                if sub_type == "click":
                    intent = {"type": "click", "name": target_name}
                    instr_tmpl = random.choice(self.prompts_click)
                elif sub_type == "move":
                    intent = {"type": "move", "name": target_name}
                    instr_tmpl = random.choice(self.prompts_move)
                else:
                    content = random.choice(self.text_contents)
                    intent = {"type": "text", "name": target_name, "content": content}
                    instr_tmpl = random.choice(self.prompts_text)
                
                instr = instr_tmpl.format(name=target_name, content=intent.get("content", ""))

            # CASE B: Direct Clicking (15%)
            elif sample_rng < 0.85:
                cx = tx + random.randint(-20, 20)
                cy = ty + random.randint(-20, 20)
                intent = {"type": "click", "name": target_name}
                instr = random.choice(self.prompts_click).format(name=target_name)

            # CASE C: Other Tasks (15%)
            else:
                if random.random() < 0.5:
                    # Text Ready
                    cx = tx + random.randint(-10, 10)
                    cy = ty + random.randint(-10, 10)
                    content = random.choice(self.text_contents)
                    intent = {"type": "text", "name": target_name, "content": content}
                    instr = random.choice(self.prompts_text).format(name=target_name, content=content)
                else:
                    # Navigation
                    intent_mode = random.choice(["back", "home"])
                    intent = {"type": "nav", "mode": intent_mode, "name": "System"}
                    cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                    cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                    tx, ty = 0, 0
                    
                    if intent_mode == "back": instr = random.choice(self.prompts_back)
                    else: instr = random.choice(self.prompts_home)

            # Draw
            if intent.get("type") != "nav":
                draw.rectangle([tx-30, ty-20, tx+30, ty+20], outline=random.choice(("green","black","blue","yellow","red")), width=3)
                try: draw.text((tx-20, ty-35), target_name, fill=random.choice(("green","black","blue","yellow","red")))
                except: pass

            for _ in range(random.randint(2, 5)):
                dx_rand = random.randint(margin, HP.IMAGE_SIZE - margin)
                dy_rand = random.randint(margin, HP.IMAGE_SIZE - margin)
                if abs(dx_rand - tx) > 60 or abs(dy_rand - ty) > 60:
                    draw.rectangle([dx_rand-30, dy_rand-20, dx_rand+30, dy_rand+20], outline=random.choice(("green","black","blue","yellow","red")), width=3)

            from ..tools.visual_utils import draw_cursor
            image = draw_cursor(image, cx, cy)

            target_bbox = (tx - 30, ty - 20, tx + 30, ty + 20)
            cot_response = get_spatial_label_with_cot(
                (cx, cy), target_bbox, intent, image.size, instr
            )
            
            msgs = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"[Action] {instr}"},
                {"role": "assistant", "content": cot_response}
            ]
            
            return {"messages": msgs, "image": image, "requires_action": True}

# =============================================================================
# 3. Collator
# =============================================================================
@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        requires_action = []
        
        for feature in features:
            original_messages = feature["messages"]
            image = feature["image"]
            
            formatted_messages = []
            for m in original_messages:
                if m["role"] == "user":
                    content = [
                        {"type": "image"}, 
                        {"type": "text", "text": m["content"]}
                    ]
                    formatted_messages.append({"role": "user", "content": content})
                else:
                    formatted_messages.append(m)
            
            text = self.processor.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(image)
            requires_action.append(bool(feature["requires_action"]))
            
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=8192, 
            return_tensors="pt"
        )
        
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        for i in range(len(input_ids)):
            start_indices = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
            if len(start_indices) > 0:
                last_turn_idx = start_indices[-1]
                labels[i, :last_turn_idx + 2] = -100
            
        batch["labels"] = labels
        batch["requires_action"] = torch.tensor(requires_action, dtype=torch.bool)
        return batch

# =============================================================================
# 4. Setup & Run
# =============================================================================

def setup_model_for_sft(model, processor):
    print("[Model Setup] Configuring Surgical Fine-Tuning strategy...")

    action_token_ids = list(atomic_action_token_ids(processor.tokenizer).values())
    
    print(f"[Model Setup] Targeted Special Tokens: {len(action_token_ids)} tokens")
    
    model.enable_input_require_grads() 
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True

    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "visual" in name:
            param.requires_grad = False
        elif "embed_tokens" in name or "wte" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = True 
            trainable_params += param.numel()

    print(f"[Model Setup] Total Params: {total_params:,}")
    print(f"[Model Setup] Trainable Params (Approx): {trainable_params:,}")
    print("[Model Setup] Strategy: Vision(Frozen) + Embeds(Train) + LLM(Train)")
    
def run_sft():
    print(f"[SFT] Loading from {HP.INIT_MODEL_PATH}")
    accumulation, world_size, global_batch = resolve_gradient_accumulation(HP.SFT_BATCH_SIZE)
    load_kwargs = {}
    if world_size == 1:
        load_kwargs["device_map"] = "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.INIT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(HP.INIT_MODEL_PATH, trust_remote_code=True)
    
    setup_model_for_sft(model, processor)
    
    primary_dataset = SFTDataset(HP.SFT_DATA_PATH)
    train_dataset = mix_with_squad(
        primary_dataset,
        HP.QA_DATA_PATH,
        qa_fraction=HP.QA_MIX_RATIO,
        seed=HP.SFT_SEED,
    )
    collator = SFTDataCollator(processor=processor)
    
    args = TrainingArguments(
        output_dir=HP.SFT_OUTPUT_PATH,
        num_train_epochs=HP.SFT_EPOCHS,
        per_device_train_batch_size=HP.SFT_BATCH_SIZE,
        gradient_accumulation_steps=accumulation,
        learning_rate=HP.SFT_LEARN_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )
    print(f"[SFT] world_size={world_size}, global_batch={global_batch}, grad_accum={accumulation}")
    
    trainer = WeightedActionTrainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset, 
        data_collator=collator,
        processing_class=processor
    )
    
    print("[SFT] Starting Training with Action-Weighted Loss...")
    trainer.train()
    
    trainer.save_model(HP.SFT_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_OUTPUT_PATH)

if __name__ == "__main__":
    run_sft()
