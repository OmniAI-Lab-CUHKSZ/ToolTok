# src/tools/runner.py

import torch
import numpy as np
import os 
from PIL import Image
from typing import List, Dict, Tuple, Any
from io import BytesIO
from transformers import LogitsProcessor, StoppingCriteria, LogitsProcessorList, StoppingCriteriaList

# Project imports
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action_logic import MOVE_SPECS, get_action_type, get_move_delta
from .visual_utils import draw_cursor, visualize_trajectory 
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..utils.action_format import extract_action_response
from ..utils.action import atomic_action_token_ids

# =============================================================================
# CUSTOM GENERATION CONTROLS
# =============================================================================

class ActionConstraintLogitsProcessor(LogitsProcessor):
    """
    After ``</think>``, allow only separators and one valid action token.
    """
    def __init__(self, tokenizer, allowed_ids_list: List[int], trigger_phrase="</think>"):
        self.tokenizer = tokenizer
        self.allowed_ids_tensor = torch.tensor(allowed_ids_list)
        self.trigger_phrase = trigger_phrase
        self.allowed_mask = None 
        self.prompt_length = None

    def reset(self, prompt_length=None):
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
            return scores
        current_seq = input_ids[0, self.prompt_length:]
        decoded_text = self.tokenizer.decode(current_seq, skip_special_tokens=False)
        
        if self.trigger_phrase in decoded_text:
            if self.allowed_mask is None or self.allowed_mask.device != scores.device:
                self.allowed_mask = torch.full((scores.shape[-1],), float("-inf"), device=scores.device)
                self.allowed_mask[self.allowed_ids_tensor.to(scores.device)] = 0.0
            
            scores[:, :] += self.allowed_mask
            
        return scores

class ActionStoppingCriteria(StoppingCriteria):
    """
    Stop on an action token only after the generated ``</think>`` delimiter.
    """
    def __init__(self, tokenizer, stop_token_ids: List[int]):
        self.tokenizer = tokenizer
        self.stop_token_ids = set(stop_token_ids)
        self.prompt_length = 0

    def reset(self, prompt_length: int):
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = input_ids[0, -1].item()
        if last_token not in self.stop_token_ids:
            return False
        generated = self.tokenizer.decode(
            input_ids[0, self.prompt_length:], skip_special_tokens=False
        )
        return "</think>" in generated

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class GTStep:
    def __init__(self, image: Image.Image, bbox: List[float], instruction: str, action_type: str):
        self.image = image
        self.bbox = bbox
        self.instruction = instruction
        self.action_type = action_type

class GTTrajectory:
    def __init__(self, gt_data: List[Dict]):
        self.steps = []
        for item in gt_data:
            img_data = item['image']
            if isinstance(img_data, dict):
                if 'bytes' in img_data and img_data['bytes'] is not None:
                    image = Image.open(BytesIO(img_data['bytes'])).convert("RGB")
                else:
                    image = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
            elif isinstance(img_data, Image.Image):
                image = img_data
            elif isinstance(img_data, np.ndarray):
                image = Image.fromarray(img_data).convert("RGB")
            else:
                image = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
            self.steps.append(GTStep(image, item['bbox'], item['instruction'], item['action_type']))
        self.total_steps = len(self.steps)
    def get_step(self, idx: int) -> GTStep:
        return self.steps[idx] if idx < self.total_steps else None

class AgentTrajectory:
    def __init__(self, input_text: str, start_gt_step: GTStep, total_gt_steps: int = 1):
        self.global_question = input_text
        self.total_gt_steps = total_gt_steps 
        self.current_gt_step_idx = 0
        self.current_base_image = start_gt_step.image.copy() 
        self.target_bbox = start_gt_step.bbox 
        w, h = self.current_base_image.size
        self.cursor_pos = [w // 2, h // 2]
        self.failed = 0 
        self.gt_steps_passed = 0
        self.step_count = 0
        self.tools = []       
        self.images = []      
        self.cursor_path = [tuple(self.cursor_pos)]
        
        self.action_history_tokens = [] 
        
        self._update_visual_history()

    def _update_visual_history(self):
        clean_copy = self.current_base_image.copy()
        viz_img = draw_cursor(clean_copy, int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        self.images.append(viz_img)

    def get_current_view(self) -> Image.Image: return self.images[-1]

    def move_cursor(self, dx, dy):
        w, h = self.current_base_image.size
        cx, cy = self.cursor_pos
        nx = max(0, min(cx + dx, w - 1))
        ny = max(0, min(cy + dy, h - 1))
        self.cursor_pos = [nx, ny]
        self.cursor_path.append((nx, ny))
        self._update_visual_history()

    def advance_to_next_gt_step(self, next_gt_step: GTStep):
        self.current_gt_step_idx += 1
        self.gt_steps_passed += 1
        self.current_base_image = next_gt_step.image.copy()
        self.target_bbox = next_gt_step.bbox
        w, h = self.current_base_image.size
        self.cursor_pos = [w // 2, h // 2]
        self.cursor_path.append((w // 2, h // 2))
        
        # [OPTIONAL] Reset history on new page/step? 
        # Usually for SFT we keep it continuous within a task, 
        # or reset if it's a completely new visual context. 
        # Here we keep it to simulate memory.
        
        self._update_visual_history()

# =============================================================================
# RUNNER (Main Execution Logic)
# =============================================================================

class Runner:
    def __init__(self):
        print(f"[Runner] Initialized. Using optimized model.generate()")

    def check_hit(self, cursor_pos, bbox, img_size):
        cx, cy = cursor_pos
        w, h = img_size
        x1, y1, x2, y2 = bbox
        # Scale normalized bbox if needed
        if all(0.0 <= c <= 1.0 for c in [x1, y1, x2, y2]):
            x1 *= w; x2 *= w; y1 *= h; y2 *= h
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def run_trajectory(
        self,
        model,
        processor,
        input_text: str,
        ground_truth_data: List[Dict],
        max_steps: int = 10,
        temperature: float = 0.0 
    ) -> Tuple[Any, List[int], List[float]]:
        
        # 1. Initialize State
        gt_traj = GTTrajectory(ground_truth_data)
        agent_traj = AgentTrajectory(input_text, gt_traj.get_step(0), total_gt_steps=gt_traj.total_steps)
        all_token_ids = []
        all_logprobs = []
        
        # 2. Setup Constraints
        action_id_map = atomic_action_token_ids(processor.tokenizer)
        valid_action_ids = set(action_id_map.values())
        separator_ids = set(processor.tokenizer.encode("\n", add_special_tokens=False))
        separator_ids.update(processor.tokenizer.encode(" ", add_special_tokens=False))
        allowed_ids_list = list(valid_action_ids | separator_ids)
        
        # 3. Setup Processors
        logits_processor = LogitsProcessorList([
            ActionConstraintLogitsProcessor(processor.tokenizer, allowed_ids_list)
        ])
        stop_criteria = StoppingCriteriaList([
            ActionStoppingCriteria(processor.tokenizer, list(valid_action_ids))
        ])

        # === ACTION LOOP ===
        try: 
            for step in range(max_steps):
                agent_traj.step_count += 1
                current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
                curr_img = agent_traj.get_current_view()

                if not agent_traj.action_history_tokens:
                    history_str = "None (Start)"
                else:
                    # e.g. "<MOVE_RIGHT> -> <MOVE_DOWN>"
                    history_str = " -> ".join(agent_traj.action_history_tokens)
                
                # Matches training format from SFT-2/3
                final_user_content = f"[Action] Task: {agent_traj.global_question}\nPrevious Actions: {history_str}"

                # Build Message
                messages = [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": curr_img},
                            {"type": "text", "text": final_user_content}
                        ]
                    }
                ]
                
                # Handle device
                if hasattr(model, "module"):
                    device = model.module.device
                else:
                    device = model.device

                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text_prompt],
                    images=[curr_img],
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                
                # [GENERATION]
                prompt_length = inputs.input_ids.shape[1]
                logits_processor[0].reset(prompt_length)
                stop_criteria[0].reset(prompt_length)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else 1.0,
                        logits_processor=logits_processor,
                        stopping_criteria=stop_criteria,
                        use_cache=False, 
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )

                # Parse Output
                input_len = inputs.input_ids.shape[1]
                generated_ids = outputs.sequences[0][input_len:] 
                generated_text = processor.decode(generated_ids, skip_special_tokens=False)
                
                reasoning_str, token_text = extract_action_response(generated_text)
                
                print(f"\n[Step {step+1}]")
                print(f"  Hist:     {history_str[-50:]}..." if len(history_str) > 50 else f"  Hist:     {history_str}")
                print(f"  Thinking: {reasoning_str[:100]}...")

                clean_gen_text = generated_text.strip()
                step_action_id = None
                step_logprob = 0.0
                
                if token_text and len(generated_ids) > 0:
                    step_action_id = generated_ids[-1].item()
                    expected_id = action_id_map[token_text]
                    if step_action_id != expected_id:
                        token_text = None
                    else:
                        last_step_scores = outputs.scores[-1][0]
                        log_probs = torch.log_softmax(last_step_scores, dim=-1)
                        step_logprob = log_probs[step_action_id].item()

                # === Execute ===
                if token_text and step_action_id is not None:
                    print(f"  Action:   {token_text} (ID: {step_action_id}) Prob: {np.exp(step_logprob):.2%}")
                    
                    agent_traj.tools.append(clean_gen_text)
                    agent_traj.action_history_tokens.append(token_text)
                    all_token_ids.append(step_action_id)
                    all_logprobs.append(step_logprob) 

                    token_action_type = get_action_type(token_text)
                    
                    if token_action_type == "move":
                        if token_text in MOVE_SPECS:
                            dx, dy = get_move_delta(token_text, curr_img.size)
                            agent_traj.move_cursor(dx, dy)
                        else: agent_traj.failed = 1; break
                    
                    elif token_action_type in ["click", "scroll", "text", "nav"]:
                        if token_action_type != current_gt.action_type:
                            print(f"    -> Wrong type (expected {current_gt.action_type}).")
                            agent_traj.failed = 2
                            break
                        
                        if self.check_hit(agent_traj.cursor_pos, current_gt.bbox, curr_img.size):
                            print("    -> Hit! Target reached.")
                            if agent_traj.current_gt_step_idx < gt_traj.total_steps - 1:
                                next_step = gt_traj.get_step(agent_traj.current_gt_step_idx + 1)
                                agent_traj.advance_to_next_gt_step(next_step)
                            else:
                                agent_traj.current_gt_step_idx += 1
                                agent_traj.gt_steps_passed += 1
                                print("    -> Task Success.")
                                agent_traj.failed = 0 
                                break 
                        else:
                            print("    -> Miss! Position off.")
                            agent_traj.failed = 3; break
                    
                    elif token_action_type == "end":
                        if agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                            print("    -> Task Success.")
                            agent_traj.failed = 0
                        else:
                            print("    -> Premature Stop.")
                            agent_traj.failed = 4
                        break
                    else: agent_traj.failed = 1; break
                
                else:
                    print(f"  Action:   [FAILED] {generated_text[-20:]}")
                    agent_traj.failed = 1 
                    agent_traj.tools.append("INVALID")
                    break

                del inputs, outputs, generated_ids
            
            # Loop finished
            else:
                print(f"  [Runner] Max steps ({max_steps}) reached. Timeout.")
                agent_traj.failed = 4

        except BaseException as e:
            print(f"\n[Runner] Exception: {e}")
            torch.cuda.empty_cache()
            agent_traj.failed = 4 

        # [Safety Check]
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             if agent_traj.current_gt_step_idx < gt_traj.total_steps:
                 agent_traj.failed = 4 

        # Visualization
        try:
            os.makedirs("./results", exist_ok=True)
            base_clean = gt_traj.get_step(0).image
            viz_bbox = agent_traj.target_bbox
            w, h = base_clean.size
            if viz_bbox and all(0.0 <= c <= 1.0 for c in viz_bbox):
                viz_bbox = [viz_bbox[0]*w, viz_bbox[1]*h, viz_bbox[2]*w, viz_bbox[3]*h]
            vis_img = visualize_trajectory(
                base_image=base_clean,
                cursor_path=agent_traj.cursor_path,
                actions=agent_traj.tools,
                gt_bbox=viz_bbox,
                success=(agent_traj.failed == 0),
                instruction=agent_traj.global_question
            )
            vis_img.save("./results/current.png")
        except Exception: pass

        return agent_traj, all_token_ids, all_logprobs
