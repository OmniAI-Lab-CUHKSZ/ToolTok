# src/eval/baseline_vlm.py

import base64
import os
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Optional
from openai import OpenAI

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action_logic import MOVE_SPECS, get_action_type, get_move_delta
from ..utils.action_format import extract_action_response
from ..utils.prompts import BASELINE_API_PROMPT
from ..tools.runner import Runner, AgentTrajectory, GTTrajectory

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# =============================================================================
# API RUNNER CLASS
# =============================================================================

class BaselineAPIRunner(Runner):
    """
    Runner for External VLM API (e.g., Qwen/GPT-4o) adapting the robust logic
    from the trained agent's runner.py.
    """
    def __init__(self, temperature: float = 0.0, model_name: Optional[str] = None):
        # Initialize API Client
        self.model_name = model_name or HP.BASELINE_MODEL_NAME
        print(f"[BaselineAPIRunner] Initializing API Client...")
        print(f" - Base URL: {HP.VLM_BASE_URL}")
        print(f" - Model: {self.model_name}")
        
        self.client = OpenAI(
            api_key=HP.VLM_API_KEY,
            base_url=HP.VLM_BASE_URL
        )
        self.temperature = temperature

    def parse_api_response(self, content: str) -> Tuple[str, Optional[str]]:
        """
        Extracts Reasoning and Action Token from raw API response.
        """
        return extract_action_response(content.strip())

    def run_trajectory(
        self,
        input_text: str,
        ground_truth_data: List[dict],
        max_steps: int = 10
    ) -> Tuple[AgentTrajectory, None, None]:
        
        # 1. Init State
        gt_traj = GTTrajectory(ground_truth_data)

        # [Guard] Check for empty GT data
        if gt_traj.total_steps == 0:
            print("[Error] Ground Truth data is empty.")
            # Return a dummy failed trajectory
            dummy = AgentTrajectory(input_text, None, 0)
            dummy.failed = 4
            return dummy, None, None

        agent_traj = AgentTrajectory(input_text, gt_traj.get_step(0), total_gt_steps=gt_traj.total_steps)
        
        print(f"\n=== API TASK: {input_text} ===")

        # === ACTION LOOP ===
        for step in range(max_steps):
            agent_traj.step_count += 1
            
            # [Guard] Safe GT retrieval
            current_gt = gt_traj.get_step(agent_traj.current_gt_step_idx)
            if current_gt is None:
                print(" -> [Fail] GT Index out of bounds (Agent continued after end).")
                agent_traj.failed = 4
                break

            # 2. Prepare View
            current_view_with_cursor = agent_traj.get_current_view()
            base64_img = encode_image_to_base64(current_view_with_cursor)

            # [Debug] Save monitoring image
            try:
                os.makedirs("./results", exist_ok=True) 
                current_view_with_cursor.save("./results/current.png")
            except Exception: pass
            
            # 3. Build Prompt
            prompt_text = (
                f"Instruction: {agent_traj.global_question}\n"
                "Return one <think>...</think> block followed by one action token."
            )
            
            messages = [
                {"role": "system", "content": BASELINE_API_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            # 4. Call API
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=512   
                )
                response_text = completion.choices[0].message.content
            except Exception as e:
                print(f"[API Error] Step {step}: {e}")
                agent_traj.failed = 1
                break
            
            # 5. Parse
            reasoning, token_text = self.parse_api_response(response_text)
            
            print(f"\n[Step {step+1}]")
            print(f"  Thinking: {reasoning[:100]}...")
            
            if not token_text:
                print(f"  Action:   [INVALID] {response_text[:20]}...")
                agent_traj.failed = 1
                agent_traj.tools.append(f"INVALID: {response_text}")
                break
            
            print(f"  Action:   {token_text}")
            agent_traj.tools.append(token_text)

            # 6. Execute Logic (Strictly following runner.py)
            action_type = get_action_type(token_text)
            
            # --- Move ---
            if action_type == "move":
                if token_text in MOVE_SPECS:
                    dx, dy = get_move_delta(token_text, current_view_with_cursor.size)
                    agent_traj.move_cursor(dx, dy)
                else:
                    agent_traj.failed = 1
                    break
            
            # --- Interact (Click, Scroll, Text, Nav) ---
            elif action_type in ["click", "scroll", "text", "nav"]:
                # Check Type Consistency
                if action_type != current_gt.action_type:
                    print(f" -> [Fail] Wrong Type (Expected {current_gt.action_type})")
                    agent_traj.failed = 2
                    break

                # Check Hit
                if self.check_hit(agent_traj.cursor_pos, current_gt.bbox, current_view_with_cursor.size):
                    print(" -> Hit! Target reached.")
                    
                    # Logic for Multi-step vs Single-step
                    if agent_traj.current_gt_step_idx < gt_traj.total_steps - 1:
                        # Advance to next target
                        next_step = gt_traj.get_step(agent_traj.current_gt_step_idx + 1)
                        agent_traj.advance_to_next_gt_step(next_step)
                    else:
                        # Final target hit
                        agent_traj.current_gt_step_idx += 1
                        agent_traj.gt_steps_passed += 1
                        print(" -> Task Success.")
                        agent_traj.failed = 0
                        break # [CRITICAL] Break loop immediately on success
                else:
                    print(" -> Miss! Position off.")
                    agent_traj.failed = 3
                    break
            
            # --- End ---
            elif action_type == "end":
                if agent_traj.current_gt_step_idx >= gt_traj.total_steps:
                    print(" -> Task Success.")
                    agent_traj.failed = 0
                else:
                    print(" -> Premature Stop.")
                    agent_traj.failed = 4
                break
            
            else:
                agent_traj.failed = 1
                break

        # Timeout / Loop finish check
        if agent_traj.failed == 0 and agent_traj.step_count >= max_steps:
             # Ensure we actually finished the GT steps
             if agent_traj.current_gt_step_idx < gt_traj.total_steps:
                 print(f" [Runner] Max steps ({max_steps}) reached. Timeout.")
                 agent_traj.failed = 4

        return agent_traj, None, None
