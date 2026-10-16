# src/utils/generate_data.py

import json
import random
import os
from .parameters import HYPERPARAMS as HP
from .action_format import format_action_response
from .action_logic import MOVE_SPECS
from .spatial_cot import format_task_opener

# ================= CONFIGURATION =================
OUTPUT_FILE = HP.SFT_DATA_PATH
TOTAL_ACTION_SAMPLES = HP.SFT_ACTION_SAMPLES
TOTAL_EXPLANATION_SAMPLES = HP.SFT_DEFINITION_SAMPLES

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ================= VOCABULARY (Kept same as yours, simplified for brevity) =================
VOCAB = {
    "ui_element": ["button", "icon", "link", "menu", "field", "card", "search bar"],
    "container": ["page", "screen", "window", "list", "panel"],
    "move_verb": ["Move", "Shift", "Position the cursor", "Slide"],
    "click_verb": ["Click", "Tap", "Select", "Hit"],
    "far_desc": ["significantly", "a lot", "far", "by 25% of the screen span"],
    "mid_desc": ["moderately", "somewhat", "a normal distance", "by 10% of the screen span"],
    "clo_desc": ["slightly", "a tiny bit", "a small distance", "by 2% of the screen span"],
    "up": ["up", "upwards"], "down": ["down", "downwards"],
    "left": ["left", "leftwards"], "right": ["right", "rightwards"],
    "up_left": ["up-left", "toward the upper-left"],
    "up_right": ["up-right", "toward the upper-right"],
    "down_left": ["down-left", "toward the lower-left"],
    "down_right": ["down-right", "toward the lower-right"],
    "text_content": ["Hello world", "Text data", "user@example.com", "Related information"],
    "type_verb": ["Type", "Enter", "Input"],
    "end_action": ["Stop task", "Finish interaction", "Done"]
}

# ================= TOKEN META (Same as yours) =================
TOKEN_META = {
    token: {
        "type": "move",
        "dir": token.removeprefix("<MOVE_").removesuffix(">").rsplit("_", 1)[0].lower(),
        "dist": token.removesuffix(">").rsplit("_", 1)[1].lower(),
        "fraction": fraction,
    }
    for token, (_, _, fraction) in MOVE_SPECS.items()
}
TOKEN_META.update({
    "<CLICK_SHORT>":    {"type": "click", "mode": "short"},
    "<CLICK_LONG>":     {"type": "click", "mode": "long"},
    "<GO_BACK>":        {"type": "nav", "mode": "back"},
    "<GO_HOME>":        {"type": "nav", "mode": "home"},
    "<SCROLL_UP>":      {"type": "scroll", "dir": "up"},
    "<SCROLL_DOWN>":    {"type": "scroll", "dir": "down"},
    "<SCROLL_LEFT>":    {"type": "scroll", "dir": "left"},
    "<SCROLL_RIGHT>":   {"type": "scroll", "dir": "right"},
    "<TEXT_START>":     {"type": "text_start"},
    "<TEXT_END>":       {"type": "text_end"},
    "<END_ACTION>":     {"type": "end"},
})

# ================= LOGIC WITH REASONING =================

def generate_action_sample(token_key, meta):
    """Generates (Instruction, Reasoning, ActionToken) triplet."""
    t_type = meta["type"]
    
    if t_type == "move":
        verb = random.choice(VOCAB["move_verb"])
        direction = random.choice(VOCAB[meta["dir"]])
        distance_desc = random.choice(VOCAB[f"{meta['dist']}_desc"])
        instr = f"{verb} {direction} {distance_desc}"
        
        # Semantic Reasoning
        percent = round(meta["fraction"] * 100)
        reasoning = (
            f"The user wants to {verb.lower()} the cursor {direction}. "
            f"The described distance implies a {meta['dist']} movement covering "
            f"{percent}% of the corresponding screen span."
        )
        return instr, reasoning, token_key

    elif t_type == "click":
        target = random.choice(VOCAB["ui_element"])
        verb = random.choice(VOCAB["click_verb"])
        instr = f"{verb} the {target}"
        mode = "short click" if meta["mode"] == "short" else "long press"
        reasoning = f"The instruction is to {verb.lower()} on a UI element. This requires a {mode} interaction at the current position."
        return instr, reasoning, token_key

    elif t_type == "scroll":
        verb = "Scroll"
        direction = meta["dir"]
        instr = f"{verb} {direction}"
        reasoning = f"The user wants to navigate the view by scrolling {direction}."
        return instr, reasoning, token_key

    elif t_type == "nav":
        if meta["mode"] == "back":
            return "Go back", "The instruction requires returning to the previous view.", token_key
        return "Go home", "The instruction requires returning to the system home screen.", token_key

    elif t_type == "text_start":
        content = random.choice(VOCAB["text_content"])
        instr = f"Type '{content}'"
        reasoning = f"The user wants to input the text '{content}', so text-entry mode must begin."
        return instr, reasoning, token_key

    elif t_type == "text_end":
        instr = "Finish the current text input"
        reasoning = "The text content has been entered and text-entry mode should now end."
        return instr, reasoning, token_key

    elif t_type == "end":
        instr = random.choice(VOCAB["end_action"])
        reasoning = "The task is described as complete, so the interaction should terminate."
        return instr, reasoning, token_key

    raise ValueError(f"Unsupported action metadata: {token_key} -> {meta}")

def generate_explanation_sample(token):
    """Generates definition Q&A."""
    question = f"[Define] What does {token} mean?"
    # Simple definition logic
    if "MOVE" in token:
        meta = TOKEN_META[token]
        direction = meta["dir"].replace("_", "-")
        percent = round(meta["fraction"] * 100)
        if "_" in meta["dir"]:
            definition = (
                f"It moves the cursor {direction}: {percent}% of screen width "
                f"horizontally and {percent}% of screen height vertically, "
                f"equivalent to {percent}% of the screen diagonal."
            )
        elif meta["dir"] in {"left", "right"}:
            definition = (
                f"It moves the cursor {direction} by {percent}% of screen width."
            )
        else:
            definition = (
                f"It moves the cursor {direction} by {percent}% of screen height."
            )
    elif "CLICK" in token:
        definition = random.choice(("It performs a click interaction.", "It means to click the current position."))
    elif "SCROLL" in token:
        if "UP" in token: definition = random.choice(("It is a control token. It means to scroll up.", "It means scrolling up at the current position."))
        elif "DOWN" in token: definition = random.choice(("It is a control token. It means to scroll down.", "It means scrolling down at the current position."))
        elif "LEFT" in token:
            definition = random.choice(("It is a control token. It means to scroll left.", "It means scrolling left at the current position."))
        elif "RIGHT" in token:
            definition = random.choice(("It is a control token. It means to scroll right.", "It means scrolling right at the current position."))
    elif "GO_BACK" in token:
        definition = random.choice(("It navigates back to the previous view.", "It means getting back to the previous page."))
    elif "GO_HOME" in token:
        definition = random.choice(("It navigates to the app's home screen.", "It means using the home button."))
    elif "TEXT"  in token:
        definition = "It indicates a sequence of user text input, typically wrapped with start/end text tokens."
    elif "END" in token:
        definition = random.choice(("It marks the end of an action sequence or session.", "It means stopping the action here."))
    else:
        definition = "It is a control token corresponding to a UI operation."
    
    reasoning = f"The requested control operation means: {definition}"
    return question, reasoning, definition

# ================= MAIN =================

def main():
    random.seed(HP.SFT_SEED)
    print(f"Generating Data with Reasoning...")
    all_records = []
    
    # 1. Action Execution (Semantic)
    action_tokens = list(TOKEN_META.items())
    per_action, action_remainder = divmod(TOTAL_ACTION_SAMPLES, len(action_tokens))
    for token_index, (token_key, meta) in enumerate(action_tokens):
        sample_count = per_action + (1 if token_index < action_remainder else 0)
        for _ in range(sample_count):
            instr, reasoning, resp = generate_action_sample(token_key, meta)
            reasoning = f"{format_task_opener(instr, random)} {reasoning}"
            full_response = format_action_response(reasoning, resp)
            
            record = {
                "data_type": "action", # Needs Image (Noise)
                "messages": [
                    {"role": "user", "content": f"[Action] {instr}"},
                    {"role": "assistant", "content": full_response}
                ]
            }
            all_records.append(record)
            
    # 2. Explanation (Text Only)
    tokens = list(TOKEN_META)
    per_token, remainder = divmod(TOTAL_EXPLANATION_SAMPLES, len(tokens))
    for token_index, token in enumerate(tokens):
        sample_count = per_token + (1 if token_index < remainder else 0)
        for _ in range(sample_count):
            q, reasoning, _ = generate_explanation_sample(token)
            full_response = format_action_response(reasoning, token)
            
            record = {
                "data_type": "explanation", # Black Image
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": full_response}
                ]
            }
            all_records.append(record)
    
    random.shuffle(all_records)
    print(f"Writing {len(all_records)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
