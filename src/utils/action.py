# src/utils/action.py

"""
GUI Agent Action Space Definition & Semantic Anchors.

This dictionary maps specific action tokens to a list of natural language words.
These words are used to initialize the embeddings of the action tokens, 
giving them a "Semantic Anchor" in the vector space.
"""

ACTION_BASE_EMBEDDING = {
    # ===========================
    # Movement (Cursor Control)
    # ===========================

    # Up
    "<MOVE_UP_FAR>":   ["move", "cursor", "up", "top", "far", "jump", "leap"],
    "<MOVE_UP_MID>":   ["move", "cursor", "up", "medium", "shift"],
    "<MOVE_UP_CLO>":   ["move", "cursor", "up", "near", "tiny", "nudge", "slight"],

    # Down
    "<MOVE_DOWN_FAR>": ["move", "cursor", "down", "bottom", "far", "jump", "leap"],
    "<MOVE_DOWN_MID>": ["move", "cursor", "down", "medium", "shift"],
    "<MOVE_DOWN_CLO>": ["move", "cursor", "down", "near", "tiny", "nudge", "slight"],

    # Left
    "<MOVE_LEFT_FAR>": ["move", "cursor", "left", "west", "far", "jump", "leap"],
    "<MOVE_LEFT_MID>": ["move", "cursor", "left", "medium", "shift"],
    "<MOVE_LEFT_CLO>": ["move", "cursor", "left", "near", "tiny", "nudge", "slight"],

    # Right
    "<MOVE_RIGHT_FAR>": ["move", "cursor", "right", "east", "far", "jump", "leap"],
    "<MOVE_RIGHT_MID>": ["move", "cursor", "right", "medium", "shift"],
    "<MOVE_RIGHT_CLO>": ["move", "cursor", "right", "near", "tiny", "nudge", "slight"],

    # Diagonal: Up-Left
    "<MOVE_UP_LEFT_FAR>": ["move", "cursor", "up", "left", "upper", "northwest", "far", "jump"],
    "<MOVE_UP_LEFT_MID>": ["move", "cursor", "up", "left", "upper", "northwest", "medium", "shift"],
    "<MOVE_UP_LEFT_CLO>": ["move", "cursor", "up", "left", "upper", "northwest", "near", "nudge"],

    # Diagonal: Up-Right
    "<MOVE_UP_RIGHT_FAR>": ["move", "cursor", "up", "right", "upper", "northeast", "far", "jump"],
    "<MOVE_UP_RIGHT_MID>": ["move", "cursor", "up", "right", "upper", "northeast", "medium", "shift"],
    "<MOVE_UP_RIGHT_CLO>": ["move", "cursor", "up", "right", "upper", "northeast", "near", "nudge"],

    # Diagonal: Down-Left
    "<MOVE_DOWN_LEFT_FAR>": ["move", "cursor", "down", "left", "lower", "southwest", "far", "jump"],
    "<MOVE_DOWN_LEFT_MID>": ["move", "cursor", "down", "left", "lower", "southwest", "medium", "shift"],
    "<MOVE_DOWN_LEFT_CLO>": ["move", "cursor", "down", "left", "lower", "southwest", "near", "nudge"],

    # Diagonal: Down-Right
    "<MOVE_DOWN_RIGHT_FAR>": ["move", "cursor", "down", "right", "lower", "southeast", "far", "jump"],
    "<MOVE_DOWN_RIGHT_MID>": ["move", "cursor", "down", "right", "lower", "southeast", "medium", "shift"],
    "<MOVE_DOWN_RIGHT_CLO>": ["move", "cursor", "down", "right", "lower", "southeast", "near", "nudge"],

    # ===========================
    # Interaction (Clicking)
    # ===========================
    "<CLICK_SHORT>": ["click", "tap", "select", "press", "touch", "brief", "mouse"],
    "<CLICK_LONG>":  ["click", "hold", "press", "long", "keep", "sustain"],

    # ===========================
    # Navigation (System)
    # ===========================
    "<GO_BACK>": ["navigate", "back", "return", "previous", "history", "reverse", "phone", "mobile"],
    "<GO_HOME>": ["navigate", "home", "main", "dashboard", "desktop", "launcher", "phone", "mobile"],

    # ===========================
    # Scrolling
    # ===========================
    "<SCROLL_UP>":    ["scroll", "swipe", "pan", "up", "top", "content"],
    "<SCROLL_DOWN>":  ["scroll", "swipe", "pan", "down", "bottom", "content"],
    "<SCROLL_LEFT>":  ["scroll", "swipe", "pan", "left", "west", "content"],
    "<SCROLL_RIGHT>": ["scroll", "swipe", "pan", "right", "east", "content"],

    # ===========================
    # Text Input
    # ===========================
    "<TEXT_START>": ["type", "write", "input", "keyboard", "start", "focus", "activate"],
    "<TEXT_END>":   ["type", "finish", "enter", "submit", "confirm", "return"],
    
    # ===========================
    # Termination
    # ===========================
    "<END_ACTION>": ["stop", "finish", "complete", "task", "done", "halt", "cease"],
}

# Helper list for easy access to keys
ACTION_TOKENS = list(ACTION_BASE_EMBEDDING.keys())


def atomic_action_token_ids(tokenizer):
    """Return action IDs, failing if any token is split, unknown, or duplicated."""
    token_ids = {}
    seen_ids = {}
    for token in ACTION_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"action token {token!r} must encode to exactly one ID; got {encoded}. "
                "Run Phase 0 token initialization before training."
            )
        token_id = encoded[0]
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"action token {token!r} resolves to the unknown-token ID")
        if token_id in seen_ids:
            raise ValueError(
                f"action tokens {seen_ids[token_id]!r} and {token!r} share ID {token_id}"
            )
        token_ids[token] = token_id
        seen_ids[token_id] = token
    return token_ids
