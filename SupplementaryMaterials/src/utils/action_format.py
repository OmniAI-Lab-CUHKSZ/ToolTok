"""Canonical assistant response format for ToolTok action prediction."""

from __future__ import annotations

import re

from .action import ACTION_TOKENS


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def format_action_response(reasoning: str, action_token: str) -> str:
    """Return one reasoning block followed by exactly one action token."""
    if action_token not in ACTION_TOKENS:
        raise ValueError(f"unknown action token: {action_token!r}")

    cleaned_reasoning = reasoning.strip()
    if THINK_OPEN in cleaned_reasoning or THINK_CLOSE in cleaned_reasoning:
        raise ValueError("reasoning must not contain nested <think> tags")
    return f"{THINK_OPEN}{cleaned_reasoning}{THINK_CLOSE}\n{action_token}"


def extract_action_response(text: str) -> tuple[str, str | None]:
    """Extract reasoning and the single action emitted after ``</think>``."""
    if THINK_CLOSE not in text:
        return text.strip(), None

    reasoning_part, action_part = text.rsplit(THINK_CLOSE, 1)
    reasoning = reasoning_part.split(THINK_OPEN, 1)[-1].strip()
    matches = [token for token in ACTION_TOKENS if token in action_part]
    if len(matches) != 1:
        return reasoning, None

    action_token = matches[0]
    suffix_without_specials = action_part.replace("<|im_end|>", "").strip()
    if not re.fullmatch(re.escape(action_token), suffix_without_specials):
        return reasoning, None
    return reasoning, action_token
