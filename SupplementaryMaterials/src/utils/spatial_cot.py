"""Concise, diverse, and label-faithful spatial reasoning supervision."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from .action_format import format_action_response
from .action_logic import MOVE_SPECS
from .parameters import HYPERPARAMS as HP
from .pathing import point_in_bbox


TASK_OPENERS = (
    'Okay, the user asks me to "{task}". Let me locate the target.',
    'Understood—the user wants to "{task}". I will locate the target first.',
    'The requested task is "{task}". Let me find the relevant target.',
    'I need to handle the request "{task}". First, I will locate the target.',
    'Got it: "{task}". Let me identify the target before acting.',
    'The user wants to complete "{task}". I will start by finding the target.',
    'My task is "{task}". Let me locate the correct target on the screen.',
    'To carry out "{task}", I first need to identify the target.',
    'I should complete "{task}". Let me find where the target is.',
    'The instruction is "{task}". I will locate the intended target first.',
    'Okay, I need to complete "{task}". Let me find the relevant element.',
    'The goal is "{task}". I will first locate the target on the screen.',
)

DIAGONAL_GEOMETRY = (
    "The target box is {relation}, about {gap_x} of screen width and {gap_y} of screen height away.",
    "Remaining offset: {relation}, roughly {gap_x} W and {gap_y} H.",
    "The cursor must close a {gap_x} horizontal and {gap_y} vertical gap toward {relation}.",
    "The target lies {relation}; the unresolved gaps are about {gap_x} W and {gap_y} H.",
    "From the cursor, the target box is {relation} with approximately {gap_x} W / {gap_y} H remaining.",
    "A {gap_x} width gap and {gap_y} height gap remain toward {relation}.",
    "The target is {relation} of the cursor, separated by about {gap_x} W and {gap_y} H.",
    "To reach the box, the cursor must move {relation} across roughly {gap_x} W and {gap_y} H.",
)

HORIZONTAL_GEOMETRY = (
    "The target box is directly {relation}, about {gap_x} of screen width away.",
    "Only a {gap_x} horizontal gap remains toward the {relation}.",
    "The cursor is vertically aligned with the target, with {gap_x} W remaining {relation}.",
    "No vertical correction is needed; the target is about {gap_x} W to the {relation}.",
    "The target overlaps the cursor vertically and lies {gap_x} W to the {relation}.",
    "The remaining target offset is horizontal: {gap_x} W toward {relation}.",
)

VERTICAL_GEOMETRY = (
    "The target box is directly {relation}, about {gap_y} of screen height away.",
    "Only a {gap_y} vertical gap remains toward the {relation}.",
    "The cursor is horizontally aligned with the target, with {gap_y} H remaining {relation}.",
    "No horizontal correction is needed; the target is about {gap_y} H to the {relation}.",
    "The target overlaps the cursor horizontally and lies {gap_y} H to the {relation}.",
    "The remaining target offset is vertical: {gap_y} H toward {relation}.",
)

DIAGONAL_DECISIONS = (
    "A {step}% {direction} step reduces both gap components in one move.",
    "Move {direction} by {step}% of each axis for the best one-step progress.",
    "The {step}% {direction} diagonal gives the strongest reduction in the remaining gap.",
    "A {step}% step toward {direction} is the closest available move.",
    "Use the {step}% {direction} move to close both components together.",
    "Among the available strides, the {step}% {direction} move leaves the least target-box distance.",
    "The best next correction is {step}% toward {direction} along both axes.",
    "A {step}% diagonal adjustment toward {direction} gives the best progress.",
)

CARDINAL_DECISIONS = (
    "A {step}%-of-{dimension} move {direction} best reduces the remaining distance.",
    "Move {direction} by {step}% of screen {dimension}.",
    "The {step}% {direction} stride is the closest available correction.",
    "A {step}% move along the {axis} axis gives the best one-step progress.",
    "Use a {step}%-of-{dimension} step toward {direction} to close the main gap.",
    "Among valid moves, the {step}% {direction} step leaves the least target-box distance.",
    "The best next correction is a {step}% move {direction} along the {axis} axis.",
    "A {step}% {direction} adjustment gives the strongest reduction in the remaining gap.",
)

CLICK_GEOMETRY = (
    "The cursor is already inside the target box.",
    "The target box contains the cursor.",
    "The cursor now overlaps the requested target.",
    "The cursor is positioned within the target bounds.",
    "The target has been reached and the cursor lies inside its box.",
    "The cursor is on the intended target.",
)

CLICK_DECISIONS = (
    "No movement is needed, so click now.",
    "The element is ready for a click.",
    "I can perform the click immediately.",
    "The correct next action is to click.",
    "With the target reached, I should click it.",
    "I will click the target at the current position.",
)

RELATION_NAMES = {
    (-1, -1): "upper-left",
    (0, -1): "up",
    (1, -1): "upper-right",
    (-1, 0): "left",
    (0, 0): "overlapping",
    (1, 0): "right",
    (-1, 1): "lower-left",
    (0, 1): "down",
    (1, 1): "lower-right",
}


def _stable_rng(*parts) -> random.Random:
    payload = repr((HP.SFT_SEED, *parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def _clean_task(task: str) -> str:
    cleaned = " ".join(str(task).strip().split()).strip('"')
    return cleaned[:-1] if cleaned.endswith(".") else cleaned


def format_task_opener(task: str, rng: random.Random) -> str:
    """Render the empirically useful task-restatement first sentence."""
    cleaned = _clean_task(task)
    if not cleaned:
        cleaned = "interact with the requested target"
    return rng.choice(TASK_OPENERS).format(task=cleaned)


def _signed_axis_gap(position: float, lower: float, upper: float) -> float:
    if position < lower:
        return lower - position
    if position > upper:
        return upper - position
    return 0.0


def _format_percent(ratio: float) -> str:
    percent = abs(ratio) * 100
    if 0 < percent < 1:
        return "less than 1%"
    return f"{round(percent):d}%"


def _movement_direction(token: str) -> tuple[str, bool, str, str]:
    x_sign, y_sign, _ = MOVE_SPECS[token]
    direction = RELATION_NAMES[(x_sign, y_sign)]
    diagonal = bool(x_sign and y_sign)
    axis = "horizontal" if x_sign else "vertical"
    dimension = "width" if x_sign else "height"
    return direction, diagonal, axis, dimension


def format_spatial_cot(
    task: str,
    cursor_pos: tuple[int, int],
    target_bbox: Sequence[float],
    image_size: tuple[int, int],
    action_token: str,
) -> str:
    """Format task, bbox geometry, and one-step decision as a compact CoT."""
    width, height = image_size
    raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in target_bbox)
    x1, x2 = sorted((raw_x1, raw_x2))
    y1, y2 = sorted((raw_y1, raw_y2))
    normalized_bbox = (x1, y1, x2, y2)
    cursor_x, cursor_y = cursor_pos
    rng = _stable_rng(task, cursor_pos, normalized_bbox, image_size, action_token)
    opener = format_task_opener(task, rng)

    if action_token == "<CLICK_SHORT>":
        if not point_in_bbox(cursor_pos, normalized_bbox):
            raise ValueError("click CoT requires the cursor to be inside the target bbox")
        reasoning = " ".join(
            (opener, rng.choice(CLICK_GEOMETRY), rng.choice(CLICK_DECISIONS))
        )
        return format_action_response(reasoning, action_token)

    if action_token not in MOVE_SPECS:
        raise ValueError(f"spatial CoT requires a move or short-click token: {action_token}")

    gap_x = _signed_axis_gap(cursor_x, x1, x2)
    gap_y = _signed_axis_gap(cursor_y, y1, y2)
    signed_x = 1 if gap_x > 0 else (-1 if gap_x < 0 else 0)
    signed_y = 1 if gap_y > 0 else (-1 if gap_y < 0 else 0)
    relation = RELATION_NAMES[(signed_x, signed_y)]
    gap_x_ratio = gap_x / width
    gap_y_ratio = gap_y / height
    values = {
        "relation": relation,
        "gap_x": _format_percent(gap_x_ratio),
        "gap_y": _format_percent(gap_y_ratio),
    }
    if gap_x and gap_y:
        geometry = rng.choice(DIAGONAL_GEOMETRY).format(**values)
    elif gap_x:
        geometry = rng.choice(HORIZONTAL_GEOMETRY).format(**values)
    elif gap_y:
        geometry = rng.choice(VERTICAL_GEOMETRY).format(**values)
    else:
        raise ValueError("movement CoT received a cursor already inside the target bbox")

    direction, diagonal, axis, dimension = _movement_direction(action_token)
    step = round(MOVE_SPECS[action_token][2] * 100)
    decision_values = {
        "direction": direction,
        "step": step,
        "axis": axis,
        "dimension": dimension,
    }
    templates = DIAGONAL_DECISIONS if diagonal else CARDINAL_DECISIONS
    decision = rng.choice(templates).format(**decision_values)
    return format_action_response(" ".join((opener, geometry, decision)), action_token)


def format_simple_action_cot(task: str, fact: str, decision: str, action_token: str) -> str:
    """Use the same first-line task restatement for non-movement action labels."""
    rng = _stable_rng(task, fact, decision, action_token)
    reasoning = " ".join((format_task_opener(task, rng), fact.strip(), decision.strip()))
    return format_action_response(reasoning, action_token)
