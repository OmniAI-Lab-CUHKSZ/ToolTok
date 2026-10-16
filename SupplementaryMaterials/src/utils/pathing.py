"""Shared, label-safe cursor path planning helpers."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .action_logic import MOVE_ACTION_TOKENS, get_move_delta


Point = tuple[int, int]
ActionStep = tuple[str, Point]


def clamp_bbox(bbox: Sequence[float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Sort and clamp an absolute xyxy box to the image."""
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    x1, x2 = sorted((max(0.0, min(width - 1.0, x1)), max(0.0, min(width - 1.0, x2))))
    y1, y2 = sorted((max(0.0, min(height - 1.0, y1)), max(0.0, min(height - 1.0, y2))))
    return x1, y1, x2, y2


def bbox_center(bbox: Sequence[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_bbox(point: Point, bbox: Sequence[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def distance_to_bbox(point: Point, bbox: Sequence[float]) -> float:
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def normalized_distance_to_bbox(
    point: Point,
    bbox: Sequence[float],
    image_size: tuple[int, int],
) -> float:
    """Measure bbox distance in screen-width/screen-height units."""
    width, height = image_size
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2) / width
    dy = max(y1 - y, 0.0, y - y2) / height
    return math.hypot(dx, dy)


def build_action_path(
    start_pos: Point,
    target_bbox: Sequence[float],
    image_size: tuple[int, int],
    max_steps: int = 20,
) -> list[ActionStep]:
    """Greedily approach a bbox and click only from a position inside it."""
    width, height = image_size
    bbox = clamp_bbox(target_bbox, image_size)
    cx, cy = start_pos
    path: list[ActionStep] = []
    moves: Iterable[tuple[str, tuple[int, int]]] = sorted(
        (
            (token, get_move_delta(token, image_size))
            for token in MOVE_ACTION_TOKENS
        ),
        key=lambda item: math.hypot(item[1][0], item[1][1]),
        reverse=True,
    )

    for _ in range(max_steps):
        if point_in_bbox((cx, cy), bbox):
            path.append(("<CLICK_SHORT>", (cx, cy)))
            return path

        current_distance = normalized_distance_to_bbox(
            (cx, cy), bbox, image_size
        )
        best_step: ActionStep | None = None
        best_distance = current_distance
        for token, (delta_x, delta_y) in moves:
            next_x, next_y = cx + delta_x, cy + delta_y
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            remaining = normalized_distance_to_bbox(
                (next_x, next_y), bbox, image_size
            )
            if remaining + 1e-6 < best_distance:
                best_distance = remaining
                best_step = token, (next_x, next_y)

        if best_step is None:
            return path
        path.append(best_step)
        cx, cy = best_step[1]

    if point_in_bbox((cx, cy), bbox):
        path.append(("<CLICK_SHORT>", (cx, cy)))
    return path


def sample_start_position(
    target_center: Point,
    image_size: tuple[int, int],
    rng: random.Random | None = None,
) -> Point:
    """Sample 50% random, 40% opposite-corner, and 10% center starts."""
    rng = random if rng is None else rng
    width, height = image_size
    target_x, target_y = target_center
    draw = rng.random()
    if draw < 0.5:
        return rng.randint(0, width - 1), rng.randint(0, height - 1)
    if draw < 0.9:
        margin_x = max(10, int(width * 0.1))
        margin_y = max(10, int(height * 0.1))
        start_x = rng.randint(width - margin_x, width - 1) if target_x < width / 2 else rng.randint(0, margin_x)
        start_y = rng.randint(height - margin_y, height - 1) if target_y < height / 2 else rng.randint(0, margin_y)
        return start_x, start_y
    return width // 2, height // 2
