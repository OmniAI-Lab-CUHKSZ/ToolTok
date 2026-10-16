"""Resolution-independent behavior definitions for ToolTok actions."""

from __future__ import annotations


MOVE_DISTANCE_FRACTIONS = {
    "FAR": 0.25,
    "MID": 0.10,
    "CLO": 0.02,
}

MOVE_DIRECTION_SIGNS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP_LEFT": (-1, -1),
    "UP_RIGHT": (1, -1),
    "DOWN_LEFT": (-1, 1),
    "DOWN_RIGHT": (1, 1),
}

# token -> (horizontal sign, vertical sign, fraction of the corresponding axis)
MOVE_SPECS = {
    f"<MOVE_{direction}_{distance}>": (x_sign, y_sign, fraction)
    for direction, (x_sign, y_sign) in MOVE_DIRECTION_SIGNS.items()
    for distance, fraction in MOVE_DISTANCE_FRACTIONS.items()
}
MOVE_ACTION_TOKENS = tuple(MOVE_SPECS)


def get_move_delta(token: str, image_size: tuple[int, int]) -> tuple[int, int]:
    """Resolve a move token to pixels for a ``(width, height)`` screen.

    Horizontal components use a fraction of screen width; vertical components
    use the same fraction of screen height. Consequently a diagonal action
    traverses the configured fraction of the screen diagonal.
    """
    if token not in MOVE_SPECS:
        raise ValueError(f"unknown movement token: {token!r}")
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError(f"invalid image size: {image_size!r}")
    x_sign, y_sign, fraction = MOVE_SPECS[token]
    delta_x = x_sign * max(1, round(width * fraction)) if x_sign else 0
    delta_y = y_sign * max(1, round(height * fraction)) if y_sign else 0
    return delta_x, delta_y


def get_action_type(token: str) -> str:
    """Classify tokens into action categories."""
    if token in MOVE_SPECS:
        return "move"
    if "CLICK" in token:
        return "click"
    if "SCROLL" in token:
        return "scroll"
    if "GO_" in token:
        return "nav"
    if "TEXT" in token:
        return "text"
    if "END_ACTION" in token:
        return "end"
    return "unknown"
