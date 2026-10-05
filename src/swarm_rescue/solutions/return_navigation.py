"""Return shortcuts certified against sensor-observed free space."""
from __future__ import annotations
import math
from typing import Optional, Sequence
import cv2
import numpy as np
from swarm_rescue.solutions.online_navigation import plan_occupancy_path, _line_is_free

Point = tuple[float, float]


def path_length(start: Point, points: Sequence[Point]) -> float:
    return sum(math.hypot(b[0]-a[0], b[1]-a[1])
               for a, b in zip([start, *points], points))


def observed_return_route(size, evidence: np.ndarray, obstacles: np.ndarray,
                          start: Point, trail: Sequence[Point],
                          disposal: Optional[Point] = None, *,
                          evidence_cell_size: int = 8, clearance: int = 28):
    """Unknown space is blocked; keep the old trail if no better certified route exists.

    No simulator map, peer truth or bomb locations enter this planner. The
    clearance is a conservative centre buffer, not a complete payload model.
    """
    original = list(trail)
    if size is None or not original or evidence.size == 0:
        return original, False
    width, height = int(size[0]), int(size[1])
    known = np.repeat(np.repeat(evidence <= -2, evidence_cell_size, axis=0),
                      evidence_cell_size, axis=1)[:height, :width]
    if known.shape != (height, width) or obstacles.shape != known.shape:
        return original, False
    blocked = ((~known) | (obstacles > 0)).astype(np.uint8)
    diameter = 2*clearance+1
    inflated = cv2.dilate(blocked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    inflated[:clearance+1, :] = inflated[-clearance-1:, :] = 1
    inflated[:, :clearance+1] = inflated[:, -clearance-1:] = 1
    key = width, height
    home = disposal if disposal is not None else original[-1]
    def cost(route):
        return path_length(start, route)+math.hypot(route[-1][0]-home[0], route[-1][1]-home[1])
    baseline = cost(original)
    candidates = []
    # The start trail anchor remains useful when the shared disposal has not
    # been observed locally and hence cannot be reached in the certified grid.
    for goal in dict.fromkeys((home, original[-1])):
        route = plan_occupancy_path(key, blocked, start, goal, clearance=clearance, cell_size=12)
        if not route or math.hypot(route[-1][0]-goal[0], route[-1][1]-goal[1]) > 1.0:
            continue
        if not all(_line_is_free(a, b, key, inflated)
                   for a, b in zip([start, *route], route)):
            continue
        candidate_cost = cost(route)
        if candidate_cost < baseline*0.95 or (len(route) < len(original) and candidate_cost <= baseline):
            candidates.append((candidate_cost, len(route), route))
    if not candidates:
        return original, False
    return min(candidates, key=lambda item: item[:2])[2], True
