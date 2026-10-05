"""Map-agnostic coverage and planning built only from online observations."""

from __future__ import annotations

import heapq
import math
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

WorldPoint = tuple[float, float]


def _size_key(size: Sequence[int] | None) -> Optional[tuple[int, int]]:
    if size is None or len(size) < 2:
        return None
    return int(size[0]), int(size[1])


def _world_to_grid(point: WorldPoint, key: tuple[int, int]) -> tuple[int, int]:
    width, height = key
    return int(round(-point[1] + height / 2)), int(round(point[0] + width / 2))


def _grid_to_world(row: float, col: float, key: tuple[int, int]) -> WorldPoint:
    width, height = key
    return float(col - width / 2), float(-(row - height / 2))


def _nearest_free(
    node: tuple[int, int], blocked: np.ndarray, max_radius: int = 8
) -> Optional[tuple[int, int]]:
    rows, cols = blocked.shape
    row = max(0, min(rows - 1, node[0]))
    col = max(0, min(cols - 1, node[1]))
    if not blocked[row, col]:
        return row, col
    for radius in range(1, max_radius + 1):
        candidates: list[tuple[int, int]] = []
        for rr in range(row - radius, row + radius + 1):
            candidates.extend(((rr, col - radius), (rr, col + radius)))
        for cc in range(col - radius + 1, col + radius):
            candidates.extend(((row - radius, cc), (row + radius, cc)))
        valid = [
            point
            for point in candidates
            if 0 <= point[0] < rows
            and 0 <= point[1] < cols
            and not blocked[point]
        ]
        if valid:
            return min(
                valid,
                key=lambda point: (point[0] - row) ** 2 + (point[1] - col) ** 2,
            )
    return None


def _bidirectional_astar(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Meet two weighted searches in free space, with no diagonal corner cuts.

    Online LiDAR maps change often.  Replanning through a disconnected maze
    must have a bounded cost, so each search visits at most the coarse grid.
    """
    if start == goal:
        return [start]
    rows, cols = blocked.shape
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    roots = (start, goal)
    heaps: list[list[tuple[float, float, tuple[int, int]]]] = [
        [(math.hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, start)],
        [(math.hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, goal)],
    ]
    costs: list[dict[tuple[int, int], float]] = [{start: 0.0}, {goal: 0.0}]
    parents: list[dict[tuple[int, int], tuple[int, int]]] = [{}, {}]
    visited: list[set[tuple[int, int]]] = [set(), set()]
    meeting: Optional[tuple[int, int]] = None
    expansions = 0
    # The bounded search avoids spending the whole control budget proving
    # that a partially mapped wall has disconnected the goal.
    expansion_limit = min(rows * cols, 5000)
    while heaps[0] and heaps[1] and expansions < expansion_limit:
        side = 0 if heaps[0][0][0] <= heaps[1][0][0] else 1
        _, cost, node = heapq.heappop(heaps[side])
        if cost > costs[side].get(node, float("inf")) + 1e-9 or node in visited[side]:
            continue
        visited[side].add(node)
        expansions += 1
        if node in visited[1 - side]:
            meeting = node
            break
        for dr, dc, move_cost in moves:
            nxt = node[0] + dr, node[1] + dc
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if blocked[nxt]:
                continue
            if dr and dc and (
                blocked[node[0] + dr, node[1]]
                or blocked[node[0], node[1] + dc]
            ):
                continue
            new_cost = cost + move_cost
            if new_cost >= costs[side].get(nxt, float("inf")):
                continue
            costs[side][nxt] = new_cost
            parents[side][nxt] = node
            target = roots[1 - side]
            heuristic = math.hypot(target[0] - nxt[0], target[1] - nxt[1])
            heapq.heappush(heaps[side], (new_cost + 1.15 * heuristic, new_cost, nxt))
    if meeting is not None:
        left = [meeting]
        while left[-1] != start:
            left.append(parents[0][left[-1]])
        left.reverse()
        right = [meeting]
        while right[-1] != goal:
            right.append(parents[1][right[-1]])
        return left + right[1:]
    return []


def _line_is_free(
    start: WorldPoint,
    end: WorldPoint,
    key: tuple[int, int],
    blocked: np.ndarray,
) -> bool:
    row0, col0 = _world_to_grid(start, key)
    row1, col1 = _world_to_grid(end, key)
    distance = max(abs(row1 - row0), abs(col1 - col0))
    samples = max(2, int(math.ceil(distance / 5.0)) + 1)
    rows = np.rint(np.linspace(row0, row1, samples)).astype(int)
    cols = np.rint(np.linspace(col0, col1, samples)).astype(int)
    if np.any(rows < 0) or np.any(rows >= blocked.shape[0]):
        return False
    if np.any(cols < 0) or np.any(cols >= blocked.shape[1]):
        return False
    return not bool(np.any(blocked[rows, cols]))


def _smooth_path(
    points: list[WorldPoint], key: tuple[int, int], blocked: np.ndarray
) -> list[WorldPoint]:
    if len(points) <= 2:
        return points
    result = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1:
            if _line_is_free(points[anchor], points[candidate], key, blocked):
                break
            candidate -= 1
        result.append(points[candidate])
        anchor = candidate
    return result


def plan_occupancy_path(
    size: Sequence[int] | None,
    obstacle_grid: np.ndarray,
    start: WorldPoint,
    goal: WorldPoint,
    *,
    clearance: int = 24,
    cell_size: int = 10,
) -> list[WorldPoint]:
    """Plan through the wall cells accumulated from Lidar observations."""
    key = _size_key(size)
    if key is None:
        return []
    width, height = key
    grid = (np.asarray(obstacle_grid) > 0).astype(np.uint8)
    if grid.shape != (height, width):
        return []
    distance = cv2.distanceTransform(1-grid, cv2.DIST_L2, 5)
    # A one-pixel margin covers the chamfer approximation's error.
    blocked = (distance <= float(clearance) + 1.0).astype(np.uint8)
    blocked[: clearance + 1, :] = 1
    blocked[-clearance - 1 :, :] = 1
    blocked[:, : clearance + 1] = 1
    blocked[:, -clearance - 1 :] = 1

    coarse_width = int(math.ceil(width / cell_size))
    coarse_height = int(math.ceil(height / cell_size))
    coarse = cv2.resize(
        blocked.astype(np.float32),
        (coarse_width, coarse_height),
        interpolation=cv2.INTER_AREA,
    ) > 0.02
    start_row, start_col = _world_to_grid(start, key)
    goal_row, goal_col = _world_to_grid(goal, key)
    coarse_start = _nearest_free(
        (start_row // cell_size, start_col // cell_size), coarse, 12
    )
    coarse_goal = _nearest_free(
        (goal_row // cell_size, goal_col // cell_size), coarse, 18
    )
    if coarse_start is None or coarse_goal is None:
        return []
    nodes = _bidirectional_astar(coarse, coarse_start, coarse_goal)
    if not nodes:
        return []
    points = [start]
    for row, col in nodes[1:-1]:
        points.append(
            _grid_to_world(
                row * cell_size + cell_size / 2,
                col * cell_size + cell_size / 2,
                key,
            )
        )
    # A target can lie inside an inflated wall or just beyond a map edge.
    # End at the reachable free cell in that case instead of appending an
    # unsafe straight segment from A*'s last node to the original target.
    goal_center = _grid_to_world(
        coarse_goal[0] * cell_size + cell_size / 2,
        coarse_goal[1] * cell_size + cell_size / 2,
        key,
    )
    if _line_is_free(goal_center, goal, key, blocked):
        points.append(goal)
    else:
        points.append(goal_center)
    return _smooth_path(points, key, blocked)[1:]


def choose_lidar_detour(
    position: WorldPoint,
    heading: float,
    goal: WorldPoint,
    distances: Sequence[float] | None,
    angles: Sequence[float] | None,
) -> Optional[WorldPoint]:
    """Choose a bounded free-ray subgoal after a disconnected grid plan."""
    if distances is None or angles is None:
        return None
    count = min(len(distances), len(angles))
    if count == 0:
        return None
    ranges = np.asarray(distances[:count], dtype=float)
    bearings = np.asarray(angles[:count], dtype=float)
    valid = np.isfinite(ranges) & np.isfinite(bearings) & (ranges >= 70.0)
    if not np.any(valid):
        return None
    desired = math.atan2(goal[1] - position[1], goal[0] - position[0]) - heading
    error = np.abs(np.arctan2(
        np.sin(bearings - desired), np.cos(bearings - desired)
    ))
    # Prefer progress toward the goal, with clearance breaking close ties.
    score = error - 0.0015 * np.minimum(ranges, 180.0)
    score[~valid] = float("inf")
    index = int(np.argmin(score))
    distance = min(105.0, float(ranges[index]) - 32.0)
    bearing = heading + float(bearings[index])
    return (
        position[0] + distance * math.cos(bearing),
        position[1] + distance * math.sin(bearing),
    )


def generic_patrol_route(
    size: Sequence[int] | None,
    drone_id: int,
    number_drones: int,
    *,
    spacing: float = 135.0,
    margin: float = 65.0,
) -> tuple[WorldPoint, ...]:
    """Create coverage lanes using only the arena dimensions supplied by sensors."""
    key = _size_key(size)
    if key is None:
        return ()
    width, height = key
    count_drones = max(1, int(number_drones))
    lane = int(drone_id) % count_drones
    safe_margin = min(float(margin), width * 0.14, height * 0.14)
    usable_height = max(1.0, height - 2 * safe_margin)
    count = max(3, int(math.ceil(usable_height / max(60.0, spacing))) + 1)
    base_ys = np.linspace(-height / 2 + safe_margin, height / 2 - safe_margin, count)
    lane_width = max(1.0, width - 2 * safe_margin) / count_drones
    points: list[WorldPoint] = []
    for pass_index, active_lane in enumerate((lane, (lane + 1) % count_drones)):
        x = -width / 2 + safe_margin + (active_lane + 0.5) * lane_width
        ys = base_ys[::-1] if (active_lane + pass_index) % 2 == 0 else base_ys
        points.extend((float(x), float(y)) for y in ys)

    return tuple(points)


def path_length(start: WorldPoint, points: Iterable[WorldPoint]) -> float:
    total = 0.0
    previous = start
    for point in points:
        total += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
    return total
