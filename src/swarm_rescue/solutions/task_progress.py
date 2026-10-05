"""Task progress from permitted observations, independent of radio heartbeats."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

Token = tuple[int, int]
Point = tuple[float, float]


@dataclass
class Progress:
    mode: str
    token: Token
    started: int
    advanced: int
    best_distance: float = float('inf')
    metric_anchor: Optional[Point] = None
    cells: set[tuple[int, int]] = field(default_factory=set)


class TaskProgressMonitor:
    """Accept route-distance reduction or new traversed space as progress.

    New space permits a legitimate detour away from a goal. Revisiting the
    same cells, rotating, jittering, or broadcasting cannot renew progress.
    """
    def __init__(self):
        self.current: Optional[Progress] = None

    def update(self, mode: str, token: Optional[Token], step: int,
               position: Optional[Point], distance: Optional[float]) -> Optional[Progress]:
        if mode not in ('claim', 'carry') or token is None:
            self.current = None
            return None
        state = self.current
        if state is None or (state.mode, state.token) != (mode, token):
            state = self.current = Progress(mode, token, step, step)
        if distance is not None and math.isfinite(distance):
            if distance <= state.best_distance - 18.0:
                moved = (state.metric_anchor is None or (position is not None and
                         math.hypot(position[0]-state.metric_anchor[0], position[1]-state.metric_anchor[1]) >= 8.0))
                # A replanned route may be shorter while the drone stays still.
                # Rebase its cost, but do not turn that into measured motion.
                state.best_distance = distance
                if moved:
                    state.advanced = step
                    state.metric_anchor = position
        if position is not None and all(math.isfinite(v) for v in position):
            # Overlapping cells avoid renewing progress by jittering across a
            # single grid boundary. Only space >32 away from visited cells counts.
            cell = (int(math.floor(position[0] / 32)), int(math.floor(position[1] / 32)))
            neighbours = {(cell[0]+dx, cell[1]+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            if not state.cells.intersection(neighbours) and len(state.cells) < 512:
                state.cells.add(cell)
                state.advanced = step
        return state

    def stalled(self, step: int, timeout: int) -> bool:
        return self.current is not None and step - self.current.advanced >= timeout


def valid_progress_snapshot(mode: str, token: Optional[Token], task_step: int,
                            snapshot: Optional[tuple[str, Token, int, int, int]],
                            now: int, timeout: int, lease: int) -> bool:
    """A fresh relayed heartbeat cannot make old progress fresh again."""
    if snapshot is None or token is None:
        return False
    observed_mode, observed_token, started, advanced, reported = snapshot
    return (mode == observed_mode and token == observed_token
            and 0 <= started <= advanced <= reported <= now
            and reported <= task_step and now - reported <= lease
            and now - advanced < timeout)
