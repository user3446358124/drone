"""Path validity checks for the online drone planner."""

from types import SimpleNamespace

import numpy as np

from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample
from swarm_rescue.solutions.online_navigation import (
    _bidirectional_astar,
    choose_lidar_detour,
    plan_occupancy_path,
)


def test_bidirectional_search_uses_wall_opening():
    blocked = np.zeros((30, 30), dtype=bool)
    blocked[:, 14] = True
    blocked[19, 14] = False
    route = _bidirectional_astar(blocked, (5, 5), (5, 24))
    assert route[0] == (5, 5)
    assert route[-1] == (5, 24)
    assert (19, 14) in route
    assert all(not blocked[cell] for cell in route)
    for first, second in zip(route, route[1:]):
        dr, dc = second[0] - first[0], second[1] - first[1]
        assert abs(dr) <= 1 and abs(dc) <= 1
        if dr and dc:
            assert not blocked[first[0] + dr, first[1]]
            assert not blocked[first[0], first[1] + dc]


def test_bidirectional_search_respects_disconnected_regions():
    blocked = np.zeros((24, 24), dtype=bool)
    blocked[:, 12] = True
    assert _bidirectional_astar(blocked, (5, 5), (5, 18)) == []


def test_planner_does_not_append_blocked_goal():
    obstacles = np.zeros((240, 240), dtype=np.uint8)
    obstacles[110:130, 110:130] = 1
    path = plan_occupancy_path(
        (240, 240), obstacles, (-80.0, 0.0), (0.0, 0.0),
        clearance=20, cell_size=12,
    )
    assert path
    assert path[-1] != (0.0, 0.0)


def test_disconnected_plan_fallback_uses_a_clear_lidar_ray():
    detour = choose_lidar_detour(
        (0.0, 0.0), 0.0, (100.0, 0.0),
        [35.0, 150.0, 150.0],
        [0.0, np.pi / 2.0, -np.pi / 2.0],
    )
    assert detour is not None
    assert abs(detour[0]) < 1e-6
    assert 0.0 < abs(detour[1]) <= 105.0
    assert choose_lidar_detour(
        (0.0, 0.0), 0.0, (100.0, 0.0), [35.0], [0.0]
    ) is None


def test_online_navigation_executes_pending_lidar_escape():
    controller = SimpleNamespace(
        _ARRIVE_RADIUS=10.0,
        _nav_age=0,
        _lidar_blocked_steps=120,
        measured_gps_position=lambda: (0.0, 0.0),
        _run_lidar_escape_if_needed=lambda cmd: cmd.update(rotation=0.8) or True,
    )
    cmd = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
    assert not MyDroneRescueExample._navigate_online(controller, (100.0, 0.0), cmd)
    assert cmd["rotation"] == 0.8
    assert controller._nav_age == 1
