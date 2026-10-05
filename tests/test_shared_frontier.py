"""Map knowledge must survive a radio relay without a new local scan."""

from types import SimpleNamespace

import numpy as np

from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def _scout(identifier, step, messages=()):
    return SimpleNamespace(
        identifier=identifier,
        elapsed_timestep=step,
        _frontier_grid=np.zeros((4, 4), dtype=np.uint8),
        _frontier_snapshot_cursor=0,
        _peer_frontier_targets={},
        _recent_frontier_updates=(),
        communicator=SimpleNamespace(received_messages=list(messages)),
        measured_gps_position=lambda: None,
        measured_compass_angle=lambda: None,
        lidar_values=lambda: None,
    )


def test_coverage_cell_relays_to_a_drone_that_never_met_the_observer():
    observer = _scout(0, 0)
    observer._frontier_grid[1, 2] = 1
    update = MyDroneRescueExample._frontier_snapshot(observer)
    assert update == ((1, 2, 1),)

    relay = _scout(1, 1, [(None, {"id": 0, "step": 0, "frontier_updates": update})])
    MyDroneRescueExample._update_frontier_map(relay)
    assert relay._frontier_grid[1, 2] == 1

    forwarded = MyDroneRescueExample._frontier_snapshot(relay)
    receiver = _scout(2, 2, [(None, {"id": 1, "step": 1, "frontier_updates": forwarded})])
    MyDroneRescueExample._update_frontier_map(receiver)
    assert receiver._frontier_grid[1, 2] == 1
