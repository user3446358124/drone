"""Yielding may use only a previously traversed route away from a peer."""

from types import SimpleNamespace

from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def test_yield_path_backtracks_away_from_oncoming_peer():
    drone = SimpleNamespace(_breadcrumbs=[(-120.0, 0.0), (-90.0, 0.0), (-60.0, 0.0), (-30.0, 0.0), (0.0, 0.0)])
    path = MyDroneRescueExample._select_peer_yield_path(
        drone, (0.0, 0.0), (45.0, 0.0)
    )
    assert path[0] == (-30.0, 0.0)
    assert path[-1] == (-120.0, 0.0)


def test_yield_path_rejects_a_route_toward_peer():
    drone = SimpleNamespace(_breadcrumbs=[(120.0, 0.0), (90.0, 0.0), (60.0, 0.0), (30.0, 0.0), (0.0, 0.0)])
    assert MyDroneRescueExample._select_peer_yield_path(
        drone, (0.0, 0.0), (45.0, 0.0)
    ) == []
