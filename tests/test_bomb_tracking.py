"""Nearby bombs must remain separate tasks despite localization noise."""

from types import SimpleNamespace

from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def test_nearby_bombs_do_not_share_a_claim_token():
    tracker = SimpleNamespace(_bomb_token_centers={})
    first = MyDroneRescueExample._token_for_world(tracker, (-513.3, -245.0))
    second = MyDroneRescueExample._token_for_world(tracker, (-486.4, -254.6))
    assert first != second
    assert MyDroneRescueExample._token_for_world(tracker, (-511.0, -246.0)) == first
    assert MyDroneRescueExample._token_for_world(tracker, (-488.0, -253.0)) == second


def test_quantization_collision_creates_a_distinct_token():
    tracker = SimpleNamespace(_bomb_token_centers={(0, 0): (-30.0, 0.0)})
    token = MyDroneRescueExample._token_for_world(tracker, (0.0, 0.0))
    assert token != (0, 0)

