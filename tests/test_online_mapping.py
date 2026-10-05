"""Check that sensor evidence can add and remove a navigation obstacle."""

import numpy as np

from swarm_rescue.solutions.online_mapping import OnlineEvidenceMap


def test_static_hit_is_confirmed_and_later_free_rays_clear_it():
    evidence = OnlineEvidenceMap((240, 240), cell_size=8)
    for _ in range(2):
        evidence.update_scan((0.0, 0.0), 0.0, [80.0], [0.0], (), stride=1)
    cell = (15, 25)
    assert evidence.occupied[cell]
    assert evidence.bitmap[120, 200] == 1
    for _ in range(8):
        evidence.update_scan((0.0, 0.0), 0.0, [300.0], [0.0], (), stride=1)
    assert not evidence.occupied[cell]
    assert evidence.bitmap[120, 200] == 0


def test_semantically_identified_dynamic_return_is_not_a_wall():
    evidence = OnlineEvidenceMap((240, 240), cell_size=8)
    for _ in range(6):
        evidence.update_scan(
            (0.0, 0.0), 0.0, np.array([80.0]), np.array([0.0]),
            [(80.0, 0.0)], stride=1,
        )
    assert not evidence.occupied.any()
