"""A locally rejected old position must not be resurrected by radio relays."""
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def _tracker():
    obj = SimpleNamespace(_invalid_targets={}, _discovered_targets={}, _recovery_targets={},
                          _disposed_tokens=set(), _diagnostic_event=lambda *_args, **_kw: None)
    obj._invalidate_target = lambda token, step, **kw: MyDroneRescueExample._invalidate_target(obj, token, step, **kw)
    return obj


def test_negative_observation_removes_old_recovery_and_discovery_not_disposal():
    obj = _tracker()
    obj._discovered_targets[(1, 2)] = ((100, 100), 50)
    obj._recovery_targets[(1, 2)] = ((100, 100), 60)
    obj._invalidate_target((1, 2), 100)
    assert not obj._discovered_targets and not obj._recovery_targets
    assert MyDroneRescueExample._target_invalid(obj, (1, 2))
    assert not obj._disposed_tokens


def test_new_or_simultaneous_positive_observation_remains_actionable():
    obj = _tracker()
    obj._discovered_targets[(1, 2)] = ((100, 100), 100)
    obj._invalidate_target((1, 2), 100)
    assert not MyDroneRescueExample._target_invalid(obj, (1, 2))
    obj._discovered_targets[(1, 2)] = ((110, 100), 101)
    obj._invalidate_target((1, 2), 99)
    assert not MyDroneRescueExample._target_invalid(obj, (1, 2))
    assert obj._invalid_targets[(1, 2)] == 100


def test_old_positive_relay_cannot_resurrect_invalidated_target():
    obj = _tracker()
    obj._team_size=10
    obj.identifier=0
    obj.elapsed_timestep=100
    obj._peer_state={}
    obj._team_deliveries={}
    obj._bomb_token_centers={}
    obj._team_tasks={}
    obj._team_progress={}
    obj.communicator=SimpleNamespace(received_messages=[(None, {
        'id': 1, 'step': 100, 'health':80, 'position':(0,0), 'heading':0, 'velocity':(0,0),
        'invalid_targets':(((1,2),90),),
        'discoveries':(((1,2),(100,100),80),),
        'recovery_targets':(((1,2),(100,100),80),), 'tasks':(), 'task_progress':()
    })])
    MyDroneRescueExample._merge_team_status(obj)
    assert not obj._discovered_targets and not obj._recovery_targets
    assert MyDroneRescueExample._target_invalid(obj, (1,2))
    obj.communicator.received_messages[0][1]['discoveries']=(((1,2),(100,100),95),)
    MyDroneRescueExample._merge_team_status(obj)
    assert not MyDroneRescueExample._target_invalid(obj, (1,2))


def test_nearby_position_occluded_by_wall_cannot_be_declared_empty():
    obj=SimpleNamespace(measured_gps_position=lambda: np.array([0,0]),
                        measured_compass_angle=lambda: 0,
                        lidar_values=lambda: np.array([8,8,8]),
                        lidar_rays_angles=lambda: np.array([-0.05,0,0.05]))
    assert not MyDroneRescueExample._target_area_visible(obj,(25,0))
    obj.lidar_values=lambda: None
    assert not MyDroneRescueExample._target_area_visible(obj,(25,0))


def test_clear_local_target_area_can_support_negative_observation():
    obj=SimpleNamespace(measured_gps_position=lambda: np.array([0,0]),
                        measured_compass_angle=lambda: 0,
                        lidar_values=lambda: np.array([100,100,100]),
                        lidar_rays_angles=lambda: np.array([-0.05,0,0.05]))
    assert MyDroneRescueExample._target_area_visible(obj,(25,0))
    assert not MyDroneRescueExample._target_area_visible(obj,(80,0))
