"""Claims, physical loads and confirmed disposal must remain distinct."""
from types import SimpleNamespace
from swarm_rescue.solutions.task_progress import TaskProgressMonitor, valid_progress_snapshot
from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def test_broadcasting_does_not_renew_stalled_claim():
    monitor = TaskProgressMonitor()
    for step in range(301):
        monitor.update('claim', (1, 2), step, (0, 0), 100)
    assert monitor.current.advanced == 0
    assert monitor.stalled(300, 140)


def test_backtracking_loop_and_gps_jitter_do_not_count_forever():
    monitor = TaskProgressMonitor()
    for step in range(500):
        monitor.update('claim', (1, 2), step, (31.0 + step % 3, 0), 100 + step % 4)
    assert monitor.current.advanced == 0
    for step in range(500, 700):
        monitor.update('claim', (1, 2), step, (96.0 if step % 2 else 0.0, 0), 150)
    assert monitor.stalled(699, 140)


def test_real_detour_keeps_lease_even_when_straight_distance_increases():
    monitor = TaskProgressMonitor()
    for step in range(500):
        monitor.update('claim', (1, 2), step, (step, 0), 100 + step)
    assert not monitor.stalled(499, 140)


def test_approach_and_pickup_reset_progress_phase():
    monitor = TaskProgressMonitor()
    for step in range(100):
        monitor.update('claim', (1, 2), step, (step, 0), 100-step)
    assert not monitor.stalled(99, 50)
    state = monitor.update('carry', (1, 2), 100, (99, 0), 400)
    assert state.started == state.advanced == 100


def test_relay_heartbeat_cannot_refresh_old_progress_or_future_timestamp():
    assert not valid_progress_snapshot('claim', (1, 2), 999,
                                      ('claim', (1, 2), 0, 10, 999), 999, 140, 180)
    assert not valid_progress_snapshot('carry', (1, 2), 1000,
                                      ('carry', (1, 2), 0, 1001, 1001), 1000, 220, 180)
    assert valid_progress_snapshot('carry', (1, 2), 999,
                                   ('carry', (1, 2), 0, 990, 999), 999, 220, 180)


def _standby_scout():
    obj = SimpleNamespace(identifier=0, elapsed_timestep=1000, _max_steps=2000,
                          _EXPECTED_BOMBS=10, _TASK_LEASE_STEPS=180,
                          _team_deliveries={0: 8}, _disposed_tokens=set(),
                          _discovered_targets={}, _recovery_targets={},
                          _team_tasks={1: ('carry', (1, 1), None, 1000), 2: ('carry', (2, 2), None, 1000)},
                          _team_progress={1: ('carry', (1, 1), 0, 990, 1000), 2: ('carry', (2, 2), 0, 990, 1000)},
                          _peer_state={i: ((0, 0), 80, 1000, 0, True, (0, 0)) for i in (1, 2)},
                          grasped_bombs=lambda: [])
    obj._progress_timeout = lambda mode: MyDroneRescueExample._progress_timeout(obj, mode)
    obj._task_has_progress = lambda owner, task: MyDroneRescueExample._task_has_progress(obj, owner, task)
    return obj


def test_claims_cannot_mark_team_complete_or_authorize_standby():
    scout = _standby_scout()
    scout._team_tasks = {i: ('claim', (i, i), None, 1000) for i in (1, 2)}
    assert not MyDroneRescueExample._all_bombs_accounted_for(scout)
    assert not MyDroneRescueExample._should_wait_for_carriers(scout)


def test_moving_attached_loads_allow_temporary_standby_but_not_completion():
    scout = _standby_scout()
    assert MyDroneRescueExample._should_wait_for_carriers(scout)
    assert not MyDroneRescueExample._all_bombs_accounted_for(scout)


def test_stalled_carrier_or_drop_wakes_standby_scouts():
    scout = _standby_scout()
    scout._team_progress[1] = ('carry', (1, 1), 0, 500, 1000)
    assert not MyDroneRescueExample._should_wait_for_carriers(scout)
    scout = _standby_scout()
    scout._peer_state[1] = ((0, 0), 80, 1000, 0, False, (0, 0))
    assert not MyDroneRescueExample._should_wait_for_carriers(scout)


def test_unknown_target_prevents_standby_and_rewards_confirm_completion():
    scout = _standby_scout()
    scout._discovered_targets[(3, 3)] = ((0, 0), 1000)
    assert not MyDroneRescueExample._should_wait_for_carriers(scout)
    scout._team_deliveries[0] = 10
    assert MyDroneRescueExample._all_bombs_accounted_for(scout)
