"""A real attached load needs physical recovery, not lease reassignment."""
from types import SimpleNamespace
from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample
from swarm_rescue.solutions.task_progress import TaskProgressMonitor


def _carrier(health=40):
    monitor=TaskProgressMonitor()
    monitor.update('carry',(1,2),0,(0,0),500)
    obj=SimpleNamespace(_current_bomb_token=(1,2), elapsed_timestep=800, drone_health=health,
                        _carrier_recovery_token=(1,2), _carrier_recovery_step=-10000,
                        _carrier_recovery_attempts=0, _task_monitor=monitor,
                        _known_disposal=(0,500), _return_path=[(0,0),(0,500)], _online_map=None,
                        _recovery_targets={}, _bomb_token_centers={}, _claim_cooldown_until={},
                        _handoff_hold_until=-1, measured_gps_position=lambda: (0,0),
                        _available_carrier_helper=lambda: None,
                        _own_task_stalled=lambda *_: True, _progress_timeout=lambda _:500,
                        _set_own_task=lambda *_:None,
                        _set_motion=lambda cmd,**kw:cmd.update(kw),
                        _diagnostic_event=lambda *_args,**_kw:None)
    obj._optimize_return_trail=lambda position,trail: MyDroneRescueExample._optimize_return_trail(obj,position,trail)
    return obj


def test_stall_recovery_preserves_a_new_certified_return_route():
    obj=_carrier()
    def optimized(position, trail):
        obj._return_route_verified=True
        return [(0,100),(0,500)]
    obj._optimize_return_trail=optimized
    cmd={'grasper':1}
    assert not MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    assert obj._return_path==[(0,100),(0,500)] and cmd['grasper']==1


def test_first_stall_replans_keeps_load_and_does_not_fake_progress():
    obj=_carrier()
    cmd={'grasper':1}
    assert not MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    assert not obj._return_path and obj._nav_target is None
    assert cmd['grasper']==1 and obj._task_monitor.current.advanced==0
    assert obj._carrier_recovery_attempts==1


def test_second_stall_with_available_helper_releases_and_publishes_recovery():
    obj=_carrier()
    cmd={'grasper':1}
    MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    obj.elapsed_timestep=1300
    obj._available_carrier_helper=lambda:3
    assert MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    assert cmd['grasper']==0 and obj._recovery_targets[(1,2)]==((0.,0.),1300)
    assert obj._current_bomb_token==(1,2)  # Actual drop will refine its position.
    assert obj._claim_cooldown_until[(1,2)]>1300


def test_low_health_stalled_load_can_handover_before_replan():
    obj=_carrier(12)
    obj._available_carrier_helper=lambda:3
    cmd={'grasper':1}
    assert MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    assert cmd['grasper']==0


def test_without_helper_or_recent_stall_do_not_release_load():
    obj=_carrier(12)
    cmd={'grasper':1}
    assert not MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    assert cmd['grasper']==1
    obj.elapsed_timestep=801
    assert not MyDroneRescueExample._recover_stalled_carrier(obj,cmd)
    assert obj._carrier_recovery_attempts==1
    obj._own_task_stalled=lambda *_:False
    assert not MyDroneRescueExample._recover_stalled_carrier(obj,cmd)


def test_shorter_plan_without_motion_cannot_renew_progress():
    monitor=TaskProgressMonitor()
    monitor.update('carry',(1,2),0,(0,0),500)
    monitor.update('carry',(1,2),500,(0,0),200)
    assert monitor.current.advanced==0
    assert monitor.stalled(500,220)


def test_busy_stale_or_distant_peer_is_not_a_handover_helper():
    obj=_carrier(12)
    obj._TASK_LEASE_STEPS=180
    obj._team_tasks={1:('claim',(4,4),None,800),2:('search',None,None,800),3:('search',None,None,800)}
    obj._peer_state={1:((50,0),40,800,0,False,(0,0)),2:((400,0),40,800,0,False,(0,0)),
                     3:((40,0),40,500,0,False,(0,0))}
    assert MyDroneRescueExample._available_carrier_helper(obj) is None
    obj._peer_state[3]=((40,0),24,800,0,False,(0,0))
    assert MyDroneRescueExample._available_carrier_helper(obj)==3
