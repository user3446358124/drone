from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
import math
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.motion_control import brake_command
from swarm_rescue.solutions.online_mapping import OnlineEvidenceMap
from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def test_requested_stop_brakes_in_both_axes_without_changing_grasp_or_turn():
    cmd={'forward':0.,'lateral':0.,'rotation':0.5,'grasper':1}
    out=brake_command(cmd,(3,4),0)
    assert out['forward'] < 0 and out['lateral'] < 0
    assert np.hypot(out['forward'],out['lateral']) <= 1.000001
    assert out['rotation']==0.5 and out['grasper']==1 and cmd['forward']==0


def test_braking_uses_drone_frame_and_preserves_sensor_loss_fallback():
    cmd={'forward':0.,'lateral':0.,'rotation':0.,'grasper':0}
    out=brake_command(cmd,(0,3),math.pi/2)
    assert out['forward'] < 0 and abs(out['lateral']) < 1e-10
    assert brake_command(cmd,None,0)==cmd
    assert brake_command(cmd,(float('nan'),0),0)==cmd
    assert brake_command(cmd,(0.1,0),0)==cmd


def test_semantic_classification_erases_a_previous_wall_hypothesis():
    grid=OnlineEvidenceMap((240,240))
    for _ in range(4):
        grid.update_scan((0,0),0,[80,80],[0,math.pi/2],(),stride=1)
    assert grid.occupied[15,25] and grid.occupied[5,15]
    hits, clears=grid.update_scan((0,0),0,[80,80],[0,math.pi/2],[(80,0)],stride=1)
    assert not grid.occupied[15,25] and grid.occupied[5,15]
    assert (15,25) in clears
    assert (5,15) in hits


def test_distant_noisy_sighting_reuses_a_lead_but_close_bombs_stay_separate():
    obj=SimpleNamespace(_bomb_token_centers={(0,0):(0.,0.)})
    obj._token_for_world=lambda p: MyDroneRescueExample._token_for_world(obj,p)
    far=MyDroneRescueExample._token_for_detection(obj,SimpleNamespace(distance=200),(35,0))
    near=MyDroneRescueExample._token_for_detection(obj,SimpleNamespace(distance=40),(27,0))
    assert far==(0,0) and near!=(0,0)


def test_close_claims_share_one_approach_but_next_bomb_can_be_taken_after_departure():
    token=(5,5)
    obj=SimpleNamespace(identifier=1,elapsed_timestep=20,drone_health=50,
        _TASK_LEASE_STEPS=180,_target_invalid=lambda _:False,
        _opening_role_accepts=lambda *_:True,_peer_claim_stalled=lambda *_:False,
        measured_gps_position=lambda:(0,0),_peer_state={},
        _target_bid=lambda *_:0,
        _team_tasks={0:('claim',(4,4),(120,0),20),1:('claim',token,(100,0),20)})
    assert not MyDroneRescueExample._can_claim_target(obj,token,(100,0))
    obj._team_tasks[0]=('carry',(4,4),(120,0),20)
    assert MyDroneRescueExample._can_claim_target(obj,token,(100,0))


def test_lidar_range_is_not_mistaken_for_semantic_search_coverage():
    obj=SimpleNamespace(identifier=0,elapsed_timestep=0,size_area=(800,800),
        _FRONTIER_CELL_SIZE=36,_frontier_grid=np.zeros((23,23),dtype=np.uint8),
        measured_gps_position=lambda:np.array([0.,0.]), measured_compass_angle=lambda:0.,
        lidar_values=lambda:np.array([300.]),lidar_rays_angles=lambda:np.array([0.]),
        semantic_values=lambda:[],communicator=SimpleNamespace(received_messages=[]),
        _peer_frontier_targets={})
    obj._frontier_world_to_cell=lambda p: MyDroneRescueExample._frontier_world_to_cell(obj,p)
    MyDroneRescueExample._update_frontier_map(obj)
    assert obj._frontier_grid[obj._frontier_world_to_cell((100,0))]==1
    assert obj._frontier_grid[obj._frontier_world_to_cell((260,0))]==0


def test_noisy_target_updates_do_not_force_replanning_on_every_step():
    calls=[]
    obj=SimpleNamespace(_ARRIVE_RADIUS=10,_PATROL_ARRIVE_RADIUS=20,
        _nav_age=1,_nav_target=(100.,0.),_nav_path=[(80.,0.)],_nav_plan_failed=False,
        _lidar_blocked_steps=0,_run_lidar_escape_if_needed=lambda _:False,measured_gps_position=lambda:(0.,0.),
        grasped_bombs=lambda:[],_plan_navigation=lambda *_:calls.append(True) or [(140.,0.)],
        _move_toward=lambda *_args,**_kw:None)
    MyDroneRescueExample._navigate_online(obj,(140.,0.),{})
    assert not calls
    obj._nav_age=12
    MyDroneRescueExample._navigate_online(obj,(140.,0.),{})
    assert calls==[True]


def test_pending_escape_runs_even_after_trigger_cleared_the_block_counter():
    obj=SimpleNamespace(_ARRIVE_RADIUS=10.,_nav_age=0,_lidar_blocked_steps=0,
        measured_gps_position=lambda:(0.,0.),
        _run_lidar_escape_if_needed=lambda cmd:cmd.update(forward=-0.55) or True)
    cmd={}
    assert not MyDroneRescueExample._navigate_online(obj,(100.,0.),cmd)
    assert cmd['forward']==-0.55 and obj._nav_age==1


def test_grasper_is_enabled_only_at_an_aligned_close_selected_target():
    obj=SimpleNamespace(elapsed_timestep=10,semantic_values=lambda:[])
    check=lambda distance,angle:MyDroneRescueExample._should_enable_grasper(obj,SimpleNamespace(distance=distance,angle=angle))
    assert not check(100,0) and not check(20,1)
    assert check(20,0)
    obj.semantic_values=lambda:[SimpleNamespace(entity_type=DroneSemanticSensor.TypeEntity.BOMB,grasped=False,distance=10)]
    assert not check(30,0)
    obj.elapsed_timestep=2
    assert not check(20,0)


def _reservation_claim():
    return SimpleNamespace(identifier=1,elapsed_timestep=20,drone_health=50,
        _TASK_LEASE_STEPS=180,_target_invalid=lambda _:False,
        _opening_role_accepts=lambda *_:True,
        measured_gps_position=lambda:(0,0),_peer_state={},
        _target_bid=lambda *_:0,
        _team_tasks={0:('claim',(4,4),(120,0),20),1:('claim',(5,5),(100,0),20)})


def test_stalled_neighbor_reservation_uses_its_actual_token():
    obj=_reservation_claim()
    obj._peer_claim_stalled=lambda owner,token:token==(4,4)
    assert MyDroneRescueExample._can_claim_target(obj,(5,5),(100,0))


def test_attached_load_has_priority_over_a_lower_identifier_claim():
    obj=_reservation_claim()
    obj.identifier=0
    obj._peer_claim_stalled=lambda *_:False
    obj._team_tasks={0:('claim',(5,5),(100,0),20),1:('carry',(4,4),(120,0),20)}
    obj._peer_state={1:((140,0),50,20,0.,True,(0.,0.))}
    assert not MyDroneRescueExample._can_claim_target(obj,(5,5),(100,0))


def test_visible_far_bomb_uses_navigation_then_local_alignment():
    calls=[]
    obj=SimpleNamespace(_carry_push_steps=0,_reset_explore_state=lambda:None,
        _should_enable_grasper=lambda _:False,_run_lidar_escape_if_needed=lambda _:False,
        _navigate_online=lambda point,cmd:calls.append(('plan',point)),
        _approach_bomb_frontally=lambda cmd,det:calls.append(('align',det.distance)))
    MyDroneRescueExample._control_chase_bomb(obj,{},(SimpleNamespace(distance=100),(100,0)))
    MyDroneRescueExample._control_chase_bomb(obj,{},(SimpleNamespace(distance=30),(30,0)))
    assert calls==[('plan',(100,0)),('align',30)]

