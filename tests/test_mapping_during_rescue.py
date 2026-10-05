"""Mapping must not postpone a bomb that can be chased this frame."""

from types import SimpleNamespace

from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def test_visible_bomb_is_pursued_in_the_same_step_as_map_updates():
    events = []
    bomb = (object(), (10.0, 20.0))
    drone = SimpleNamespace(
        elapsed_timestep=0,
        _handoff_hold_until=-1,
        _consume_disposal_reward=lambda: None,
        _merge_team_status=lambda: None,
        _update_active_bomb_observations=lambda: None,
        _update_shared_disposal=lambda: None,
        _update_online_obstacles=lambda: events.append("obstacles"),
        _update_frontier_map=lambda: events.append("coverage"),
        _diagnostic_snapshot=lambda: None,
        _refresh_own_team_state=lambda: None,
        _own_task_stalled=lambda *_: False,
        _set_operational_stage=lambda *_: None,
        _empty_command=lambda: {},
        _is_active_drone=lambda: True,
        grasped_bombs=lambda: [],
        _was_carrying=False,
        _record_breadcrumb=lambda: None,
        _should_retire_low_health=lambda: False,
        _select_unclaimed_bomb=lambda: bomb,
        _token_for_world=lambda _: (1, 2),
        _current_bomb_token=None,
        _last_claim_token=None,
        _claim_age=0,
        _max_steps=5000,
        _claim_cooldown_until={},
        _set_own_task=lambda *_: None,
        _control_chase_bomb=lambda cmd, _: events.append("chase") or cmd,
    )
    MyDroneRescueExample.control(drone)
    assert events == ["obstacles", "coverage", "chase"]
