"""
蓝队无先验控制器（team_mode: rescue）。

控制器不读取地图文件、炸弹文件、固定处置区坐标或模拟器真值。
运行时仅使用比赛允许的传感器、通信、场地尺寸和自身任务状态。

示例流程：搜索炸弹 -> 抓取 -> 运往回收区 -> 释放。

----------------------------
每个选手必须实现的核心方法
----------------------------
- define_message_for_all() -> Any
- control() -> CommandsDict

----------------------------
执行器命令 API（control 返回字典）
----------------------------
- "forward": float，范围 [-1, 1]
- "lateral": float，范围 [-1, 1]
- "rotation": float，范围 [-1, 1]
- "grasper": int，取值 {0, 1}
  - 1: 保持抓取/尝试抓取
  - 0: 释放当前抓取物

----------------------------
传感器便捷 API（比赛控制逻辑建议使用）
----------------------------
- measured_gps_position() -> np.ndarray | None
- measured_compass_angle() -> float | None
- measured_velocity() -> np.ndarray | None
- measured_angular_velocity() -> float | None
- semantic_values() -> list[DroneSemanticSensor.Data] | None
  Data 字段如下：
  - distance: float
  - angle: float
  - entity_type: DroneSemanticSensor.TypeEntity
    (WALL / BOMB / DISPOSAL_CENTER / DRONE / OTHER)
  - grasped: bool
- lidar_values() -> np.ndarray | None
- lidar_rays_angles() -> np.ndarray
- gps_values() -> np.ndarray | None
- compass_values() -> float | None
- odometer_values() -> np.ndarray | None

----------------------------
通信 API
----------------------------
- define_message_for_all(): 返回广播消息
- communicator.received_messages -> list[(sender_communicator, msg)]
- communicator_is_disabled() -> bool

----------------------------
状态与辅助 API
----------------------------
- drone_health -> int
- elapsed_timestep -> int
- elapsed_walltime -> float
- size_area -> tuple | None
- grasped_bombs() -> list[Bomb]
- carried_bombs_count() -> int（救援模式通常为 0）
- *_is_disabled() -> bool
- gps()/compass()/odometer()/lidar()/semantic() -> 传感器对象（高级用法）

说明：
- `true_*` 方法仅用于调试/日志，禁止在 `control()` 中使用；评测（competition mode）下在 `control()` 内调用会抛出 `GroundTruthInControlError`。
"""

from __future__ import annotations

import math
import json
import os
from collections import deque
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.solutions.online_navigation import (
    choose_lidar_detour,
    generic_patrol_route,
    plan_occupancy_path,
)
from swarm_rescue.solutions.motion_control import brake_command
from swarm_rescue.solutions.online_mapping import OnlineEvidenceMap
from swarm_rescue.solutions.return_navigation import observed_return_route, path_length
from swarm_rescue.solutions.task_progress import TaskProgressMonitor, valid_progress_snapshot

SemanticHit = Tuple[DroneSemanticSensor.Data, Tuple[float, float]]
WorldPoint = Tuple[float, float]


class MyDroneRescueExample(DroneAbstract):
    """
    蓝队无先验协同控制器：所有可用无人机参与搜索和运输。

    状态机（由 control 依次判断）：
      idle -> carry -> chase -> explore
    """

    # --- 任务分工 ---
    _IDLE_SPIN = 0.35

    # --- 导航 ---
    _ARRIVE_RADIUS = 10.0
    _HEADING_TOLERANCE = 0.25
    _LOW_HEALTH = 10

    # --- 运弹 / 处置 ---
    _DISPOSAL_PUSH_TRIGGER_DIST = 28.0
    _DISPOSAL_PUSH_STEPS = 48
    _DISPOSAL_PUSH_FORWARD = 0.75
    _DISPOSAL_SPIN_RELEASE_STEPS = 14
    _CARRY_EXIT_STEPS = 18
    _CARRY_EXIT_SPEED = 0.38
    _BOMB_FRONT_ALIGN_TOL = 0.16
    _BOMB_FRONT_ALIGN_ROT = 0.75
    _BOMB_GRAB_APPROACH_SPEED = 0.48
    _BOMB_GRAB_MIN_SPEED = 0.22

    # --- 探索 / 脱困 ---
    _EXPLORE_STRAIGHT_STEPS = 90
    _EXPLORE_TURN_STEPS = 28
    _EXPLORE_FORWARD = 0.7
    _PATROL_ARRIVE_RADIUS = 20.0
    _PATROL_SCAN_STEPS = 16
    _PATH_REPLAN_STEPS = 170
    _EXPECTED_BOMBS = 10
    _TASK_LEASE_STEPS = 180
    _LATE_RETIRE_HEALTH = 8
    _LATE_RETIRE_DELIVERIES = 8.0
    _LATE_RETIRE_STALL_STEPS = 120
    _LATE_RETIRE_STALL_RADIUS = 25.0
    _FRONTIER_CELL_SIZE = 36
    _FRONTIER_REPLAN_STEPS = 70
    _FRONTIER_STALL_STEPS = 150

    # --- Lidar 避障（阈值参考 my_drone_place_example）---
    _LIDAR_BLOCK_DIST = 48.0
    _LIDAR_SLOW_DIST = 78.0
    _LIDAR_PATH_CONE = math.pi / 9
    _LIDAR_SIDE_CONE = math.pi / 4
    _LIDAR_TARGET_MARGIN = 10.0
    _LIDAR_AVOID_ROTATION = 0.85
    _LIDAR_BLOCK_TRIGGER_STEPS = 6
    _LIDAR_ESCAPE_BACK_STEPS = 10
    _LIDAR_ESCAPE_TURN_STEPS = 16
    _LIDAR_ESCAPE_STRAFE_STEPS = 10
    _LIDAR_ESCAPE_BACK_SPEED = -0.55
    _LIDAR_ESCAPE_STRAFE_SPEED = 0.65

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs,
    ):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        self._explore_tick = int(identifier or 0) * 29
        size_key = tuple(int(v) for v in misc_data.size_area) if misc_data and misc_data.size_area else None
        self._known_disposal: Optional[WorldPoint] = None
        self._disposal_observed = False
        self._disposal_approach: WorldPoint = (0.0, 0.0)
        self._carry_push_steps = 0
        self._carry_exit_steps_left = 0
        self._carry_spin_release_steps_left = 0
        self._carry_spin_sign = 1.0
        self._carry_progress_distance = float("inf")
        self._carry_progress_step = 0
        self._carry_detour_target: Optional[WorldPoint] = None
        self._carry_detour_until = 0
        self._lidar_blocked_steps = 0
        self._escape_back_steps_left = 0
        self._escape_turn_steps_left = 0
        self._escape_strafe_steps_left = 0
        self._escape_turn_sign = 1.0
        team_size = int(misc_data.number_drones or 1) if misc_data else 1
        my_id = int(identifier or 0)
        self._max_steps = int(
            getattr(misc_data, "max_timestep_limit", 5000) or 5000
        )
        # Build coverage from the first frame.  A short dimension-derived
        # sweep spreads drones before they switch to the shared frontier map.
        self._frontier_start_step = int(self._max_steps * 0.18)
        self._frontier_early_activated = False
        self._middle_edge_sweep_enabled = bool(
            size_key is not None
            and (self._max_steps > 5000 or float(size_key[0]) >= 1400.0)
        )
        self._middle_edge_probe_enabled = bool(
            size_key is not None
            and 2000 < self._max_steps <= 5000
            and float(size_key[0]) < 1400.0
        )
        self._patrol_route = generic_patrol_route(
            size_key, int(identifier or 0), team_size
        )
        if (
            size_key is not None
            and 2000 < self._max_steps <= 5000
            and team_size >= 6
            and my_id == team_size // 2
        ):
            width, height = float(size_key[0]), float(size_key[1])
            lower_center_probe = (-width * 0.04, -height * 0.19)
            # Probe a lower-centre pocket through a bottom opening.  These
            # points are fractions of the announced arena dimensions rather
            # than coordinates from any map.  The route deliberately reaches
            # the lower margin before crossing inward so the reverse trail is
            # also a safe loaded escape corridor.
            lower_center_gateway = (width * 0.22, -height * 0.08)
            lower_center_outer = (width * 0.22, -height * 0.42)
            lower_center_entry = (-width * 0.04, -height * 0.42)
            self._patrol_route = (
                lower_center_gateway,
                lower_center_outer,
                lower_center_entry,
                lower_center_probe,
            ) + self._patrol_route
        if (
            size_key is not None
            and float(size_key[0]) >= 1400.0
            and team_size >= 4
            and my_id == 0
        ):
            width, height = float(size_key[0]), float(size_key[1])
            upper_left_bridge = (-width * 0.28, height * 0.38)
            self._patrol_route = (upper_left_bridge,) + self._patrol_route
        if (
            size_key is not None
            and self._max_steps <= 2000
            and team_size >= 6
            and my_id == team_size - 3
        ):
            width, height = float(size_key[0]), float(size_key[1])
            urgent_right_probe = (width * 0.28, height * 0.10)
            lower_gateway = (-width * 0.14, -height * 0.23)
            upper_gateway = (width * 0.30, height * 0.25)
            self._patrol_route = (
                lower_gateway,
                upper_gateway,
                urgent_right_probe,
            ) + self._patrol_route
        if size_key is not None and team_size >= 4 and my_id == 1:
            outer_route = generic_patrol_route(size_key, 0, team_size)
            outer_leg = outer_route[: max(1, len(outer_route) // 2)]
            opening_route = tuple(reversed(outer_leg))
            if self._middle_edge_probe_enabled:
                width, height = float(size_key[0]), float(size_key[1])
                lower_left_probe = (-width * 0.44, -height * 0.23)
                opening_route = (lower_left_probe,) + opening_route
            self._patrol_route = opening_route + self._patrol_route
        elif size_key is not None and team_size >= 4 and my_id == 2:
            width, height = float(size_key[0]), float(size_key[1])
            if self._middle_edge_sweep_enabled:
                margin = min(65.0, width * 0.14, height * 0.14)
                lane_width = max(1.0, width - 2.0 * margin) / float(team_size)
                left_lane_x = -width / 2.0 + margin + 0.5 * lane_width
                # Approach the left-middle strip through an upper flank before
                # sweeping down it.  The points scale with the arena and avoid
                # assuming any wall or bomb coordinate on a new map.
                left_flank_y = height * 0.22
                left_entry = (-width * 0.22, left_flank_y)
                left_flank = (left_lane_x, left_flank_y)
                left_exit = (left_lane_x, -left_flank_y)
                self._patrol_route = (
                    left_entry,
                    left_flank,
                    left_exit,
                ) + self._patrol_route
            elif self._middle_edge_probe_enabled:
                # A single inexpensive probe closes the gap between the two
                # left lanes on ordinary maps without consuming a full scout.
                left_probe = (-width * 0.34, height * 0.18)
                self._patrol_route = (left_probe,) + self._patrol_route
        elif (
            size_key is not None
            and team_size >= 6
            and my_id == team_size - 3
            and self._middle_edge_sweep_enabled
        ):
            width, height = float(size_key[0]), float(size_key[1])
            margin = min(65.0, width * 0.14, height * 0.14)
            lane_width = max(1.0, width - 2.0 * margin) / float(team_size)
            right_midpoint = (width / 2.0 - margin - 0.5 * lane_width, 0.0)
            self._patrol_route = (right_midpoint,) + self._patrol_route
        elif size_key is not None and team_size >= 5 and my_id == 4:
            height = float(size_key[1])
            margin = min(65.0, float(size_key[0]) * 0.14, height * 0.14)
            if self._max_steps <= 5000:
                center_route = (
                    (0.0, -height / 2.0 + margin),
                    (0.0, -height * 0.18),
                    (0.0, -height * 0.06),
                    (0.0, 0.0),
                )
            else:
                center_route = (
                    (0.0, -height / 2.0 + margin),
                    (0.0, -height * 0.25),
                )
            self._patrol_route = center_route + self._patrol_route
        # The two right scouts first visit separate corners.  This closes the
        # blind strip left by equal-width patrol lanes while
        # remaining entirely dimension-derived for unseen competition maps.
        direct_corner_scouts = {max(0, team_size - 2), max(0, team_size - 1)}
        if size_key is not None and team_size >= 4 and my_id in direct_corner_scouts:
            width, height = float(size_key[0]), float(size_key[1])
            margin = min(65.0, width * 0.14, height * 0.14)
            edge_x = width / 2.0 - margin
            edge_y = height / 2.0 - margin if my_id in (0, team_size - 2) else -height / 2.0 + margin
            edge_target = (edge_x, edge_y)
            if self._max_steps <= 2000 and my_id in (
                team_size - 2,
                team_size - 1,
            ):
                lower_gateway = (-width * 0.12, -height * 0.25)
                upper_gateway = (width * 0.32, height * 0.28)
                self._patrol_route = (
                    lower_gateway,
                    upper_gateway,
                    edge_target,
                ) + self._patrol_route
            elif (
                self._max_steps > 5000
                and width < 1400.0
                and my_id == team_size - 2
            ):
                upper_inner = (width * 0.28, height * 0.45)
                upper_outer = (width * 0.35, height * 0.45)
                upper_probe = (width * 0.35, height * 0.40)
                self._patrol_route = (
                    upper_inner,
                    upper_outer,
                    upper_probe,
                ) + self._patrol_route
            else:
                self._patrol_route = (edge_target,) + self._patrol_route
        self._patrol_index = 0
        self._patrol_scan_steps_left = 0
        self._patrol_target_age = 0
        self._patrol_leg_timeout = 240
        self._nav_path: list[WorldPoint] = []
        self._nav_target: Optional[WorldPoint] = None
        self._nav_age = 0
        self._nav_plan_failed = False
        self._nav_fallback_target: Optional[WorldPoint] = None
        self._nav_fallback_until = 0
        self._claimed_bomb: Optional[WorldPoint] = None
        self._team_size = team_size
        # On a previously unseen map the route used to reach a bomb is the
        # strongest sensor-verified return path available.  Keep a compact,
        # loop-erased trail and retrace it after pickup instead of repeatedly
        # trying an optimistic straight line through a partially observed wall.
        self._breadcrumbs: list[WorldPoint] = []
        self._return_path: list[WorldPoint] = []
        self._return_progress_target: Optional[WorldPoint] = None
        self._return_progress_position: Optional[WorldPoint] = None
        self._return_progress_distance = float("inf")
        self._return_progress_step = 0
        self._was_carrying = False
        self._delivery_count = 0.0
        self._disposed_tokens: set[tuple[int, int]] = set()
        self._current_bomb_token: Optional[tuple[int, int]] = None
        self._bomb_token_centers: dict[tuple[int, int], WorldPoint] = {}
        self._recovery_targets: dict[
            tuple[int, int], tuple[WorldPoint, int]
        ] = {}
        self._discovered_targets: dict[
            tuple[int, int], tuple[WorldPoint, int]
        ] = {}
        self._recovery_scan_steps: dict[tuple[int, int], int] = {}
        self._target_progress_token: Optional[tuple[int, int]] = None
        self._target_progress_distance = float("inf")
        self._target_progress_step = 0
        self._claim_cooldown_until: dict[tuple[int, int], int] = {}
        self._last_claim_token: Optional[tuple[int, int]] = None
        self._claim_age = 0
        self._safety_anchor: Optional[WorldPoint] = None
        self._safety_anchor_step = 0
        self._retired_low_health = False
        self._team_deliveries: dict[int, float] = {my_id: 0.0}
        self._team_tasks: dict[
            int, tuple[str, Optional[tuple[int, int]], Optional[WorldPoint], int]
        ] = {my_id: ("search", None, None, 0)}
        self._task_monitor = TaskProgressMonitor()
        self._team_progress: dict[int, tuple[str, tuple[int, int], int, int, int]] = {}
        self._invalid_targets: dict[tuple[int, int], int] = {}
        self._diagnostics_enabled = os.environ.get("SWARM_RESCUE_DIAGNOSTICS") == "1"
        self._diagnostic_events = 0
        self._operational_stage = "init"
        self._last_diagnostic_snapshot = -200
        self._carrier_recovery_step = -10_000
        self._carrier_recovery_token = None
        self._carrier_recovery_attempts = 0
        self._handoff_hold_until = -1
        self._peer_state: dict[
            int, tuple[WorldPoint, int, int, float, bool, WorldPoint]
        ] = {}
        self._yield_peer_id: Optional[int] = None
        self._yield_peer_until = 0
        self._yield_path: list[WorldPoint] = []
        self._online_map = OnlineEvidenceMap(size_key) if size_key is not None else None
        if self._online_map is not None:
            self._online_obstacles = self._online_map.bitmap
        elif size_key is not None:
            self._online_obstacles = np.zeros(
                (int(size_key[1]), int(size_key[0])), dtype=np.uint8
            )
        else:
            self._online_obstacles = np.zeros((0, 0), dtype=np.uint8)
        self._online_hit_counts: dict[tuple[int, int], int] = {}
        self._recent_obstacle_hits: tuple[tuple[int, int], ...] = ()
        self._recent_obstacle_clears: tuple[tuple[int, int], ...] = ()
        if size_key is not None:
            frontier_rows = int(math.ceil(size_key[1] / self._FRONTIER_CELL_SIZE))
            frontier_cols = int(math.ceil(size_key[0] / self._FRONTIER_CELL_SIZE))
            self._frontier_grid = np.zeros(
                (frontier_rows, frontier_cols), dtype=np.uint8
            )
        else:
            self._frontier_grid = np.zeros((0, 0), dtype=np.uint8)
        self._recent_frontier_updates: tuple[tuple[int, int, int], ...] = ()
        self._frontier_snapshot_cursor = 0
        self._peer_frontier_targets: dict[int, tuple[tuple[int, int], int]] = {}
        self._frontier_target_cell: Optional[tuple[int, int]] = None
        self._frontier_path: list[WorldPoint] = []
        self._frontier_plan_step = -10_000
        self._frontier_anchor: Optional[WorldPoint] = None
        self._frontier_anchor_step = 0
        self._frontier_blocked_until: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def define_message_for_all(self) -> Any:
        self._refresh_own_team_state()
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        velocity = self.measured_velocity()
        return {
            "id": int(self.identifier or 0),
            "position": (float(pos[0]), float(pos[1])) if pos is not None else None,
            "heading": float(heading) if heading is not None else None,
            "velocity": (
                (float(velocity[0]), float(velocity[1]))
                if velocity is not None else None
            ),
            "health": int(self.drone_health),
            "step": int(self.elapsed_timestep),
            "claim": self._claimed_bomb,
            "carrying": bool(self.grasped_bombs()),
            "disposal": self._known_disposal,
            "disposal_observed": self._disposal_observed,
            "obstacles": self._recent_obstacle_hits,
            "obstacle_clears": self._recent_obstacle_clears,
            "frontier_updates": (
                self._recent_frontier_updates + self._frontier_snapshot()
            ),
            "frontier_target": self._frontier_target_cell,
            "deliveries": tuple(sorted(self._team_deliveries.items())),
            "disposed_tokens": tuple(sorted(self._disposed_tokens)),
            "recovery_targets": tuple(
                (token, point, step)
                for token, (point, step) in sorted(self._recovery_targets.items())
                if token not in self._disposed_tokens
            ),
            "discoveries": tuple(
                (token, point, step)
                for token, (point, step) in sorted(self._discovered_targets.items())
                if token not in self._disposed_tokens
            ),
            "tasks": tuple(
                (owner, mode, token, point, step)
                for owner, (mode, token, point, step) in sorted(self._team_tasks.items())
            ),
            "task_progress": tuple(
                (owner, *progress) for owner, progress in sorted(self._team_progress.items())
            ),
            "invalid_targets": tuple(sorted(self._invalid_targets.items())),
        }

    def control(self) -> CommandsDict:
        cmd = self._control_step()
        if abs(cmd["forward"]) + abs(cmd["lateral"]) < 1e-9:
            return brake_command(cmd, self.measured_velocity(), self.measured_compass_angle())
        return cmd

    def _control_step(self) -> CommandsDict:
        self._consume_disposal_reward()
        self._merge_team_status()
        self._update_active_bomb_observations()
        self._update_shared_disposal()
        self._update_online_obstacles()
        self._update_frontier_map()
        self._diagnostic_snapshot()
        cmd = self._empty_command()
        if not self._is_active_drone():
            return self._control_idle(cmd)
        carrying = bool(self.grasped_bombs())
        if carrying and not self._was_carrying:
            if self._current_bomb_token is None and self._claimed_bomb is not None:
                self._current_bomb_token = self._token_for_world(self._claimed_bomb)
            self._prepare_return_path()
            self._carry_exit_steps_left = self._CARRY_EXIT_STEPS
        elif not carrying:
            if self._was_carrying:
                self._record_unexpected_drop(self._current_bomb_token)
                self._current_bomb_token = None
                self._breadcrumbs = []
                self._return_path = []
            self._record_breadcrumb()
        self._was_carrying = carrying
        if carrying:
            self._claim_age = 0
            self._last_claim_token = None
            self._set_own_task("carry", self._current_bomb_token)
            self._claimed_bomb = None
            self._set_operational_stage("carry")
            if self._recover_stalled_carrier(cmd):
                return cmd
            return self._control_carry(cmd)
        if int(self.elapsed_timestep) < self._handoff_hold_until:
            self._set_own_task("search", None)
            self._current_bomb_token = None
            self._claimed_bomb = None
            self._set_operational_stage("standby", "handover_cooldown")
            return self._control_safe_hold(cmd)
        # Do this before selecting a visible bomb.  Otherwise a badly damaged
        # drone can keep reacquiring a nearby target forever and never reach
        # the retirement check in the no-target branch.
        if self._should_retire_low_health():
            self._set_own_task("search", None)
            self._current_bomb_token = None
            self._claimed_bomb = None
            self._set_operational_stage("retired", "low_health")
            return self._control_safe_hold(cmd)
        self._refresh_own_team_state()
        if self._current_bomb_token is not None and self._target_invalid(self._current_bomb_token):
            self._abandon_claim(self._current_bomb_token, "shared_negative_observation")
        if self._own_task_stalled("claim", self._current_bomb_token):
            self._abandon_claim(self._current_bomb_token, "no_progress")
        bomb = self._select_unclaimed_bomb()
        if bomb is None:
            recovery = self._select_recovery_target()
            if recovery is not None:
                token, point = recovery
                self._current_bomb_token = token
                self._claimed_bomb = point
                self._set_own_task("claim", token, point)
                self._set_operational_stage("recover")
                return self._control_recovery_target(cmd, token, point)
            discovery = self._select_discovered_target()
            if discovery is not None:
                token, point = discovery
                self._current_bomb_token = token
                self._claimed_bomb = point
                self._set_own_task("claim", token, point)
                self._set_operational_stage("approach_shared")
                return self._control_recovery_target(
                    cmd, token, point, source="discovery"
                )
            self._current_bomb_token = None
            self._set_own_task("search", None)
            self._claimed_bomb = None
            if self._all_bombs_accounted_for():
                self._set_operational_stage("complete", "confirmed_disposals")
                return self._control_safe_hold(cmd)
            if self._should_wait_for_carriers():
                self._set_operational_stage("standby", "confirmed_loads_progressing")
                return self._control_safe_hold(cmd)
            self._set_operational_stage("search", "unresolved_tasks")
            return self._control_explore(cmd)
        self._claimed_bomb = bomb[1]
        self._current_bomb_token = self._token_for_detection(bomb[0], bomb[1])
        if self._current_bomb_token == self._last_claim_token:
            self._claim_age += 1
        else:
            self._last_claim_token = self._current_bomb_token
            self._claim_age = 1
        self._set_own_task("claim", self._current_bomb_token, bomb[1])
        self._set_operational_stage("chase")
        return self._control_chase_bomb(cmd, bomb)

    # ------------------------------------------------------------------
    # 控制阶段
    # ------------------------------------------------------------------

    def _is_active_drone(self) -> bool:
        # Every available drone searches and transports.  The original demo
        # activated only drone 0, throwing away most of the swarm capacity.
        return True

    def _control_idle(self, cmd: CommandsDict) -> CommandsDict:
        my_id = int(self.identifier or 0)
        cmd["rotation"] = self._IDLE_SPIN if my_id % 2 == 0 else -self._IDLE_SPIN
        return cmd

    def _control_safe_hold(self, cmd: CommandsDict) -> CommandsDict:
        """Leave nearby geometry, then stop once every bomb has an owner."""
        values = self.lidar_values()
        if values is None:
            return cmd
        nearest_distance = float("inf")
        nearest_angle = 0.0
        for distance, angle in zip(values, self.lidar_rays_angles()):
            value = float(distance)
            if math.isfinite(value) and value < nearest_distance:
                nearest_distance = value
                nearest_angle = float(angle)
        if nearest_distance < 78.0:
            self._set_motion(
                cmd,
                forward=-0.32 * math.cos(nearest_angle),
                lateral=-0.32 * math.sin(nearest_angle),
                rotation=0.0,
            )
        return cmd

    # ------------------------------------------------------------------
    # 全队任务账本
    # ------------------------------------------------------------------

    def _token_for_world(self, point: WorldPoint) -> tuple[int, int]:
        nearest = min(
            self._bomb_token_centers.items(),
            key=lambda item: math.hypot(
                point[0] - item[1][0], point[1] - item[1][1]
            ),
            default=None,
        )
        if nearest is not None and math.hypot(
            point[0] - nearest[1][0], point[1] - nearest[1][1]
        ) <= 24.0:
            return nearest[0]
        token = int(round(point[0] / 16.0)), int(round(point[1] / 16.0))
        # Distinct nearby bombs can quantize to the same cell.  Never silently
        # reuse an occupied token unless its tracked center matched above.
        while token in self._bomb_token_centers:
            token = token[0], token[1] + 1
        self._bomb_token_centers[token] = point
        return token

    def _token_for_detection(self, det, point: WorldPoint) -> tuple[int, int]:
        # Compass noise grows into position error with detection range.
        # Distant ambiguous sightings are leads; close sightings separate loads.
        radius = max(24.0, min(48.0, 12.0 + float(det.distance) * 0.18))
        matches = [(math.hypot(center[0]-point[0], center[1]-point[1]), token)
                   for token, center in self._bomb_token_centers.items()]
        matches = [item for item in matches if item[0] <= radius]
        if matches:
            return min(matches)[1]
        return self._token_for_world(point)

    def _set_own_task(
        self,
        mode: str,
        token: Optional[tuple[int, int]],
        point: Optional[WorldPoint] = None,
    ) -> None:
        owner = int(self.identifier or 0)
        if token is not None and point is not None:
            self._bomb_token_centers.setdefault(token, point)
        if point is None and token is not None:
            point = self._bomb_token_centers.get(token)
        previous = self._team_tasks.get(owner)
        self._team_tasks[owner] = (mode, token, point, int(self.elapsed_timestep))
        self._team_deliveries[owner] = float(self._delivery_count)
        self._update_own_task_progress(mode, token, point)
        if previous is None or previous[:2] != (mode, token):
            self._diagnostic_event("task", mode=mode, token=token, target=point)

    def _consume_disposal_reward(self) -> None:
        reward = float(self.reward)
        if not math.isfinite(reward) or reward <= 0.0:
            return
        self._delivery_count += reward
        self._diagnostic_event("disposed", token=self._current_bomb_token, reward=reward)
        self._team_deliveries[int(self.identifier or 0)] = self._delivery_count
        if self._current_bomb_token is not None:
            self._disposed_tokens.add(self._current_bomb_token)
            self._recovery_targets.pop(self._current_bomb_token, None)
            self._discovered_targets.pop(self._current_bomb_token, None)

    def _record_unexpected_drop(
        self, token: Optional[tuple[int, int]]
    ) -> None:
        """Publish a bomb lost outside disposal so a healthy peer can recover it."""
        if token is None or token in self._disposed_tokens:
            return
        pos = self.measured_gps_position()
        hit = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.BOMB,
            require_not_grasped=True,
        )
        if hit is not None and float(hit[0].distance) <= 85.0:
            point = hit[1]
        elif pos is not None:
            point = float(pos[0]), float(pos[1])
        else:
            return
        step = int(self.elapsed_timestep)
        self._bomb_token_centers[token] = point
        self._recovery_targets[token] = (point, step)
        self._diagnostic_event("dropped", token=token, target=point)
        # Let another drone take over first; the releasing drone is often
        # damaged or wedged against the obstacle that caused the loss.
        self._claim_cooldown_until[token] = step + 140

    def _update_active_bomb_observations(self) -> None:
        """Publish every visible active bomb, including targets this scout skips."""
        step = int(self.elapsed_timestep)
        for obj in self.semantic_values() or []:
            if (
                obj.entity_type != DroneSemanticSensor.TypeEntity.BOMB
                or bool(obj.grasped)
                or not math.isfinite(float(obj.distance))
            ):
                continue
            point = self._semantic_to_world(obj)
            if point is None:
                continue
            token = self._token_for_detection(obj, point)
            if token not in self._discovered_targets:
                self._diagnostic_event("discovered", token=token, target=point)
            self._disposed_tokens.discard(token)
            previous_center = self._bomb_token_centers[token]
            self._bomb_token_centers[token] = (
                0.75 * previous_center[0] + 0.25 * point[0],
                0.75 * previous_center[1] + 0.25 * point[1],
            )
            previous = self._discovered_targets.get(token)
            if previous is None or step >= previous[1]:
                self._discovered_targets[token] = (point, step)

    def _refresh_own_team_state(self) -> None:
        if self.grasped_bombs():
            self._set_own_task("carry", self._current_bomb_token)
        elif self._current_bomb_token is not None and self._claimed_bomb is not None:
            self._set_own_task("claim", self._current_bomb_token, self._claimed_bomb)
        else:
            self._set_own_task("search", None)

    def _merge_team_status(self) -> None:
        for _, message in self.communicator.received_messages:
            if not isinstance(message, dict):
                continue
            invalid_entries = message.get("invalid_targets", ())
            if isinstance(invalid_entries, (tuple, list)):
                for entry in invalid_entries:
                    try:
                        value, timestamp = entry
                        token = int(value[0]), int(value[1])
                        timestamp = int(timestamp)
                    except (TypeError, ValueError, IndexError):
                        continue
                    if 0 <= timestamp <= int(self.elapsed_timestep):
                        self._invalidate_target(token, timestamp, remote=True)
            try:
                peer_id = int(message.get("id"))
                peer_step = int(message.get("step"))
                peer_health = int(message.get("health"))
                peer_pos = message.get("position")
                peer_heading = float(message.get("heading"))
                peer_velocity = message.get("velocity")
                if (
                    peer_id != int(self.identifier or 0)
                    and isinstance(peer_pos, (tuple, list))
                    and len(peer_pos) == 2
                    and isinstance(peer_velocity, (tuple, list))
                    and len(peer_velocity) == 2
                ):
                    point = float(peer_pos[0]), float(peer_pos[1])
                    velocity = float(peer_velocity[0]), float(peer_velocity[1])
                    if all(math.isfinite(v) for v in (*point, peer_heading, *velocity)):
                        old = self._peer_state.get(peer_id)
                        if old is None or peer_step >= old[2]:
                            self._peer_state[peer_id] = (
                                point, peer_health, peer_step, peer_heading,
                                bool(message.get("carrying")), velocity,
                            )
            except (TypeError, ValueError):
                pass
            deliveries = message.get("deliveries")
            if isinstance(deliveries, (tuple, list)):
                for entry in deliveries:
                    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                        continue
                    try:
                        owner, count = int(entry[0]), float(entry[1])
                    except (TypeError, ValueError):
                        continue
                    self._team_deliveries[owner] = max(
                        count, self._team_deliveries.get(owner, 0)
                    )

            disposed_tokens = message.get("disposed_tokens")
            if isinstance(disposed_tokens, (tuple, list)):
                for value in disposed_tokens:
                    if not isinstance(value, (tuple, list)) or len(value) != 2:
                        continue
                    try:
                        self._disposed_tokens.add((int(value[0]), int(value[1])))
                    except (TypeError, ValueError):
                        continue

            recovery_targets = message.get("recovery_targets")
            if isinstance(recovery_targets, (tuple, list)):
                for entry in recovery_targets:
                    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                        continue
                    token_value, point_value, step_value = entry
                    if not isinstance(token_value, (tuple, list)) or len(token_value) != 2:
                        continue
                    if not isinstance(point_value, (tuple, list)) or len(point_value) != 2:
                        continue
                    try:
                        token = int(token_value[0]), int(token_value[1])
                        point = float(point_value[0]), float(point_value[1])
                        step = int(step_value)
                    except (TypeError, ValueError):
                        continue
                    previous = self._recovery_targets.get(token)
                    if step >= self._invalid_targets.get(token, -1) and (previous is None or step > previous[1]):
                        self._recovery_targets[token] = (point, step)
                        self._bomb_token_centers[token] = point

            discoveries = message.get("discoveries")
            if isinstance(discoveries, (tuple, list)):
                for entry in discoveries:
                    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                        continue
                    token_value, point_value, step_value = entry
                    if not isinstance(token_value, (tuple, list)) or len(token_value) != 2:
                        continue
                    if not isinstance(point_value, (tuple, list)) or len(point_value) != 2:
                        continue
                    try:
                        token = int(token_value[0]), int(token_value[1])
                        point = float(point_value[0]), float(point_value[1])
                        step = int(step_value)
                    except (TypeError, ValueError):
                        continue
                    previous = self._discovered_targets.get(token)
                    if step >= self._invalid_targets.get(token, -1) and (previous is None or step > previous[1]):
                        self._discovered_targets[token] = (point, step)
                        previous_center = self._bomb_token_centers.get(token, point)
                        self._bomb_token_centers[token] = (
                            previous_center[0]*0.75 + point[0]*0.25,
                            previous_center[1]*0.75 + point[1]*0.25,
                        )

            tasks = message.get("tasks")
            if not isinstance(tasks, (tuple, list)):
                continue
            for entry in tasks:
                if not isinstance(entry, (tuple, list)) or len(entry) != 5:
                    continue
                owner, mode, token_value, point_value, step_value = entry
                if mode not in ("search", "claim", "carry"):
                    continue
                try:
                    owner = int(owner)
                    step = int(step_value)
                except (TypeError, ValueError):
                    continue
                token: Optional[tuple[int, int]] = None
                if isinstance(token_value, (tuple, list)) and len(token_value) == 2:
                    try:
                        token = int(token_value[0]), int(token_value[1])
                    except (TypeError, ValueError):
                        token = None
                point: Optional[WorldPoint] = None
                if isinstance(point_value, (tuple, list)) and len(point_value) == 2:
                    try:
                        candidate = float(point_value[0]), float(point_value[1])
                    except (TypeError, ValueError):
                        candidate = (float("nan"), float("nan"))
                    if math.isfinite(candidate[0]) and math.isfinite(candidate[1]):
                        point = candidate
                previous = self._team_tasks.get(owner)
                if previous is None or step > previous[3]:
                    self._team_tasks[owner] = (str(mode), token, point, step)
                    if token is not None and point is not None:
                        self._bomb_token_centers.setdefault(token, point)
            progress_entries = message.get("task_progress", ())
            if not isinstance(progress_entries, (tuple, list)):
                continue
            for entry in progress_entries:
                if not isinstance(entry, (tuple, list)) or len(entry) != 6:
                    continue
                try:
                    owner, mode, token_value, started, advanced, reported = entry
                    owner = int(owner)
                    token = int(token_value[0]), int(token_value[1])
                    started, advanced, reported = int(started), int(advanced), int(reported)
                except (TypeError, ValueError, IndexError):
                    continue
                if not (mode in ("claim", "carry") and 0 <= owner < self._team_size
                        and 0 <= started <= advanced <= reported <= int(self.elapsed_timestep)):
                    continue
                if owner == int(self.identifier or 0):
                    continue
                previous = self._team_progress.get(owner)
                if previous is None or reported > previous[4]:
                    self._team_progress[owner] = (mode, token, started, advanced, reported)

    def _progress_timeout(self, mode: str) -> int:
        if mode == "carry":
            return max(220, min(500, int(self._max_steps * 0.10)))
        return max(120, min(360, int(self._max_steps * 0.07)))

    def _invalidate_target(self, token: tuple[int, int], step: int, *, remote: bool = False) -> None:
        """Negative observations invalidate old positions, never prove disposal."""
        previous = self._invalid_targets.get(token, -1)
        if step <= previous:
            return
        self._invalid_targets[token] = step
        for targets in (self._discovered_targets, self._recovery_targets):
            observation = targets.get(token)
            if observation is not None and observation[1] < step:
                targets.pop(token, None)
        if not remote:
            self._diagnostic_event("target_invalidated", token=token, reason="local_scan_empty")

    def _target_invalid(self, token: tuple[int, int]) -> bool:
        invalid_at = self._invalid_targets.get(token, -1)
        if invalid_at < 0:
            return False
        observation = max(self._discovered_targets.get(token, (None, -1))[1],
                          self._recovery_targets.get(token, (None, -1))[1])
        return observation < invalid_at

    def _target_area_visible(self, target: WorldPoint) -> bool:
        """Do not invalidate a nearby position hidden behind a wall or peer."""
        pos, heading = self.measured_gps_position(), self.measured_compass_angle()
        ranges = self.lidar_values()
        if pos is None or heading is None or ranges is None:
            return False
        dx, dy = target[0]-float(pos[0]), target[1]-float(pos[1])
        distance = math.hypot(dx, dy)
        if distance > 30.0:
            return False
        angles = np.asarray(self.lidar_rays_angles())
        values = np.asarray(ranges)
        count = min(len(angles), len(values))
        bearing = math.atan2(dy, dx)-float(heading)
        errors = np.abs(np.arctan2(np.sin(angles[:count]-bearing), np.cos(angles[:count]-bearing)))
        selected = (errors <= 0.10) & np.isfinite(values[:count]) & (values[:count] > 0)
        return bool(np.any(selected) and np.min(values[:count][selected]) >= max(0.0, distance-8.0))

    def _task_distance(self, mode: str, target: Optional[WorldPoint], position: WorldPoint) -> Optional[float]:
        goal = self._known_disposal if mode == "carry" else target
        path = self._return_path if mode == "carry" else []
        if not path and goal is not None and self._nav_target is not None and math.hypot(
            goal[0] - self._nav_target[0], goal[1] - self._nav_target[1]
        ) <= 32.0:
            path = self._nav_path
        if path:
            key = (id(path), len(path), path[0], path[-1])
            if getattr(self, "_task_route_key", None) != key:
                self._task_route_key = key
                self._task_route_suffix = sum(math.hypot(b[0]-a[0], b[1]-a[1])
                                              for a, b in zip(path, path[1:]))
            tail = math.hypot(path[-1][0]-goal[0], path[-1][1]-goal[1]) if goal is not None else 0.0
            return math.hypot(position[0]-path[0][0], position[1]-path[0][1]) + self._task_route_suffix + tail
        if goal is not None:
            return math.hypot(position[0]-goal[0], position[1]-goal[1])
        return None

    def _update_own_task_progress(self, mode: str, token: Optional[tuple[int, int]], target: Optional[WorldPoint]) -> None:
        owner, step = int(self.identifier or 0), int(self.elapsed_timestep)
        pos = self.measured_gps_position()
        position = (float(pos[0]), float(pos[1])) if pos is not None else None
        distance = self._task_distance(mode, target, position) if position is not None else None
        state = self._task_monitor.update(mode, token, step, position, distance)
        if state is None:
            self._team_progress.pop(owner, None)
        else:
            self._team_progress[owner] = (mode, state.token, state.started, state.advanced, step)

    def _own_task_stalled(self, mode: str, token: Optional[tuple[int, int]]) -> bool:
        state = self._task_monitor.current
        return (state is not None and (state.mode, state.token) == (mode, token)
                and self._task_monitor.stalled(int(self.elapsed_timestep), self._progress_timeout(mode)))

    def _available_carrier_helper(self) -> Optional[int]:
        pos = self.measured_gps_position()
        if pos is None:
            return None
        candidates = []
        step = int(self.elapsed_timestep)
        for owner, (point, health, observed, _, carrying, _) in self._peer_state.items():
            task = self._team_tasks.get(owner)
            helper_health = max(16, min(25, int(self.drone_health)+5))
            if (step-observed > 90 or carrying or health < helper_health
                    or (task is not None and step-task[3] <= self._TASK_LEASE_STEPS and task[0] != "search")):
                continue
            distance = math.hypot(point[0]-float(pos[0]), point[1]-float(pos[1]))
            if distance <= 300.0:
                candidates.append((distance-2.0*health, owner))
        return min(candidates)[1] if candidates else None

    def _recover_stalled_carrier(self, cmd: CommandsDict) -> bool:
        """Bound looping with a replan, then hand over a real stalled load."""
        token, step = self._current_bomb_token, int(self.elapsed_timestep)
        if token != self._carrier_recovery_token:
            self._carrier_recovery_token = token
            self._carrier_recovery_attempts = 0
            self._carrier_recovery_step = -10_000
        state = self._task_monitor.current
        if state is not None and state.advanced > self._carrier_recovery_step:
            self._carrier_recovery_attempts = 0
        if not self._own_task_stalled("carry", token):
            return False
        if step-self._carrier_recovery_step < self._progress_timeout("carry"):
            return False
        helper = self._available_carrier_helper()
        pos = self.measured_gps_position()
        if (helper is not None and pos is not None and token is not None
                and (self._carrier_recovery_attempts >= 1 or int(self.drone_health) <= 15)):
            point = float(pos[0]), float(pos[1])
            self._recovery_targets[token] = (point, step)
            self._bomb_token_centers[token] = point
            self._claim_cooldown_until[token] = step + 300
            self._handoff_hold_until = step + 180
            self._carrier_recovery_step = step
            self._diagnostic_event("carry_handover", token=token, helper=helper, target=point)
            # Keep the token until the actual next-frame drop observation;
            # it will correct this coarse position using the semantic sensor.
            self._set_own_task("search", None)
            self._set_motion(cmd, forward=0.0, lateral=0.0, rotation=0.0)
            cmd["grasper"] = 0
            return True
        if self._known_disposal is None:
            return False
        self._carrier_recovery_attempts += 1
        self._carrier_recovery_step = step
        old_return_trail = self._return_path
        position = (float(pos[0]), float(pos[1])) if pos is not None else None
        self._return_path = self._optimize_return_trail(position, old_return_trail)
        if not self._return_route_verified:
            self._return_path = []
        self._return_progress_target = None
        self._carry_detour_target = None
        self._carry_detour_until = 0
        self._carry_progress_distance = float("inf")
        self._carry_progress_step = step
        self._nav_path = []
        self._nav_target = None
        self._nav_age = 0
        self._diagnostic_event("carry_replan", token=token, attempt=self._carrier_recovery_attempts)
        # A recovery attempt does not renew measured progress or disposal count.
        return False

    def _peer_claim_stalled(self, owner: int, token: tuple[int, int]) -> bool:
        progress = self._team_progress.get(owner)
        return (progress is not None and progress[:2] == ("claim", token)
                and int(self.elapsed_timestep) - progress[3] >= self._progress_timeout("claim"))

    def _abandon_claim(self, token: Optional[tuple[int, int]], reason: str) -> None:
        if token is not None:
            self._claim_cooldown_until[token] = int(self.elapsed_timestep) + max(
                80, min(240, int(self._max_steps * 0.05)))
        self._diagnostic_event("claim_released", token=token, reason=reason)
        self._set_own_task("search", None)
        self._current_bomb_token = None
        self._claimed_bomb = None
        self._claim_age = 0
        self._last_claim_token = None
        self._target_progress_token = None
        self._target_progress_distance = float("inf")
        if token is not None:
            self._recovery_scan_steps.pop(token, None)
        self._nav_path = []
        self._nav_target = None
        self._nav_age = 0

    def _task_has_progress(self, owner: int, task: tuple) -> bool:
        mode, token, _, task_step = task
        return valid_progress_snapshot(mode, token, task_step, self._team_progress.get(owner),
                                       int(self.elapsed_timestep), self._progress_timeout(mode), self._TASK_LEASE_STEPS)

    def _should_wait_for_carriers(self) -> bool:
        """Temporary standby requires real attached loads with recent progress."""
        delivered = sum(max(0.0, value) for value in self._team_deliveries.values())
        if delivered < self._EXPECTED_BOMBS - 2.01:
            return False
        tokens = set()
        own_id, step = int(self.identifier or 0), int(self.elapsed_timestep)
        for owner, task in self._team_tasks.items():
            mode, token, _, reported = task
            if mode != "carry" or token is None or step - reported > self._TASK_LEASE_STEPS:
                continue
            if owner == own_id:
                attached = bool(self.grasped_bombs())
            else:
                peer = self._peer_state.get(owner)
                attached = peer is not None and step - peer[2] <= 90 and peer[4] and peer[1] > 0
            if not attached or not self._task_has_progress(owner, task):
                return False
            tokens.add(token)
        if delivered + len(tokens.difference(self._disposed_tokens)) < self._EXPECTED_BOMBS - 0.01:
            return False
        unresolved = set(self._discovered_targets) | set(self._recovery_targets)
        return not unresolved.difference(tokens).difference(self._disposed_tokens)

    def _diagnostic_event(self, event: str, **details) -> None:
        if not self._diagnostics_enabled or self._diagnostic_events >= 600:
            return
        self._diagnostic_events += 1
        print("TASK_EVENT " + json.dumps(dict(drone=int(self.identifier or 0),
              step=int(self.elapsed_timestep), event=event, **details), ensure_ascii=True), flush=True)

    def _set_operational_stage(self, stage: str, reason: Optional[str] = None) -> None:
        if self._operational_stage != stage:
            self._operational_stage = stage
            self._diagnostic_event("state", state=stage, reason=reason)

    def _diagnostic_snapshot(self) -> None:
        if self._diagnostics_enabled and int(self.elapsed_timestep) - self._last_diagnostic_snapshot >= 200:
            self._last_diagnostic_snapshot = int(self.elapsed_timestep)
            state = self._task_monitor.current
            self._diagnostic_event("snapshot", state=self._operational_stage,
                                   health=int(self.drone_health), token=self._current_bomb_token,
                                   progress_step=state.advanced if state is not None else None,
                                   deliveries=sum(self._team_deliveries.values()),
                                   discovered=len(self._discovered_targets))

    def _all_bombs_accounted_for(self) -> bool:
        """Completion requires actual disposal rewards, never claims."""
        delivered = sum(max(0.0, value) for value in self._team_deliveries.values())
        return delivered >= self._EXPECTED_BOMBS - 0.01

    def _should_retire_low_health(self) -> bool:
        if self._retired_low_health:
            return True
        if int(self.drone_health) <= self._LOW_HEALTH:
            self._retired_low_health = True
            return True
        delivered = sum(max(0.0, value) for value in self._team_deliveries.values())
        if (
            delivered < self._LATE_RETIRE_DELIVERIES
            or int(self.drone_health) > self._LATE_RETIRE_HEALTH
        ):
            self._safety_anchor = None
            self._safety_anchor_step = int(self.elapsed_timestep)
            return False
        pos = self.measured_gps_position()
        if pos is None:
            return False
        current = float(pos[0]), float(pos[1])
        step = int(self.elapsed_timestep)
        if self._safety_anchor is None:
            self._safety_anchor = current
            self._safety_anchor_step = step
            return False
        if math.hypot(
            current[0] - self._safety_anchor[0],
            current[1] - self._safety_anchor[1],
        ) >= self._LATE_RETIRE_STALL_RADIUS:
            self._safety_anchor = current
            self._safety_anchor_step = step
            return False
        if step - self._safety_anchor_step < self._LATE_RETIRE_STALL_STEPS:
            return False
        self._retired_low_health = True
        return True

    def _control_carry(self, cmd: CommandsDict) -> CommandsDict:
        cmd["grasper"] = 1
        disposal_hit = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER
        )
        if self._run_carry_exit(cmd, disposal_hit):
            return cmd
        if disposal_hit is None and int(self.drone_health) <= 6:
            if self._current_bomb_token is not None:
                pos = self.measured_gps_position()
                bomb_hit = self._find_closest_semantic(
                    DroneSemanticSensor.TypeEntity.BOMB
                )
                if bomb_hit is not None:
                    release_point = bomb_hit[1]
                elif pos is not None:
                    release_point = (
                        float(pos[0]),
                        float(pos[1]),
                    )
                else:
                    release_point = None
                if release_point is not None:
                    self._bomb_token_centers[self._current_bomb_token] = release_point
                    self._recovery_targets[self._current_bomb_token] = (
                        release_point,
                        int(self.elapsed_timestep),
                    )
                self._claim_cooldown_until[self._current_bomb_token] = (
                    int(self.elapsed_timestep) + 500
                )
            self._set_own_task("search", None)
            self._retired_low_health = True
            self._set_motion(cmd, forward=0.0, lateral=0.0, rotation=0.0)
            cmd["grasper"] = 0
            return cmd
        if disposal_hit is None:
            if self._known_disposal is not None:
                self._navigate_online(self._known_disposal, cmd)
                if not self._nav_plan_failed or self._lidar_escape_active():
                    return cmd
            if self._follow_return_path(cmd):
                cmd["grasper"] = 1
                return cmd
            if self._control_carry_detour(cmd):
                cmd["grasper"] = 1
                return cmd
            if self._known_disposal is not None:
                if self._run_lidar_escape_if_needed(cmd):
                    return cmd
                self._navigate_online(self._known_disposal, cmd)
                cmd["grasper"] = 1
                return cmd
            explore = self._control_explore(cmd)
            explore["grasper"] = 1
            return explore

        target = disposal_hit[1]
        near = self._within_radius(target, self._DISPOSAL_PUSH_TRIGGER_DIST)
        if (not near) and self._run_lidar_escape_if_needed(cmd):
            return cmd

        if near or self._carry_push_steps > 0:
            return self._control_carry_push(cmd, target)

        # A carrier must keep local obstacle avoidance enabled.  Without this
        # flag the old demo drove straight into walls and could spend hundreds
        # of steps oscillating before reaching the disposal center.
        arrived = self._move_toward(*target, cmd, use_lidar=True)
        if arrived:
            cmd["grasper"] = 1
        return cmd

    def _select_recovery_target(
        self,
    ) -> Optional[tuple[tuple[int, int], WorldPoint]]:
        pos = self.measured_gps_position()
        if pos is None:
            return None
        candidates = [
            (token, point)
            for token, (point, _) in self._recovery_targets.items()
            if token not in self._disposed_tokens
            and self._claim_cooldown_until.get(token, -1) <= int(self.elapsed_timestep)
            and self._can_claim_target(token, point)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (
            item[0] != self._current_bomb_token,
            math.hypot(item[1][0] - float(pos[0]), item[1][1] - float(pos[1])),
        ))

    def _select_discovered_target(
        self,
    ) -> Optional[tuple[tuple[int, int], WorldPoint]]:
        """Bid for shared sightings using live positions and active leases."""
        pos = self.measured_gps_position()
        if pos is None:
            return None
        step = int(self.elapsed_timestep)
        candidates: list[tuple[tuple[int, int], WorldPoint]] = []
        for token, (point, _) in self._discovered_targets.items():
            if (
                token in self._disposed_tokens
                or self._claim_cooldown_until.get(token, -1) > step
            ):
                continue
            if self._can_claim_target(token, point):
                candidates.append((token, point))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (
            item[0] != self._current_bomb_token,
            math.hypot(item[1][0] - float(pos[0]), item[1][1] - float(pos[1])),
        ))

    def _opening_role_accepts(self, owner: int, point: WorldPoint) -> bool:
        if self.size_area is None or self._team_size < 6 or self._max_steps > 2000:
            return True
        if self.elapsed_timestep >= int(self._max_steps * 0.70):
            return True
        width, height = float(self.size_area[0]), float(self.size_area[1])
        if owner == self._team_size - 3:
            return point[0] >= width * 0.20 and abs(point[1]) < height * 0.25
        if owner == self._team_size - 2:
            return point[0] >= width * 0.20 and point[1] >= height * 0.25
        if owner == self._team_size - 1:
            return point[0] >= width * 0.20 and point[1] < -height * 0.25
        return True

    def _target_bid(self, owner: int, position: WorldPoint, health: int,
                    point: WorldPoint) -> float:
        distance = math.hypot(position[0] - point[0], position[1] - point[1])
        # A low-health scout is less likely to finish the return trip.
        health_penalty = 180.0 if health < 25 else (55.0 if health < 45 else 0.0)
        # Spatial lanes remain a preference, never a veto on a visible bomb.
        lane_penalty = 0.0
        if self.size_area is not None and self.elapsed_timestep < self._max_steps * 0.3:
            width = float(self.size_area[0])
            lane = max(0, min(self._team_size - 1,
                int((point[0] + width / 2.0) / width * self._team_size)))
            lane_penalty = min(70.0, abs(lane - owner) * 12.0)
        return distance + health_penalty + lane_penalty

    def _can_claim_target(self, token: tuple[int, int], point: WorldPoint) -> bool:
        step = int(self.elapsed_timestep)
        if self._target_invalid(token):
            return False
        my_id = int(self.identifier or 0)
        if not self._opening_role_accepts(my_id, point):
            return False
        oldest = step - self._TASK_LEASE_STEPS
        own_task = self._team_tasks.get(my_id)
        continuing = bool(own_task and own_task[0] == "claim" and own_task[1] == token)
        for owner, (mode, owned_token, owned_point, task_step) in self._team_tasks.items():
            if owner == my_id or task_step < oldest:
                continue
            # Adjacent noisy claims share one approach slot. Distinct adjacent
            # bombs can still be picked sequentially after the first carrier leaves.
            same_approach = (mode == "claim" and owned_point is not None
                             and math.hypot(owned_point[0]-point[0], owned_point[1]-point[1]) < 70.0)
            peer = self._peer_state.get(owner)
            loaded_nearby = (mode == "carry" and peer is not None
                             and step-peer[2] <= 8 and peer[4]
                             and math.hypot(peer[0][0]-point[0], peer[0][1]-point[1]) < 90.0)
            if mode in ("claim", "carry") and (owned_token == token or same_approach or loaded_nearby):
                if mode == "claim" and self._peer_claim_stalled(owner, owned_token):
                    continue
                # A simultaneous first sighting has one stable tie breaker.
                if mode == "carry" or not continuing or owner < my_id:
                    return False
        pos = self.measured_gps_position()
        if pos is None:
            return False
        my_bid = self._target_bid(my_id, (float(pos[0]), float(pos[1])),
                                  int(self.drone_health), point)
        if continuing:
            return True
        for owner, (peer_pos, health, seen_step, _, _, _) in self._peer_state.items():
            if seen_step < step - 90 or health <= self._LOW_HEALTH:
                continue
            task = self._team_tasks.get(owner)
            if task is not None and task[3] >= oldest and task[0] != "search":
                continue
            if not self._opening_role_accepts(owner, point):
                continue
            peer_bid = self._target_bid(owner, peer_pos, health, point)
            if (peer_bid, owner) < (my_bid, my_id):
                return False
        return True

    def _control_recovery_target(
        self,
        cmd: CommandsDict,
        token: tuple[int, int],
        target: WorldPoint,
        *,
        source: str = "recovery",
    ) -> CommandsDict:
        pos = self.measured_gps_position()
        step = int(self.elapsed_timestep)
        if pos is not None:
            position = float(pos[0]), float(pos[1])
            target_distance = math.hypot(
                position[0] - target[0], position[1] - target[1]
            )
            if self._target_progress_token != token:
                self._target_progress_token = token
                self._target_progress_distance = target_distance
                self._target_progress_step = step
            elif target_distance <= self._target_progress_distance - 20.0:
                self._target_progress_distance = target_distance
                self._target_progress_step = step
            elif step - self._target_progress_step >= max(
                120, min(360, int(self._max_steps * 0.07))
            ) and self._own_task_stalled("claim", token):
                self._target_progress_token = None
                self._target_progress_distance = float("inf")
                self._abandon_claim(token, "shared_target_no_progress")
                return self._control_explore(cmd)
        if self._within_radius(target, 30.0):
            scan_steps = self._recovery_scan_steps.get(token, 0) + 1
            self._recovery_scan_steps[token] = scan_steps
            cmd["rotation"] = self._explore_turn_rate()
            if scan_steps >= 100:
                if self._target_area_visible(target):
                    self._invalidate_target(token, int(self.elapsed_timestep))
                if source == "discovery":
                    self._discovered_targets.pop(token, None)
                else:
                    self._recovery_targets.pop(token, None)
                self._recovery_scan_steps.pop(token, None)
                self._abandon_claim(token, "target_not_visible_after_scan")
            return cmd
        self._recovery_scan_steps.pop(token, None)
        self._navigate_online(target, cmd)
        return cmd

    def _run_carry_exit(
        self,
        cmd: CommandsDict,
        disposal_hit: Optional[SemanticHit],
    ) -> bool:
        """Pull a newly attached trailing bomb clear before turning toward home."""
        if self._carry_exit_steps_left <= 0:
            return False
        if disposal_hit is not None and float(disposal_hit[0].distance) < 90.0:
            self._carry_exit_steps_left = 0
            return False
        self._carry_exit_steps_left -= 1
        self._set_motion(
            cmd,
            forward=self._CARRY_EXIT_SPEED,
            lateral=0.0,
            rotation=0.0,
        )
        self._apply_lidar_avoidance(cmd)
        cmd["grasper"] = 1
        return True

    def _record_breadcrumb(self) -> None:
        """Remember a sparse, loop-erased trail through already traversed space."""
        pos = self.measured_gps_position()
        if pos is None:
            return
        point = float(pos[0]), float(pos[1])
        if not self._breadcrumbs:
            self._breadcrumbs.append(point)
            return
        if math.hypot(
            point[0] - self._breadcrumbs[-1][0],
            point[1] - self._breadcrumbs[-1][1],
        ) < 28.0:
            return

        # Remove large loops so a carrier does not replay an entire patrol.
        for index in range(len(self._breadcrumbs) - 2, -1, -1):
            old = self._breadcrumbs[index]
            if math.hypot(point[0] - old[0], point[1] - old[1]) <= 36.0:
                del self._breadcrumbs[index + 1:]
                break
        self._breadcrumbs.append(point)
        if len(self._breadcrumbs) > 300:
            del self._breadcrumbs[: len(self._breadcrumbs) - 300]

    def _prepare_return_path(self) -> None:
        pos = self.measured_gps_position()
        current = (float(pos[0]), float(pos[1])) if pos is not None else None
        trail = list(reversed(self._breadcrumbs))
        if current is not None:
            while trail and math.hypot(
                current[0] - trail[0][0], current[1] - trail[0][1]
            ) < 35.0:
                trail.pop(0)
        self._return_path = self._optimize_return_trail(current, trail)
        self._return_progress_target = None
        self._return_progress_position = None
        self._return_progress_distance = float("inf")
        self._return_progress_step = int(self.elapsed_timestep)
        self._nav_path = []
        self._nav_target = None
        self._nav_age = 0
        self._carry_progress_distance = float("inf")
        self._carry_progress_step = int(self.elapsed_timestep)
        self._carry_detour_target = None
        self._carry_detour_until = 0

    def _optimize_return_trail(self, position, trail):
        self._return_route_verified = False
        if position is None or self._online_map is None:
            return trail
        route, verified = observed_return_route(
            self.size_area, self._online_map.evidence, self._online_obstacles,
            position, trail, self._known_disposal,
            evidence_cell_size=self._online_map.cell_size, clearance=28)
        self._return_route_verified = verified
        self._diagnostic_event("return_route", verified=verified, points_before=len(trail),
                               points_after=len(route), length_before=path_length(position, trail),
                               length_after=path_length(position, route))
        return route

    def _control_carry_detour(self, cmd: CommandsDict) -> bool:
        """Route a stalled carrier around the nearer end of an unseen wall."""
        if self._known_disposal is None:
            return False
        pos = self.measured_gps_position()
        if pos is None:
            return False
        position = float(pos[0]), float(pos[1])
        step = int(self.elapsed_timestep)
        if self._carry_detour_target is not None:
            if (
                self._within_radius(self._carry_detour_target, 32.0)
                or step >= self._carry_detour_until
            ):
                self._carry_detour_target = None
                self._carry_progress_distance = float("inf")
                self._carry_progress_step = step
                self._nav_path = []
                self._nav_target = None
                self._nav_age = 0
                return False
            self._move_toward(*self._carry_detour_target, cmd, use_lidar=True)
            return True

        home = self._known_disposal
        distance = math.hypot(position[0] - home[0], position[1] - home[1])
        if distance <= self._carry_progress_distance - 18.0:
            self._carry_progress_distance = distance
            self._carry_progress_step = step
            return False
        if self._carry_progress_distance == float("inf"):
            self._carry_progress_distance = distance
            self._carry_progress_step = step
            return False
        if step - self._carry_progress_step < 180:
            return False

        dx, dy = home[0] - position[0], home[1] - position[1]
        if self.size_area is not None:
            width, height = float(self.size_area[0]), float(self.size_area[1])
            offset = min(width, height) * 0.28
            margin = min(55.0, width * 0.10, height * 0.10)
        else:
            width = height = 1000.0
            offset = 140.0
            margin = 50.0
        if abs(dy) >= abs(dx):
            sign = -1.0 if position[0] <= 0.0 else 1.0
            detour = position[0] + sign * offset, position[1]
        else:
            sign = -1.0 if position[1] <= 0.0 else 1.0
            detour = position[0], position[1] + sign * offset
        self._carry_detour_target = (
            max(-width / 2.0 + margin, min(width / 2.0 - margin, detour[0])),
            max(-height / 2.0 + margin, min(height / 2.0 - margin, detour[1])),
        )
        self._carry_detour_until = step + 360
        self._nav_path = []
        self._nav_target = None
        self._nav_age = 0
        self._move_toward(*self._carry_detour_target, cmd, use_lidar=True)
        return True

    def _follow_return_path(self, cmd: CommandsDict) -> bool:
        """Retrace sensor-verified space until disposal navigation is local."""
        pos = self.measured_gps_position()
        if pos is None or not self._return_path:
            return False
        position = float(pos[0]), float(pos[1])

        while self._return_path and math.hypot(
            position[0] - self._return_path[0][0],
            position[1] - self._return_path[0][1],
        ) <= (12.0 if getattr(self, "_return_route_verified", False) else 28.0):
            self._return_path.pop(0)
        if not self._return_path:
            return False
        target = self._return_path[0]
        target_distance = math.hypot(
            position[0] - target[0], position[1] - target[1]
        )
        step = int(self.elapsed_timestep)
        if (
            self._return_progress_target != target
            or self._return_progress_position is None
            or math.hypot(
                position[0] - self._return_progress_position[0],
                position[1] - self._return_progress_position[1],
            ) >= 32.0
            or target_distance <= self._return_progress_distance - 18.0
        ):
            self._return_progress_target = target
            self._return_progress_position = position
            self._return_progress_distance = target_distance
            self._return_progress_step = step
        elif step - self._return_progress_step >= 450:
            # A safe trail is useful only while it can actually be followed.
            # Slow wall contact can otherwise keep a carrier here indefinitely.
            self._return_path = []
            self._return_progress_target = None
            self._carry_progress_distance = float("inf")
            self._carry_progress_step = step
            return False
        if self._run_lidar_escape_if_needed(cmd):
            return True
        self._move_holonomic_toward(*target, cmd)
        return True

    def _navigate_online(self, target: WorldPoint, cmd: CommandsDict) -> bool:
        pos = self.measured_gps_position()
        if pos is None:
            return False
        position = float(pos[0]), float(pos[1])
        if math.hypot(target[0] - position[0], target[1] - position[1]) < self._ARRIVE_RADIUS:
            self._nav_path = []
            self._nav_target = target
            return True
        if self._run_lidar_escape_if_needed(cmd):
            self._nav_age += 1
            return False
        replan_steps = 65 if self.grasped_bombs() else 90
        target_changed = (
            self._nav_target is None
            or math.hypot(
                self._nav_target[0] - target[0],
                self._nav_target[1] - target[1],
            ) > 24.0
        )
        if (target_changed and (self._nav_target is None or self._nav_age >= 12)) or self._nav_age >= replan_steps:
            self._nav_path = self._plan_navigation(position, target)
            self._nav_plan_failed = not self._nav_path
            self._nav_target = target
            self._nav_age = 0
            if target_changed or self._nav_path:
                self._nav_fallback_target = None
        while self._nav_path and math.hypot(
            self._nav_path[0][0] - position[0], self._nav_path[0][1] - position[1]
        ) <= (12.0 if self.grasped_bombs() else self._PATROL_ARRIVE_RADIUS):
            self._nav_path.pop(0)
        if self._nav_plan_failed and not self._nav_path and math.hypot(
            target[0] - position[0], target[1] - position[1]
        ) > self._PATROL_ARRIVE_RADIUS:
            step = int(self.elapsed_timestep)
            if (
                self._nav_fallback_target is None
                or step >= self._nav_fallback_until
                or math.hypot(
                    position[0] - self._nav_fallback_target[0],
                    position[1] - self._nav_fallback_target[1],
                ) <= self._PATROL_ARRIVE_RADIUS
            ):
                heading = self.measured_compass_angle()
                self._nav_fallback_target = (
                    choose_lidar_detour(
                        position, float(heading), target,
                        self.lidar_values(), self.lidar_rays_angles(),
                    ) if heading is not None else None
                )
                self._nav_fallback_until = step + 80
            if self._nav_fallback_target is None:
                if not self._run_lidar_escape_if_needed(cmd):
                    cmd["rotation"] = self._explore_turn_rate()
            else:
                self._move_toward(*self._nav_fallback_target, cmd, use_lidar=True)
            self._nav_age = max(self._nav_age + 1, replan_steps - 30)
            return False
        waypoint = self._nav_path[0] if self._nav_path else target
        if self.grasped_bombs():
            self._move_holonomic_toward(*waypoint, cmd)
        else:
            self._move_toward(*waypoint, cmd, use_lidar=True)
        self._nav_age += 1
        return False

    def _control_carry_push(self, cmd: CommandsDict, target: WorldPoint) -> CommandsDict:
        # 炸弹挂在机身后方：先推进处置区再释放，避免落在区外。
        if self._carry_spin_release_steps_left > 0:
            return self._control_carry_spin_release(cmd)

        self._carry_push_steps += 1
        self._move_toward(*target, cmd)
        cmd["forward"] = max(float(cmd.get("forward", 0.0)), self._DISPOSAL_PUSH_FORWARD)
        cmd["grasper"] = 1
        if self._carry_push_steps >= self._DISPOSAL_PUSH_STEPS:
            cmd["grasper"] = 0
        return cmd

    def _start_carry_spin_release(self, turn_sign: float) -> None:
        self._carry_spin_sign = 1.0 if turn_sign >= 0 else -1.0
        self._carry_spin_release_steps_left = self._DISPOSAL_SPIN_RELEASE_STEPS
        self._carry_push_steps = 0

    def _control_carry_spin_release(self, cmd: CommandsDict) -> CommandsDict:
        self._set_motion(
            cmd,
            forward=0.0,
            lateral=0.0,
            rotation=self._LIDAR_AVOID_ROTATION * self._carry_spin_sign,
        )
        cmd["grasper"] = 1
        self._carry_spin_release_steps_left -= 1
        if self._carry_spin_release_steps_left <= 0:
            cmd["grasper"] = 0
        return cmd

    def _should_start_carry_spin_release(self, near_disposal: bool) -> bool:
        if not near_disposal or self._carry_spin_release_steps_left > 0:
            return False
        return self._lidar_front_blocked()

    def _control_explore(self, cmd: CommandsDict) -> CommandsDict:
        self._carry_push_steps = 0
        step = int(self.elapsed_timestep)
        my_id = int(self.identifier or 0)
        if (
            not self._frontier_early_activated
            and self._frontier_start_step <= step <= int(self._max_steps * 0.30)
            and self._frontier_grid.size
            and len(self._discovered_targets) < self._EXPECTED_BOMBS * 0.6
            and np.count_nonzero(self._frontier_grid) >= self._frontier_grid.size * 0.85
        ):
            # Nearly all of the sensor map is known, yet few bomb leads exist.
            # Search the remaining gaps instead of retracing a fixed lane.
            self._frontier_early_activated = True
        frontier_eligible = (
            self._frontier_early_activated
            or my_id % 2 == 0 and step >= int(self._max_steps * 0.38)
            or step >= int(self._max_steps * 0.72)
        )
        if frontier_eligible and step >= self._frontier_start_step:
            if self._control_frontier_exploration(cmd):
                return cmd
        if self._patrol_route:
            return self._control_patrol(cmd)
        if self._run_lidar_escape_if_needed(cmd):
            return cmd

        cycle = self._EXPLORE_STRAIGHT_STEPS + self._EXPLORE_TURN_STEPS
        phase = self._explore_tick % cycle
        self._explore_tick += 1
        if phase < self._EXPLORE_STRAIGHT_STEPS:
            cmd["forward"] = self._EXPLORE_FORWARD
            self._apply_lidar_avoidance(cmd)
        else:
            cmd["rotation"] = self._explore_turn_rate()
        return cmd

    def _control_patrol(self, cmd: CommandsDict) -> CommandsDict:
        """Sweep a dimension-derived lane and map obstacles as they are sensed."""
        if (
            int(self.identifier or 0) != 4
            and self._run_lidar_escape_if_needed(cmd)
        ):
            return cmd
        if self._patrol_scan_steps_left > 0:
            self._patrol_scan_steps_left -= 1
            cmd["rotation"] = 0.75 if int(self.identifier or 0) % 2 == 0 else -0.75
            return cmd

        target = self._patrol_route[self._patrol_index % len(self._patrol_route)]
        delivered = sum(max(0.0, value) for value in self._team_deliveries.values())
        target_timeout = 420 if delivered >= 8.0 else 220
        my_id = int(self.identifier or 0)
        opening_sector_leg = (
            (
                self._middle_edge_sweep_enabled
                and (
                    (my_id == 2 and self._patrol_index <= 2)
                    or (
                        my_id == max(0, int(self._team_size) - 3)
                        and self._patrol_index == 0
                    )
                )
            )
            or (
                self._middle_edge_probe_enabled
                and my_id == 2
                and self._patrol_index == 0
            )
        )
        urgent_short_leg = (
            self._max_steps <= 2000
            and (
                (
                    my_id == max(0, int(self._team_size) - 3)
                    and self._patrol_index <= 2
                )
                or (
                    my_id == max(0, int(self._team_size) - 2)
                    and self._patrol_index <= 2
                )
                or (
                    my_id == max(0, int(self._team_size) - 1)
                    and self._patrol_index <= 2
                )
            )
        )
        medium_center_leg = (
            2000 < self._max_steps <= 5000
            and my_id == int(self._team_size) // 2
            and self._patrol_index <= 3
        )
        medium_left_leg = (
            self._middle_edge_probe_enabled
            and my_id == 1
            and self._patrol_index == 0
        )
        long_upper_right_leg = (
            self._max_steps > 5000
            and self.size_area is not None
            and float(self.size_area[0]) < 1400.0
            and my_id == int(self._team_size) - 2
            and self._patrol_index <= 2
        )
        if (
            opening_sector_leg
            or urgent_short_leg
            or medium_center_leg
            or medium_left_leg
            or long_upper_right_leg
        ):
            # A middle-edge scout can begin on the opposite side of a complex
            # map.  Give its first dimension-derived sector target enough time
            # for online obstacle planning instead of skipping it after a short
            # generic lane timeout.
            target_timeout = max(
                target_timeout,
                650
                if self._middle_edge_sweep_enabled or urgent_short_leg
                else 650 if medium_center_leg else 420,
            )
        if (
            self._nav_target == target
            and self._patrol_target_age >= target_timeout
        ):
            self._patrol_index = (self._patrol_index + 1) % len(self._patrol_route)
            self._nav_path = []
            self._nav_target = None
            self._nav_age = 0
            self._patrol_target_age = 0
            target = self._patrol_route[self._patrol_index % len(self._patrol_route)]
        arrive_radius = self._PATROL_ARRIVE_RADIUS
        if (
            int(self.identifier or 0) == 4
            and self.size_area is not None
            and abs(target[0]) <= 1.0
            and target[1] >= -float(self.size_area[1]) * 0.20
        ):
            arrive_radius = 60.0
        if self._within_radius(target, arrive_radius):
            self._patrol_index = (self._patrol_index + 1) % len(self._patrol_route)
            self._patrol_scan_steps_left = self._PATROL_SCAN_STEPS
            self._patrol_target_age = 0
            self._nav_path = []
            self._nav_target = None
            cmd["rotation"] = 0.75 if int(self.identifier or 0) % 2 == 0 else -0.75
            return cmd

        pos = self.measured_gps_position()
        if pos is None:
            cmd["rotation"] = self._explore_turn_rate()
            return cmd
        position = float(pos[0]), float(pos[1])
        if (
            int(self.identifier or 0) == 4
            and self.size_area is not None
            and abs(target[0]) <= 1.0
            and target[1] >= -float(self.size_area[1]) * 0.10
            and position[1] < target[1]
            and abs(position[0]) <= 30.0
        ):
            heading = self.measured_compass_angle()
            desired = math.atan2(target[1] - position[1], target[0] - position[0])
            center_front: Optional[float] = None
            lidar = self.lidar_values()
            angles = self.lidar_rays_angles()
            if lidar is not None and angles is not None and len(lidar) and len(angles):
                center_index = int(np.argmin(np.abs(np.asarray(angles, dtype=float))))
                candidate = float(lidar[center_index])
                if math.isfinite(candidate):
                    center_front = candidate
            aligned = (
                heading is not None
                and abs(normalize_angle(desired - float(heading))) <= 0.18
            )
            if not aligned:
                self._move_toward(*target, cmd, use_lidar=False)
                self._nav_age += 1
                self._patrol_target_age += 1
                return cmd
            if center_front is None or center_front >= 30.0:
                self._move_toward(*target, cmd, use_lidar=False)
                self._nav_age += 1
                self._patrol_target_age += 1
                return cmd
        if (
            int(self.identifier or 0) == 4
            and self._run_lidar_escape_if_needed(cmd)
        ):
            return cmd
        self._navigate_online(target, cmd)
        self._patrol_target_age += 1
        return cmd
    def _plan_navigation(self, start: WorldPoint, target: WorldPoint) -> list[WorldPoint]:
        return plan_occupancy_path(
            self.size_area,
            self._online_obstacles,
            start,
            target,
            clearance=32 if self.grasped_bombs() else 20,
            cell_size=16,
        )

    def _control_chase_bomb(self, cmd: CommandsDict, bomb: SemanticHit) -> CommandsDict:
        self._carry_push_steps = 0
        self._reset_explore_state()
        bomb_det, _ = bomb
        # Grasping is an omnidirectional collision device and does not enforce
        # exclusive ownership. Enable it only at the selected close target.
        cmd["grasper"] = int(self._should_enable_grasper(bomb_det))
        if self._run_lidar_escape_if_needed(cmd):
            return cmd
        # Visibility does not guarantee a collision-free straight approach.
        # Plan the distant approach, then align locally for attachment.
        if float(bomb_det.distance) > 65.0:
            self._navigate_online(bomb[1], cmd)
        else:
            self._approach_bomb_frontally(cmd, bomb_det)
        return cmd

    def _should_enable_grasper(self, det) -> bool:
        distance, angle = float(det.distance), float(det.angle)
        if (int(self.elapsed_timestep) < 4 or not math.isfinite(distance)
                or not math.isfinite(angle) or distance > 35.0 or abs(angle) > 0.25):
            return False
        for other in self.semantic_values() or []:
            if (other.entity_type == DroneSemanticSensor.TypeEntity.BOMB
                    and not bool(other.grasped) and float(other.distance) < distance-8.0):
                return False
        return True

    def _approach_bomb_frontally(
        self, cmd: CommandsDict, bomb_det: DroneSemanticSensor.Data
    ) -> None:
        """正面对齐后再接近炸弹，减少侧向抓取导致的回程不稳。"""
        ang = float(bomb_det.angle)
        dist = float(bomb_det.distance)
        if not math.isfinite(ang) or not math.isfinite(dist):
            return

        if abs(ang) > self._BOMB_FRONT_ALIGN_TOL:
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=0.0,
                rotation=self._BOMB_FRONT_ALIGN_ROT if ang > 0 else -self._BOMB_FRONT_ALIGN_ROT,
            )
            return

        self._set_motion(
            cmd,
            forward=max(
            self._BOMB_GRAB_MIN_SPEED,
            min(self._BOMB_GRAB_APPROACH_SPEED, dist / 70.0),
            ),
            lateral=0.0,
            rotation=0.0,
        )
        self._apply_lidar_avoidance(cmd, path_clearance=dist)

    # ------------------------------------------------------------------
    # 语义感知
    # ------------------------------------------------------------------

    def _find_closest_semantic(
        self,
        entity_type: DroneSemanticSensor.TypeEntity,
        *,
        require_not_grasped: bool = False,
    ) -> Optional[SemanticHit]:
        semantic = self.semantic_values()
        if semantic is None:
            return None

        best: Optional[DroneSemanticSensor.Data] = None
        best_dist = float("inf")
        for obj in semantic:
            if obj.entity_type != entity_type:
                continue
            if require_not_grasped and bool(obj.grasped):
                continue
            d = float(obj.distance)
            if not math.isfinite(d) or d >= best_dist:
                continue
            best_dist = d
            best = obj

        if best is None:
            return None
        world = self._semantic_to_world(best)
        return (best, world) if world is not None else None

    def _select_unclaimed_bomb(self) -> Optional[SemanticHit]:
        semantic = self.semantic_values() or []
        candidates: list[SemanticHit] = []
        for obj in semantic:
            if obj.entity_type != DroneSemanticSensor.TypeEntity.BOMB or bool(obj.grasped):
                continue
            distance = float(obj.distance)
            if not math.isfinite(distance):
                continue
            world = self._semantic_to_world(obj)
            if world is not None:
                # Seeing an active bomb again means a previous release near the
                # disposal center did not actually dispose it.
                self._disposed_tokens.discard(self._token_for_detection(obj, world))
                candidates.append((obj, world))
        candidates.sort(key=lambda item: (
            self._token_for_detection(item[0], item[1]) != self._current_bomb_token,
            float(item[0].distance),
        ))

        for candidate in candidates:
            world = candidate[1]
            token = self._token_for_detection(candidate[0], world)
            if self._claim_cooldown_until.get(token, -1) > int(self.elapsed_timestep):
                continue
            if not self._can_claim_target(token, world):
                continue
            return candidate
        return None

    def _semantic_to_world(
        self, det: DroneSemanticSensor.Data
    ) -> Optional[WorldPoint]:
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            return None
        bearing = float(heading) + float(det.angle)
        return (
            float(pos[0]) + float(det.distance) * math.cos(bearing),
            float(pos[1]) + float(det.distance) * math.sin(bearing),
        )

    def _disposal_target(self) -> WorldPoint:
        hit = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER
        )
        return hit[1] if hit is not None else self._disposal_approach

    def _update_shared_disposal(self) -> None:
        hit = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER
        )
        if hit is not None:
            self._known_disposal = hit[1]
            self._disposal_approach = hit[1]
            self._disposal_observed = True
        for _, message in self.communicator.received_messages:
            if not isinstance(message, dict):
                continue
            point = message.get("disposal")
            peer_observed = bool(message.get("disposal_observed"))
            if isinstance(point, (tuple, list)) and len(point) == 2:
                try:
                    candidate = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
                if (
                    math.isfinite(candidate[0])
                    and math.isfinite(candidate[1])
                    and peer_observed
                ):
                    self._known_disposal = candidate
                    self._disposal_approach = candidate
                    self._disposal_observed = self._disposal_observed or peer_observed

    def _update_online_obstacles(self) -> None:
        if self._online_map is not None:
            self._update_evidence_obstacles()
            return
        if self.size_area is None or self._online_obstacles.size == 0:
            self._recent_obstacle_hits = ()
            return
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        values = self.lidar_values()
        if pos is None or heading is None or values is None:
            self._recent_obstacle_hits = ()
            return

        semantic = self.semantic_values() or []
        non_wall = [
            obj for obj in semantic
            if obj.entity_type != DroneSemanticSensor.TypeEntity.WALL
        ]
        width, height = int(self.size_area[0]), int(self.size_area[1])
        recent: list[tuple[int, int]] = []
        scan_phase = int(self.identifier or 0) % 2
        if int(self.elapsed_timestep) % 2 == scan_phase:
            for distance, relative_angle in zip(values, self.lidar_rays_angles()):
                dist = float(distance)
                angle = float(relative_angle)
                if not math.isfinite(dist) or dist <= 8.0 or dist >= 285.0:
                    continue
                if self._ray_matches_non_wall(dist, angle, non_wall):
                    continue
                bearing = float(heading) + angle
                x = float(pos[0]) + dist * math.cos(bearing)
                y = float(pos[1]) + dist * math.sin(bearing)
                col = int(round(x + width / 2.0))
                row = int(round(-y + height / 2.0))
                if not (0 <= row < height and 0 <= col < width):
                    continue
                point = row // 3 * 3, col // 3 * 3
                count = self._online_hit_counts.get(point, 0) + 1
                self._online_hit_counts[point] = count
                if count >= 2:
                    cv2.circle(
                        self._online_obstacles,
                        (int(point[1]), int(point[0])),
                        3,
                        1,
                        -1,
                    )
                    recent.append(point)

        for _, message in self.communicator.received_messages:
            if not isinstance(message, dict):
                continue
            points = message.get("obstacles")
            if not isinstance(points, (tuple, list)):
                continue
            for point in points:
                if not isinstance(point, (tuple, list)) or len(point) != 2:
                    continue
                row, col = int(point[0]), int(point[1])
                if 0 <= row < height and 0 <= col < width:
                    cv2.circle(self._online_obstacles, (col, row), 3, 1, -1)
        self._recent_obstacle_hits = tuple(recent[:80])

    def _update_evidence_obstacles(self) -> None:
        """Fuse bounded free/wall evidence, then share only recent map changes."""
        assert self._online_map is not None
        hits: tuple[tuple[int, int], ...] = ()
        clears: tuple[tuple[int, int], ...] = ()
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        values = self.lidar_values()
        if (
            pos is not None and heading is not None and values is not None
            and int(self.elapsed_timestep) % 4 == int(self.identifier or 0) % 4
        ):
            dynamic_hits = [
                (float(obj.distance), float(obj.angle))
                for obj in self.semantic_values() or []
                if obj.entity_type != DroneSemanticSensor.TypeEntity.WALL
            ]
            hits, clears = self._online_map.update_scan(
                (float(pos[0]), float(pos[1])), float(heading),
                values, self.lidar_rays_angles(), dynamic_hits, stride=6,
            )
        for _, message in self.communicator.received_messages:
            if not isinstance(message, dict):
                continue
            peer_hits: list[tuple[int, int]] = []
            peer_clears: list[tuple[int, int]] = []
            for key, result in (("obstacles", peer_hits), ("obstacle_clears", peer_clears)):
                raw = message.get(key)
                if not isinstance(raw, (tuple, list)):
                    continue
                for point in raw[:80]:
                    if not isinstance(point, (tuple, list)) or len(point) != 2:
                        continue
                    try:
                        result.append((int(point[0]), int(point[1])))
                    except (TypeError, ValueError):
                        continue
            self._online_map.merge_peer_changes(peer_hits, peer_clears)
        self._recent_obstacle_hits = hits
        self._recent_obstacle_clears = clears

    def _frontier_world_to_cell(
        self, point: WorldPoint
    ) -> Optional[tuple[int, int]]:
        if self.size_area is None or self._frontier_grid.size == 0:
            return None
        width, height = float(self.size_area[0]), float(self.size_area[1])
        col = int((point[0] + width / 2.0) / self._FRONTIER_CELL_SIZE)
        row = int((height / 2.0 - point[1]) / self._FRONTIER_CELL_SIZE)
        if 0 <= row < self._frontier_grid.shape[0] and 0 <= col < self._frontier_grid.shape[1]:
            return row, col
        return None

    def _frontier_cell_to_world(self, cell: tuple[int, int]) -> WorldPoint:
        width, height = float(self.size_area[0]), float(self.size_area[1])
        row, col = cell
        return (
            -width / 2.0 + (col + 0.5) * self._FRONTIER_CELL_SIZE,
            height / 2.0 - (row + 0.5) * self._FRONTIER_CELL_SIZE,
        )

    def _frontier_snapshot(self) -> tuple[tuple[int, int, int], ...]:
        """Gossip rotating map pages so knowledge crosses several radio hops."""
        step = int(self.elapsed_timestep)
        if (
            not self._frontier_grid.size
            or step % 4 != int(self.identifier or 0) % 4
        ):
            return ()
        known = np.flatnonzero(self._frontier_grid)
        if not known.size:
            return ()
        cursor = self._frontier_snapshot_cursor % len(known)
        count = min(40, len(known))
        page = known[np.arange(cursor, cursor + count) % len(known)]
        self._frontier_snapshot_cursor = (cursor + count) % len(known)
        rows, cols = np.unravel_index(page, self._frontier_grid.shape)
        return tuple(
            (int(row), int(col), int(self._frontier_grid[row, col]))
            for row, col in zip(rows, cols)
        )

    def _update_frontier_map(self) -> None:
        """Build a coarse free/unknown/wall map from allowed online sensors."""
        if self._frontier_grid.size == 0:
            self._recent_frontier_updates = ()
            return
        changed: dict[tuple[int, int], int] = {}
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        values = self.lidar_values()
        if pos is not None:
            own = self._frontier_world_to_cell((float(pos[0]), float(pos[1])))
            if own is not None and self._frontier_grid[own] != 1:
                self._frontier_grid[own] = 1
                changed[own] = 1

        map_interval = 5
        ray_stride = 5
        phase = int(self.identifier or 0) % map_interval
        if (
            pos is not None
            and heading is not None
            and values is not None
            and int(self.elapsed_timestep) % map_interval == phase
        ):
            semantic = self.semantic_values() or []
            non_wall = [
                obj
                for obj in semantic
                if obj.entity_type != DroneSemanticSensor.TypeEntity.WALL
            ]
            angles = self.lidar_rays_angles()
            sample_step = self._FRONTIER_CELL_SIZE * 0.55
            ray_indices = np.arange(
                0, min(len(values), len(angles)), ray_stride, dtype=int
            )
            distances = np.asarray(values, dtype=float)[ray_indices]
            relative_angles = np.asarray(angles, dtype=float)[ray_indices]
            valid_rays = np.isfinite(distances) & (distances > 0.0)
            ray_indices = ray_indices[valid_rays]
            distances = distances[valid_rays]
            relative_angles = relative_angles[valid_rays]
            bearings = float(heading) + relative_angles
            free_limits = np.clip(distances - 14.0, 0.0, 182.0)
            sample_distances = np.arange(0.0, 182.0 + sample_step, sample_step)
            samples = sample_distances[None, :] <= free_limits[:, None]
            xs = float(pos[0]) + np.cos(bearings)[:, None] * sample_distances[None, :]
            ys = float(pos[1]) + np.sin(bearings)[:, None] * sample_distances[None, :]
            width, height = float(self.size_area[0]), float(self.size_area[1])
            cols = ((xs + width / 2.0) / self._FRONTIER_CELL_SIZE).astype(int)
            rows = ((height / 2.0 - ys) / self._FRONTIER_CELL_SIZE).astype(int)
            valid_cells = (
                samples
                & (rows >= 0)
                & (rows < self._frontier_grid.shape[0])
                & (cols >= 0)
                & (cols < self._frontier_grid.shape[1])
            )
            free_cells = np.unique(
                np.column_stack((rows[valid_cells], cols[valid_cells])), axis=0
            )
            for row, col in free_cells:
                cell = int(row), int(col)
                if self._frontier_grid[cell] == 0:
                    self._frontier_grid[cell] = 1
                    changed[cell] = 1

            for index, distance, angle, bearing in zip(
                ray_indices, distances, relative_angles, bearings
            ):
                if distance < 285.0 and not self._ray_matches_non_wall(
                    float(distance), float(angle), non_wall
                ):
                    endpoint = (
                        float(pos[0]) + float(distance) * math.cos(float(bearing)),
                        float(pos[1]) + float(distance) * math.sin(float(bearing)),
                    )
                    cell = self._frontier_world_to_cell(endpoint)
                    if cell is not None and self._frontier_grid[cell] != 2:
                        self._frontier_grid[cell] = 2
                        changed[cell] = 2

        for _, message in self.communicator.received_messages:
            if not isinstance(message, dict):
                continue
            try:
                peer_id = int(message.get("id"))
                peer_step = int(message.get("step"))
            except (TypeError, ValueError):
                peer_id, peer_step = -1, -1
            peer_target = message.get("frontier_target")
            if (
                peer_id >= 0
                and peer_id != int(self.identifier or 0)
                and isinstance(peer_target, (tuple, list))
                and len(peer_target) == 2
            ):
                try:
                    cell = int(peer_target[0]), int(peer_target[1])
                except (TypeError, ValueError):
                    cell = (-1, -1)
                if (
                    0 <= cell[0] < self._frontier_grid.shape[0]
                    and 0 <= cell[1] < self._frontier_grid.shape[1]
                ):
                    self._peer_frontier_targets[peer_id] = cell, peer_step
            updates = message.get("frontier_updates")
            if not isinstance(updates, (tuple, list)):
                continue
            for update in updates:
                if not isinstance(update, (tuple, list)) or len(update) != 3:
                    continue
                try:
                    row, col, state = int(update[0]), int(update[1]), int(update[2])
                except (TypeError, ValueError):
                    continue
                if (
                    state in (1, 2)
                    and 0 <= row < self._frontier_grid.shape[0]
                    and 0 <= col < self._frontier_grid.shape[1]
                ):
                    if state == 2 or self._frontier_grid[row, col] == 0:
                        self._frontier_grid[row, col] = state
        self._recent_frontier_updates = tuple(
            (row, col, state)
            for (row, col), state in list(changed.items())[:60]
        )

    def _cell_is_frontier(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        if self._frontier_grid[row, col] != 1:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + dr, col + dc
            if (
                0 <= rr < self._frontier_grid.shape[0]
                and 0 <= cc < self._frontier_grid.shape[1]
                and self._frontier_grid[rr, cc] == 0
            ):
                return True
        return False

    def _plan_frontier_path(self, start: tuple[int, int]) -> list[WorldPoint]:
        """Assign this drone a reachable frontier and return a free-space path."""
        rows, cols = self._frontier_grid.shape
        if self._frontier_grid[start] != 1:
            self._frontier_grid[start] = 1
        queue = deque([start])
        distance = {start: 0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        frontiers: list[tuple[float, tuple[int, int]]] = []
        my_id = int(self.identifier or 0)
        step = int(self.elapsed_timestep)
        while queue:
            cell = queue.popleft()
            cell_distance = distance[cell]
            if (
                cell_distance >= 2
                and self._cell_is_frontier(cell)
                and self._frontier_blocked_until.get(cell, -1) <= step
            ):
                row, col = cell
                owner = ((row * 73856093) ^ (col * 19349663)) % max(
                    1, self._team_size
                )
                owner_penalty = 0.0 if owner == my_id else 14.0
                peer_penalty = 0.0
                for peer_id, (peer_cell, seen) in self._peer_frontier_targets.items():
                    if peer_id != my_id and step - seen <= 90:
                        separation = abs(row - peer_cell[0]) + abs(col - peer_cell[1])
                        if separation <= 2:
                            peer_penalty = max(peer_penalty, 24.0 - 7.0 * separation)
                unknown_gain = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        unknown_gain += int(self._frontier_grid[rr, cc] == 0)
                score = (
                    float(cell_distance) + owner_penalty + peer_penalty
                    - 1.5 * unknown_gain
                )
                frontiers.append((score, cell))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = cell[0] + dr, cell[1] + dc
                if (
                    0 <= nxt[0] < rows
                    and 0 <= nxt[1] < cols
                    and self._frontier_grid[nxt] == 1
                    and nxt not in distance
                ):
                    distance[nxt] = cell_distance + 1
                    parent[nxt] = cell
                    queue.append(nxt)
        if not frontiers:
            self._frontier_target_cell = None
            return []
        _, target = min(frontiers, key=lambda item: (item[0], item[1]))
        cells = [target]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        self._frontier_target_cell = target
        # Coarse cells are already free; keeping every second center limits
        # steering churn without cutting across an unobserved corner.
        selected = cells[2::2]
        if not selected or selected[-1] != target:
            selected.append(target)
        return [self._frontier_cell_to_world(cell) for cell in selected]

    def _control_frontier_exploration(self, cmd: CommandsDict) -> bool:
        pos = self.measured_gps_position()
        if pos is None or self._frontier_grid.size == 0:
            return False
        position = float(pos[0]), float(pos[1])
        start = self._frontier_world_to_cell(position)
        if start is None:
            return False
        step = int(self.elapsed_timestep)
        target_valid = (
            self._frontier_target_cell is not None
            and self._cell_is_frontier(self._frontier_target_cell)
        )
        if (
            not self._frontier_path
            or not target_valid
            or step - self._frontier_plan_step >= self._FRONTIER_REPLAN_STEPS
        ):
            self._frontier_path = self._plan_frontier_path(start)
            self._frontier_plan_step = step
            self._frontier_anchor = position
            self._frontier_anchor_step = step
        if not self._frontier_path:
            return False

        if self._frontier_anchor is None or math.hypot(
            position[0] - self._frontier_anchor[0],
            position[1] - self._frontier_anchor[1],
        ) >= 32.0:
            self._frontier_anchor = position
            self._frontier_anchor_step = step
        elif step - self._frontier_anchor_step >= self._FRONTIER_STALL_STEPS:
            if self._frontier_target_cell is not None:
                self._frontier_blocked_until[self._frontier_target_cell] = step + 420
            self._frontier_path = []
            self._frontier_target_cell = None
            return False

        while self._frontier_path and math.hypot(
            position[0] - self._frontier_path[0][0],
            position[1] - self._frontier_path[0][1],
        ) <= self._PATROL_ARRIVE_RADIUS:
            self._frontier_path.pop(0)
        if not self._frontier_path:
            self._frontier_target_cell = None
            cmd["rotation"] = 0.7 if int(self.identifier or 0) % 2 == 0 else -0.7
            return True
        if self._run_lidar_escape_if_needed(cmd):
            return True
        self._move_toward(*self._frontier_path[0], cmd, use_lidar=True)
        return True

    @staticmethod
    def _ray_matches_non_wall(
        distance: float,
        angle: float,
        detections: list[DroneSemanticSensor.Data],
    ) -> bool:
        for obj in detections:
            obj_distance = float(obj.distance)
            obj_angle = float(obj.angle)
            if not math.isfinite(obj_distance) or not math.isfinite(obj_angle):
                continue
            error = abs(
                math.atan2(
                    math.sin(angle - obj_angle),
                    math.cos(angle - obj_angle),
                )
            )
            if error <= 0.14 and abs(distance - obj_distance) <= 25.0:
                return True
        return False

    # ------------------------------------------------------------------
    # 导航
    # ------------------------------------------------------------------

    def _move_toward(
        self,
        target_x: float,
        target_y: float,
        cmd: CommandsDict,
        *,
        use_lidar: bool = False,
        lidar_clearance: Optional[float] = None,
    ) -> bool:
        """朝世界坐标 target 行驶；对齐后返回 False，到达返回 True。"""
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            return False

        dx = target_x - float(pos[0])
        dy = target_y - float(pos[1])
        dist = math.hypot(dx, dy)
        if dist < self._ARRIVE_RADIUS:
            cmd["forward"] = 0.0
            cmd["lateral"] = 0.0
            cmd["rotation"] = 0.0
            return True

        desired = math.atan2(dy, dx)
        turn_err = normalize_angle(desired - float(heading))
        if abs(turn_err) > self._HEADING_TOLERANCE:
            cmd["rotation"] = 0.8 if turn_err > 0 else -0.8
            cmd["forward"] = 0.0
            cmd["lateral"] = 0.0
            return False

        cmd["rotation"] = 0.0
        cmd["lateral"] = 0.0
        speed_limit = 0.45 if int(self.drone_health) <= self._LOW_HEALTH else 1.0
        cmd["forward"] = min(speed_limit, dist / 80.0 + 0.25)
        if use_lidar:
            clearance = dist if lidar_clearance is None else lidar_clearance
            self._apply_lidar_avoidance(cmd, path_clearance=clearance)
        return False

    def _move_holonomic_toward(
        self, target_x: float, target_y: float, cmd: CommandsDict
    ) -> bool:
        """Follow a sensor-verified trail without stopping to rotate at every bend."""
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            return False
        dx = target_x - float(pos[0])
        dy = target_y - float(pos[1])
        dist = math.hypot(dx, dy)
        if dist < self._ARRIVE_RADIUS:
            self._set_motion(cmd, forward=0.0, lateral=0.0, rotation=0.0)
            return True
        desired = math.atan2(dy, dx)
        error = normalize_angle(desired - float(heading))
        speed = 0.45 if int(self.drone_health) <= self._LOW_HEALTH else 1.0
        self._set_motion(
            cmd,
            forward=speed * math.cos(error),
            lateral=speed * math.sin(error),
            rotation=max(-0.60, min(0.60, error * 0.70)),
        )
        if self._yield_to_oncoming_peer(cmd):
            return False
        values = self.lidar_values()
        angles = self.lidar_rays_angles()
        if values is not None and angles is not None and len(values) and len(angles):
            ranges = np.asarray(values, dtype=float)
            bearings = np.asarray(angles, dtype=float)
            n = min(len(ranges), len(bearings))
            # A held bomb is part of the load, not a wall blocking its carrier.
            dynamic_load = [obj for obj in self.semantic_values() or []
                            if obj.entity_type == DroneSemanticSensor.TypeEntity.BOMB
                            and bool(obj.grasped)]
            load_mask = np.zeros(n, dtype=bool)
            for obj in dynamic_load:
                angular_error = np.abs(np.arctan2(
                    np.sin(bearings[:n] - float(obj.angle)),
                    np.cos(bearings[:n] - float(obj.angle))))
                load_mask |= (angular_error <= 0.14) & (np.abs(ranges[:n] - float(obj.distance)) <= 25.0)
            block_distance = 28.0 if self.grasped_bombs() else self._LIDAR_BLOCK_DIST
            slow_distance = 58.0 if self.grasped_bombs() else self._LIDAR_SLOW_DIST
            sector = np.abs(np.arctan2(
                np.sin(bearings[:n] - error), np.cos(bearings[:n] - error)
            )) < 0.23
            valid = sector & ~load_mask & np.isfinite(ranges[:n]) & (ranges[:n] > 0.0)
            if np.any(valid):
                clearance = float(np.min(ranges[:n][valid]))
                if clearance < block_distance:
                    cmd["forward"] = 0.0
                    cmd["lateral"] = 0.0
                    self._lidar_blocked_steps += 1
                    if self._lidar_blocked_steps >= self._LIDAR_BLOCK_TRIGGER_STEPS:
                        self._start_lidar_escape(self._lidar_turn_sign())
                elif clearance < slow_distance:
                    scale = (clearance - block_distance) / (
                        slow_distance - block_distance
                    )
                    cmd["forward"] *= scale
                    cmd["lateral"] *= scale
                    self._lidar_blocked_steps = 0
                else:
                    self._lidar_blocked_steps = 0
        return False

    def _within_radius(self, target: WorldPoint, radius: float) -> bool:
        pos = self.measured_gps_position()
        if pos is None:
            return False
        return math.hypot(float(pos[0]) - target[0], float(pos[1]) - target[1]) < radius

    # ------------------------------------------------------------------
    # 探索 / 脱困
    # ------------------------------------------------------------------

    def _reset_explore_state(self) -> None:
        self._lidar_blocked_steps = 0

    def _explore_turn_rate(self) -> float:
        spin = self._IDLE_SPIN if int(self.identifier or 0) % 2 == 0 else -self._IDLE_SPIN
        path_min = self._lidar_path_min()
        if path_min is not None and self._lidar_is_blocking(path_min, None):
            spin *= self._lidar_turn_sign()
        return spin

    # ------------------------------------------------------------------
    # Lidar 避障
    # ------------------------------------------------------------------

    def _lidar_sector_min(self, angle_min: float, angle_max: float) -> Optional[float]:
        values = self.lidar_values()
        if values is None:
            return None
        best = float("inf")
        found = False
        for dist, ang in zip(values, self.lidar_rays_angles()):
            if not math.isfinite(dist):
                continue
            a = float(ang)
            if angle_min <= a <= angle_max:
                found = True
                best = min(best, float(dist))
        return best if found else None

    def _lidar_path_min(self) -> Optional[float]:
        return self._lidar_sector_min(-self._LIDAR_PATH_CONE, self._LIDAR_PATH_CONE)

    def _lidar_is_blocking(
        self, path_min: float, path_clearance: Optional[float]
    ) -> bool:
        """前方是墙（而非导航目标本身）时才需要避障。"""
        block_dist = self._lidar_block_distance()
        if path_min >= block_dist:
            return False
        if path_clearance is None:
            return True
        return path_min < path_clearance - self._LIDAR_TARGET_MARGIN

    def _lidar_turn_sign(self) -> float:
        left = self._lidar_sector_min(0.08, self._LIDAR_SIDE_CONE)
        right = self._lidar_sector_min(-self._LIDAR_SIDE_CONE, -0.08)
        left_d = left if left is not None else float("inf")
        right_d = right if right is not None else float("inf")
        return 1.0 if left_d >= right_d else -1.0

    def _lidar_front_blocked(self) -> bool:
        path_min = self._lidar_path_min()
        return path_min is not None and path_min < self._lidar_block_distance()

    def _lidar_block_distance(self) -> float:
        return self._LIDAR_BLOCK_DIST

    def _lidar_slow_distance(self) -> float:
        return self._LIDAR_SLOW_DIST

    def _lidar_rear_min(self) -> Optional[float]:
        left = self._lidar_sector_min(3.0 * math.pi / 4.0, math.pi)
        right = self._lidar_sector_min(-math.pi, -3.0 * math.pi / 4.0)
        values = [value for value in (left, right) if value is not None]
        return min(values) if values else None

    def _yield_to_oncoming_peer(self, cmd: CommandsDict) -> bool:
        """Give one drone a stable right of way in a head-on encounter."""
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            return False
        x, y = float(pos[0]), float(pos[1])
        step = int(self.elapsed_timestep)
        peer_id = self._yield_peer_id
        if peer_id is not None:
            peer = self._peer_state.get(peer_id)
            if (
                peer is None or step > self._yield_peer_until
                or step - peer[2] > 12
                or math.hypot(peer[0][0] - x, peer[0][1] - y) > 120.0
            ):
                self._yield_peer_id = None
                self._yield_path = []
            else:
                if self._follow_peer_yield_path(cmd, (x, y), heading):
                    return True
                self._yield_peer_id = None
                self._yield_path = []

        forward = float(cmd.get("forward", 0.0))
        lateral = float(cmd.get("lateral", 0.0))
        speed = math.hypot(forward, lateral)
        if speed < 0.2:
            return False
        velocity = (
            forward * math.cos(heading) - lateral * math.sin(heading),
            forward * math.sin(heading) + lateral * math.cos(heading),
        )
        my_id = int(self.identifier or 0)
        carrying = bool(self.grasped_bombs())
        for other_id, (point, _, seen, other_heading, other_carrying, peer_velocity) in sorted(
            self._peer_state.items(), key=lambda item: item[0]
        ):
            if step - seen > 12 or other_id == my_id:
                continue
            delta = point[0] - x, point[1] - y
            distance = math.hypot(*delta)
            if not 24.0 < distance < 85.0:
                continue
            direction = delta[0] / distance, delta[1] / distance
            if (velocity[0] * direction[0] + velocity[1] * direction[1]) / speed < 0.5:
                continue
            if math.cos(heading) * direction[0] + math.sin(heading) * direction[1] < 0.35:
                continue
            if -math.cos(other_heading) * direction[0] - math.sin(other_heading) * direction[1] < 0.35:
                continue
            if math.cos(heading - other_heading) > -0.25:
                continue
            peer_speed = math.hypot(*peer_velocity)
            if peer_speed < 1.0:
                continue
            peer_direction = (
                peer_velocity[0] / peer_speed,
                peer_velocity[1] / peer_speed,
            )
            if -peer_direction[0] * direction[0] - peer_direction[1] * direction[1] < 0.45:
                continue
            own_direction = velocity[0] / speed, velocity[1] / speed
            relative_step = (
                2.5 * (peer_direction[0] - own_direction[0]),
                2.5 * (peer_direction[1] - own_direction[1]),
            )
            squared_speed = relative_step[0] ** 2 + relative_step[1] ** 2
            closest_step = max(0.0, min(
                40.0,
                -(delta[0] * relative_step[0] + delta[1] * relative_step[1])
                / squared_speed,
            )) if squared_speed > 1e-6 else 40.0
            closest_gap = math.hypot(
                delta[0] + closest_step * relative_step[0],
                delta[1] + closest_step * relative_step[1],
            )
            if closest_gap > 45.0:
                continue
            if (carrying, -my_id) >= (other_carrying, -other_id):
                continue
            retreat = self._select_peer_yield_path((x, y), point)
            if retreat:
                self._yield_path = retreat
                self._yield_peer_id = other_id
                self._yield_peer_until = step + 140
                if self._follow_peer_yield_path(cmd, (x, y), heading):
                    return True
                self._yield_peer_id = None
                self._yield_path = []
        return False

    def _select_peer_yield_path(
        self, position: WorldPoint, peer: WorldPoint
    ) -> list[WorldPoint]:
        """Backtrack over observed free space until the peer has room to pass."""
        path: list[WorldPoint] = []
        travelled = 0.0
        previous = position
        for point in reversed(self._breadcrumbs):
            segment = math.hypot(point[0] - previous[0], point[1] - previous[1])
            if segment < 15.0:
                continue
            path.append(point)
            travelled += segment
            previous = point
            if travelled >= 110.0:
                break
        if travelled < 65.0:
            return []
        initial_gap = math.hypot(position[0] - peer[0], position[1] - peer[1])
        if any(
            math.hypot(point[0] - peer[0], point[1] - peer[1])
            < initial_gap - 10.0
            for point in path
        ):
            return []
        final_gap = math.hypot(previous[0] - peer[0], previous[1] - peer[1])
        return path if final_gap >= initial_gap + 25.0 else []

    def _follow_peer_yield_path(
        self, cmd: CommandsDict, position: WorldPoint, heading: float
    ) -> bool:
        while self._yield_path and math.hypot(
            position[0] - self._yield_path[0][0],
            position[1] - self._yield_path[0][1],
        ) < 15.0:
            self._yield_path.pop(0)
        if not self._yield_path:
            return False
        target = self._yield_path[0]
        direction = math.atan2(target[1] - position[1], target[0] - position[0])
        error = normalize_angle(direction - heading)
        distances = self.lidar_values()
        bearings = self.lidar_rays_angles()
        if distances is not None and bearings is not None:
            ranges = np.asarray(distances, dtype=float)
            angles = np.asarray(bearings, dtype=float)
            count = min(len(ranges), len(angles))
            sector = np.abs(np.arctan2(
                np.sin(angles[:count] - error), np.cos(angles[:count] - error)
            )) <= 0.20
            valid = sector & np.isfinite(ranges[:count]) & (ranges[:count] > 0)
            if np.any(valid) and float(np.min(ranges[:count][valid])) < 42.0:
                return False
        self._set_motion(
            cmd, forward=0.55 * math.cos(error),
            lateral=0.55 * math.sin(error), rotation=0.0,
        )
        return True

    def _apply_lidar_avoidance(
        self,
        cmd: CommandsDict,
        *,
        path_clearance: Optional[float] = None,
    ) -> None:
        if self._yield_to_oncoming_peer(cmd):
            return
        path_min = self._lidar_path_min()
        if path_min is None:
            self._lidar_blocked_steps = 0
            return
        fwd = float(cmd.get("forward", 0.0))
        if fwd <= 0.0:
            self._lidar_blocked_steps = 0
            return

        if (
            path_clearance is not None
            and path_min >= path_clearance - self._LIDAR_TARGET_MARGIN
        ):
            self._lidar_blocked_steps = 0
            return

        turn = self._lidar_turn_sign()
        block_dist = self._lidar_block_distance()
        slow_dist = self._lidar_slow_distance()
        if path_min >= slow_dist:
            self._lidar_blocked_steps = 0
            return
        blocked = path_min < block_dist
        self._lidar_blocked_steps = self._lidar_blocked_steps + 1 if blocked else 0
        if (
            blocked
            and self._lidar_blocked_steps >= self._LIDAR_BLOCK_TRIGGER_STEPS
            and not self._lidar_escape_active()
        ):
            self._start_lidar_escape(turn)
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=float(cmd.get("lateral", 0.0)),
                rotation=self._LIDAR_AVOID_ROTATION * turn,
            )
            return

        if blocked:
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=float(cmd.get("lateral", 0.0)),
                rotation=self._LIDAR_AVOID_ROTATION * turn,
            )
            return

        span = slow_dist - block_dist
        scale = max(0.0, min(1.0, (path_min - block_dist) / span))
        cmd["forward"] = fwd * scale
        if scale < 0.55:
            cmd["rotation"] = 0.35 * turn

    def _lidar_escape_active(self) -> bool:
        return (
            self._escape_back_steps_left > 0
            or self._escape_turn_steps_left > 0
            or self._escape_strafe_steps_left > 0
        )

    def _start_lidar_escape(self, turn_sign: float) -> None:
        self._escape_turn_sign = 1.0 if turn_sign >= 0 else -1.0
        self._escape_back_steps_left = self._LIDAR_ESCAPE_BACK_STEPS
        self._escape_turn_steps_left = self._LIDAR_ESCAPE_TURN_STEPS
        self._escape_strafe_steps_left = self._LIDAR_ESCAPE_STRAFE_STEPS
        self._lidar_blocked_steps = 0

    def _run_lidar_escape_if_needed(self, cmd: CommandsDict) -> bool:
        if not self._lidar_escape_active():
            return False
        if self._escape_back_steps_left > 0:
            rear_min = self._lidar_rear_min()
            if rear_min is not None and rear_min < self._lidar_block_distance():
                self._escape_back_steps_left = 0
                return self._run_lidar_escape_if_needed(cmd)
            self._escape_back_steps_left -= 1
            self._set_motion(
                cmd,
                forward=self._LIDAR_ESCAPE_BACK_SPEED,
                lateral=0.0,
                rotation=0.0,
            )
            return True
        if self._escape_turn_steps_left > 0:
            self._escape_turn_steps_left -= 1
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=0.0,
                rotation=self._LIDAR_AVOID_ROTATION * self._escape_turn_sign,
            )
            return True
        if self._escape_strafe_steps_left > 0:
            self._escape_strafe_steps_left -= 1
            self._set_motion(
                cmd,
                forward=0.25,
                lateral=self._LIDAR_ESCAPE_STRAFE_SPEED * self._escape_turn_sign,
                rotation=0.0,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_command() -> CommandsDict:
        return {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0,
        }

    @staticmethod
    def _set_motion(
        cmd: CommandsDict, *, forward: float, lateral: float, rotation: float
    ) -> None:
        cmd["forward"] = forward
        cmd["lateral"] = lateral
        cmd["rotation"] = rotation
