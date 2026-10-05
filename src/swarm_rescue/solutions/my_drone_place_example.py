"""
红队示例控制器（team_mode: place）。

策略概要：
- 探索巡航 + Lidar 避障（忽略已放置炸弹，红队与炸弹无碰撞）+ 雷达脱困
- 周期性尝试放置炸弹
- 0 号机汇总全队库存后提交探索地图并结束回合

必实现方法：define_message_for_all()、control()。
传感器与执行器 API 见 DroneAbstract 文档及蓝队示例 my_drone_rescue_example.py。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.online_navigation import generic_patrol_route

WorldPoint = Tuple[float, float]
TeamMessage = dict[str, Any]


class MyDronePlaceExample(DroneAbstract):
    """红队最小示例：巡航放弹 + 0 号机提交探索矩阵。"""

    # --- 任务分工 ---
    _LEADER_DRONE_ID = 0
    _WAYPOINT_ARRIVE_RADIUS = 22.0
    _INVENTORY_LOG_EVERY_STEPS = 60
    _MAP_SUBMIT_RATIO = 0.48
    _MAP_STAMP_RADIUS = 5
    _MAP_CONFIRMATIONS = 2

    # --- 探索 / 脱困（与 my_drone_rescue_example 一致）---
    _EXPLORE_STRAIGHT_STEPS = 90
    _EXPLORE_TURN_STEPS = 28
    _EXPLORE_FORWARD = 0.7
    _ROAM_SPIN = 0.35

    # --- Lidar 避障 ---
    _LIDAR_BLOCK_DIST = 35.0
    _LIDAR_SLOW_DIST = 52.0
    _LIDAR_PATH_CONE = math.pi / 9
    _LIDAR_SIDE_CONE = math.pi / 4
    _LIDAR_SIDE_GAP = 0.08
    _LIDAR_TARGET_MARGIN = 10.0
    _LIDAR_AVOID_ROTATION = 0.85
    _LIDAR_BLOCK_TRIGGER_STEPS = 6
    _LIDAR_ESCAPE_BACK_STEPS = 10
    _LIDAR_ESCAPE_TURN_STEPS = 16
    _LIDAR_ESCAPE_STRAFE_STEPS = 10
    _LIDAR_ESCAPE_BACK_SPEED = -0.55
    _LIDAR_ESCAPE_STRAFE_SPEED = 0.65
    _LIDAR_BOMB_ANGLE_TOL = math.pi / 18
    _LIDAR_BOMB_DIST_TOL = 18.0

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs,
    ):
        super().__init__(
            identifier=identifier,
            misc_data=misc_data,
            display_lidar_graph=False,
            **kwargs,
        )
        self._steps = 0
        self._explore_tick = int(identifier or 0) * 31
        self._lidar_blocked_steps = 0
        self._escape_back_steps_left = 0
        self._escape_turn_steps_left = 0
        self._escape_strafe_steps_left = 0
        self._escape_turn_sign = 1.0
        self._exploration_submitted = False
        self._peer_inventory: dict[int, int] = {}
        self._map_hit_counts: dict[tuple[int, int], int] = {}
        self._team_map_hits: set[tuple[int, int]] = set()
        self._last_scan_hits: tuple[tuple[int, int], ...] = ()
        self._peer_full_maps: set[int] = set()
        size_key = tuple(int(v) for v in misc_data.size_area) if misc_data and misc_data.size_area else None
        my_id = int(identifier or 0)
        self._last_inventory = self.carried_bombs_count()
        self._place_pulse_high = False
        team_size = int(misc_data.number_drones or 1) if misc_data else 1
        self._unknown_patrol_route = generic_patrol_route(
            size_key, my_id, team_size, spacing=125.0
        )
        self._unknown_patrol_index = 0
        self._unknown_patrol_age = 0
        self._last_drop_position: Optional[WorldPoint] = None
        self._observed_disposal: Optional[WorldPoint] = None
        self._rendezvous: WorldPoint = (0.0, 0.0)
        self._home_position: Optional[WorldPoint] = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def define_message_for_all(self) -> TeamMessage:
        """Broadcast status plus compact lidar-map observations."""
        inventory = self.carried_bombs_count()
        full_map = None
        if (
            inventory == 0
            and self._at_rendezvous(radius=115.0)
            and self.elapsed_timestep % 10 == int(self.identifier or 0) % 10
        ):
            full_map = tuple(
                point for point, count in self._map_hit_counts.items()
                if count >= self._MAP_CONFIRMATIONS
            )
        return {
            "id": int(self.identifier or 0),
            "gps": self.measured_gps_position(),
            "heading": self.measured_compass_angle(),
            "inventory": inventory,
            "scan_hits": self._last_scan_hits,
            "full_map": full_map,
            "at_rendezvous": self._at_rendezvous(radius=115.0),
            "disposal": self._observed_disposal,
        }

    def control(self) -> CommandsDict:
        self._steps += 1
        if self._home_position is None:
            pos = self.measured_gps_position()
            if pos is not None:
                self._home_position = float(pos[0]), float(pos[1])
                self._rendezvous = self._home_position
        self._update_lidar_map()
        self._sync_peer_inventory_from_messages()
        self._update_target_after_placement()
        if self.carried_bombs_count() == 0:
            cmd = self._control_return_to_rendezvous()
        else:
            cmd = self._control_roam()
        self._maybe_place_unknown_bomb(cmd)
        if self._is_leader():
            self._leader_try_submit_exploration_map()
        return cmd

    # ------------------------------------------------------------------
    # 放置炸弹
    # ------------------------------------------------------------------

    def _update_target_after_placement(self) -> None:
        inventory = self.carried_bombs_count()
        if inventory < self._last_inventory:
            pos = self.measured_gps_position()
            if pos is not None:
                self._last_drop_position = float(pos[0]), float(pos[1])
        self._last_inventory = inventory

    def _maybe_place_unknown_bomb(self, cmd: CommandsDict) -> None:
        inventory = self.carried_bombs_count()
        if inventory <= 0:
            return
        if self._place_pulse_high:
            cmd["place_bomb"] = 0
            self._place_pulse_high = False
            return

        max_steps = int(getattr(self._misc_data, "max_timestep_limit", 2000) or 2000)
        my_id = int(self.identifier or 0)
        if inventory >= 2:
            threshold = int(max_steps * (0.14 + 0.012 * (my_id % 5)))
        else:
            threshold = int(max_steps * (0.30 + 0.012 * (my_id % 5)))
        force_deadline = int(max_steps * 0.41)
        if self._steps < threshold:
            return

        pos = self.measured_gps_position()
        if pos is None:
            return
        position = float(pos[0]), float(pos[1])
        if self._last_drop_position is not None:
            separation = math.hypot(
                position[0] - self._last_drop_position[0],
                position[1] - self._last_drop_position[1],
            )
            if separation < 110.0 and self._steps < force_deadline:
                return

        if self._steps < force_deadline and self._local_wall_opportunity() < 5:
            return
        cmd["forward"] = 0.0
        cmd["lateral"] = 0.0
        cmd["rotation"] = 0.0
        cmd["place_bomb"] = 1
        self._place_pulse_high = True

    def _local_wall_opportunity(self) -> int:
        values = self.lidar_values()
        if values is None or self.size_area is None:
            return 0
        blast_radius = 0.12 * min(float(self.size_area[0]), float(self.size_area[1]))
        semantic = self.semantic_values() or []
        non_wall = [
            obj for obj in semantic
            if obj.entity_type != DroneSemanticSensor.TypeEntity.WALL
        ]
        score = 0
        for distance, angle in zip(values, self.lidar_rays_angles()):
            dist = float(distance)
            ang = float(angle)
            if not math.isfinite(dist) or dist <= 8.0 or dist > blast_radius:
                continue
            if not self._ray_matches_non_wall(dist, ang, non_wall):
                score += 1
        return score

    def _move_to_world(self, target: WorldPoint, cmd: CommandsDict) -> None:
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            cmd["rotation"] = self._roam_spin_sign()
            return
        dx = target[0] - float(pos[0])
        dy = target[1] - float(pos[1])
        distance = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        error = math.atan2(math.sin(desired - float(heading)), math.cos(desired - float(heading)))
        if abs(error) > 0.22:
            cmd["rotation"] = max(-0.9, min(0.9, error * 1.6))
            return
        cmd["forward"] = min(0.9, max(0.12, distance / 70.0))
        cmd["rotation"] = max(-0.35, min(0.35, error * 0.8))
        self._apply_lidar_avoidance(cmd, path_clearance=distance)

    @staticmethod
    def _within_world_radius(a: WorldPoint, b: WorldPoint, radius: float) -> bool:
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= radius

    # ------------------------------------------------------------------
    # 探索地图提交（仅 0 号机）
    # ------------------------------------------------------------------

    def _is_leader(self) -> bool:
        return int(self.identifier or 0) == self._LEADER_DRONE_ID

    def _leader_try_submit_exploration_map(self) -> None:
        if self._exploration_submitted:
            return

        if self._steps % self._INVENTORY_LOG_EVERY_STEPS == 0:
            self._log_team_inventory_status()

        if self.size_area is None:
            return

        team_size = self._team_size() or 1
        max_steps = int(getattr(self._misc_data, "max_timestep_limit", 2000) or 2000)
        deadline = int(max_steps * self._MAP_SUBMIT_RATIO)
        all_inventory_reported_empty = self._all_teammates_report_zero_inventory()
        if not all_inventory_reported_empty and self._steps < deadline:
            return
        all_maps_received = (
            len(self._peer_full_maps | {int(self.identifier or 0)}) >= team_size
        )
        if not all_maps_received and self._steps < deadline:
            return

        self._log_team_inventory_status()
        print("[Drone0] 全队炸弹已放完，提交探索地图并结束回合")
        self._submit_lidar_exploration_map()
        self._exploration_submitted = True

    def _submit_lidar_exploration_map(self) -> None:
        w, h = (int(self.size_area[0]), int(self.size_area[1]))
        grid = np.zeros((h, w), dtype=np.uint8)
        local_hits = {
            point for point, count in self._map_hit_counts.items()
            if count >= self._MAP_CONFIRMATIONS
        }
        all_hits = local_hits | self._team_map_hits
        for row, col in all_hits:
            if 0 <= row < h and 0 <= col < w:
                cv2.circle(grid, (int(col), int(row)), self._MAP_STAMP_RADIUS, 1, -1)
        self._mask_disposal_region(grid)
        # Moving drones/bombs leave compact or irregular clusters.  Real walls
        # form elongated components, so keep long thin shapes and the boundary.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(grid, 8)
        cleaned = np.zeros_like(grid)
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            touches_boundary = x == 0 or y == 0 or x + width >= w or y + height >= h
            major = max(int(width), int(height))
            minor = max(1, min(int(width), int(height)))
            elongated = major >= 34 and major / minor >= 2.8
            if touches_boundary or (area >= 55 and elongated):
                cleaned[labels == label] = 1
        self.submit_exploration_map(cleaned)

    def _mask_disposal_region(self, grid: np.ndarray) -> None:
        if self.size_area is None:
            return
        if self._observed_disposal is not None:
            center_x, center_y = self._observed_disposal
            region_w, region_h = 280.0, 240.0
        else:
            return
        map_w, map_h = int(self.size_area[0]), int(self.size_area[1])
        padding = 16.0
        left = int(round(center_x - region_w / 2.0 - padding + map_w / 2.0))
        right = int(round(center_x + region_w / 2.0 + padding + map_w / 2.0))
        top = int(round(-(center_y + region_h / 2.0 + padding) + map_h / 2.0))
        bottom = int(round(-(center_y - region_h / 2.0 - padding) + map_h / 2.0))
        grid[max(0, top):min(map_h, bottom + 1), max(0, left):min(map_w, right + 1)] = 0

    def _sync_peer_inventory_from_messages(self) -> None:
        self._peer_inventory[int(self.identifier)] = self.carried_bombs_count()
        for _, msg in self.communicator.received_messages:
            if not isinstance(msg, dict):
                continue
            drone_id, inv = msg.get("id"), msg.get("inventory")
            if isinstance(drone_id, int) and isinstance(inv, (int, float)):
                self._peer_inventory[drone_id] = int(inv)
            hits = msg.get("scan_hits")
            if isinstance(hits, (tuple, list)):
                self._team_map_hits.update(
                    (int(p[0]), int(p[1])) for p in hits if isinstance(p, (tuple, list)) and len(p) == 2
                )
            full_map = msg.get("full_map")
            if isinstance(drone_id, int) and isinstance(full_map, (tuple, list)):
                self._team_map_hits.update(
                    (int(p[0]), int(p[1])) for p in full_map if isinstance(p, (tuple, list)) and len(p) == 2
                )
                self._peer_full_maps.add(drone_id)
            disposal = msg.get("disposal")
            if isinstance(disposal, (tuple, list)) and len(disposal) == 2:
                try:
                    point = float(disposal[0]), float(disposal[1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(point[0]) and math.isfinite(point[1]):
                    self._observed_disposal = point

    def _team_size(self) -> Optional[int]:
        if self._misc_data is None:
            return None
        return self._misc_data.number_drones

    def _all_teammates_report_zero_inventory(self) -> bool:
        n = self._team_size()
        if n is None:
            return False
        for i in range(n):
            if i not in self._peer_inventory or self._peer_inventory[i] != 0:
                return False
        return True

    def _log_team_inventory_status(self) -> None:
        n = self._team_size()
        if n is None:
            print(
                f"[Drone0] 提交等待: number_drones 未知, "
                f"已记录={dict(sorted(self._peer_inventory.items()))}"
            )
            return

        missing: list[int] = []
        not_zero: list[int] = []
        lines: list[str] = []
        for i in range(n):
            if i not in self._peer_inventory:
                lines.append(f"  #{i}: 未收到消息")
                missing.append(i)
            else:
                inv = self._peer_inventory[i]
                lines.append(f"  #{i}: {inv}")
                if inv != 0:
                    not_zero.append(i)

        ready = self._all_teammates_report_zero_inventory()
        in_range = len(self.communicator.comms_in_range)
        print(
            f"[Drone0] 库存汇总 step={self.elapsed_timestep} "
            f"已确认={len(self._peer_inventory)}/{n} "
            f"通信范围内={in_range} 可提交={ready}\n"
            + "\n".join(lines)
        )
        if missing:
            print(f"[Drone0] 仍缺消息: {missing}")
        if not_zero:
            print(f"[Drone0] 仍有库存: {not_zero}")

    # ------------------------------------------------------------------
    # Distributed lidar mapping and rendezvous
    # ------------------------------------------------------------------

    def _update_lidar_map(self) -> None:
        """Project wall-like lidar returns into the submitted pixel grid."""
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        values = self.lidar_values()
        if pos is None or heading is None or values is None or self.size_area is None:
            self._last_scan_hits = ()
            return

        semantic = self.semantic_values() or []
        for obj in semantic:
            if obj.entity_type != DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER:
                continue
            distance = float(obj.distance)
            angle = float(obj.angle)
            if not math.isfinite(distance) or not math.isfinite(angle):
                continue
            bearing = float(heading) + angle
            self._observed_disposal = (
                float(pos[0]) + distance * math.cos(bearing),
                float(pos[1]) + distance * math.sin(bearing),
            )
            break
        non_wall = [
            obj for obj in semantic
            if obj.entity_type != DroneSemanticSensor.TypeEntity.WALL
        ]
        width, height = int(self.size_area[0]), int(self.size_area[1])
        scan_hits: set[tuple[int, int]] = set()
        for distance, rel_angle in zip(values, self.lidar_rays_angles()):
            if not math.isfinite(float(distance)):
                continue
            dist = float(distance)
            angle = float(rel_angle)
            # Values close to the 300 px sensor range are noisy "no return"
            # samples, not obstacles.  Treating them as hits fills the map with
            # sparse rings and is heavily penalised by the scorer.
            if dist <= 8.0 or dist >= 285.0:
                continue
            if self._ray_matches_non_wall(dist, angle, non_wall):
                continue
            world_angle = float(heading) + angle
            x = float(pos[0]) + dist * math.cos(world_angle)
            y = float(pos[1]) + dist * math.sin(world_angle)
            if self._point_in_disposal_region(x, y):
                continue
            col = int(round(x + width / 2.0))
            row = int(round(-y + height / 2.0))
            if 0 <= row < height and 0 <= col < width:
                # Two-pixel quantisation removes most duplicate noisy returns;
                # the final disk stamp restores the physical wall thickness.
                scan_hits.add((row // 2 * 2, col // 2 * 2))

        confirmed: list[tuple[int, int]] = []
        for point in scan_hits:
            count = self._map_hit_counts.get(point, 0) + 1
            self._map_hit_counts[point] = count
            if count >= self._MAP_CONFIRMATIONS:
                confirmed.append(point)
        self._last_scan_hits = tuple(confirmed)

    def _point_in_disposal_region(self, x: float, y: float) -> bool:
        if self._observed_disposal is not None:
            center_x, center_y = self._observed_disposal
            width, height = 280.0, 240.0
        else:
            return False
        padding = 16.0
        return (
            abs(x - center_x) <= width / 2.0 + padding
            and abs(y - center_y) <= height / 2.0 + padding
        )

    @staticmethod
    def _ray_matches_non_wall(
        distance: float,
        angle: float,
        detections: list[DroneSemanticSensor.Data],
    ) -> bool:
        for obj in detections:
            obj_dist = float(obj.distance)
            obj_angle = float(obj.angle)
            if not math.isfinite(obj_dist) or not math.isfinite(obj_angle):
                continue
            angle_error = abs(math.atan2(math.sin(angle - obj_angle), math.cos(angle - obj_angle)))
            if angle_error <= 0.12 and abs(distance - obj_dist) <= 24.0:
                return True
        return False

    def _at_rendezvous(self, radius: float = 70.0) -> bool:
        pos = self.measured_gps_position()
        if pos is None:
            return False
        return math.hypot(
            float(pos[0]) - self._rendezvous[0],
            float(pos[1]) - self._rendezvous[1],
        ) <= radius

    def _control_return_to_rendezvous(self) -> CommandsDict:
        cmd = self._empty_command()
        if self._at_rendezvous():
            return cmd
        if self._run_lidar_escape_if_needed(cmd):
            return cmd
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            cmd["rotation"] = self._roam_spin_sign()
            return cmd
        dx = self._rendezvous[0] - float(pos[0])
        dy = self._rendezvous[1] - float(pos[1])
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        error = math.atan2(math.sin(desired - float(heading)), math.cos(desired - float(heading)))
        if abs(error) > 0.24:
            cmd["rotation"] = 0.8 if error > 0.0 else -0.8
            return cmd
        cmd["forward"] = min(0.85, 0.28 + dist / 180.0)
        self._apply_lidar_avoidance(cmd, path_clearance=dist)
        return cmd

    # ------------------------------------------------------------------
    # 巡航（探索 + 脱困 + Lidar）
    # ------------------------------------------------------------------

    def _control_roam(self) -> CommandsDict:
        cmd = self._empty_command()
        if self._run_lidar_escape_if_needed(cmd):
            return cmd

        if self._unknown_patrol_route:
            target = self._unknown_patrol_route[
                self._unknown_patrol_index % len(self._unknown_patrol_route)
            ]
            pos = self.measured_gps_position()
            arrived = False
            if pos is not None:
                arrived = self._within_world_radius(
                    (float(pos[0]), float(pos[1])), target, 28.0
                )
            if arrived or self._unknown_patrol_age >= 260:
                self._unknown_patrol_index = (
                    self._unknown_patrol_index + 1
                ) % len(self._unknown_patrol_route)
                self._unknown_patrol_age = 0
                target = self._unknown_patrol_route[self._unknown_patrol_index]
            self._move_to_world(target, cmd)
            self._unknown_patrol_age += 1
            return cmd

        cycle = self._EXPLORE_STRAIGHT_STEPS + self._EXPLORE_TURN_STEPS
        if self._explore_tick % cycle < self._EXPLORE_STRAIGHT_STEPS:
            cmd["forward"] = self._EXPLORE_FORWARD
            self._apply_lidar_avoidance(cmd)
        else:
            cmd["rotation"] = self._explore_turn_rate()
        self._explore_tick += 1
        return cmd

    def _roam_spin_sign(self) -> float:
        return self._ROAM_SPIN if int(self.identifier or 0) % 2 == 0 else -self._ROAM_SPIN

    def _explore_turn_rate(self) -> float:
        spin = self._roam_spin_sign()
        path_min = self._lidar_path_min()
        if path_min is not None and self._lidar_is_blocking(path_min, None):
            spin *= self._lidar_turn_sign()
        return spin

    # ------------------------------------------------------------------
    # Lidar 避障（红队：已放置炸弹不计入障碍）
    # ------------------------------------------------------------------

    def _lidar_path_min(self) -> Optional[float]:
        return self._lidar_sector_min(-self._LIDAR_PATH_CONE, self._LIDAR_PATH_CONE)

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
            if not (angle_min <= a <= angle_max):
                continue
            d = float(dist)
            if self._lidar_ray_hits_bomb(d, a):
                continue
            found = True
            best = min(best, d)
        return best if found else None

    def _lidar_ray_hits_bomb(self, dist: float, ang: float) -> bool:
        """语义确认为 BOMB 的 Lidar 读数忽略（放置阶段无碰撞）。"""
        semantic = self.semantic_values()
        if semantic is None:
            return False
        for obj in semantic:
            if obj.entity_type != DroneSemanticSensor.TypeEntity.BOMB:
                continue
            if bool(obj.grasped):
                continue
            if abs(float(obj.angle) - ang) > self._LIDAR_BOMB_ANGLE_TOL:
                continue
            if abs(float(obj.distance) - dist) <= self._LIDAR_BOMB_DIST_TOL:
                return True
        return False

    def _lidar_is_blocking(
        self, path_min: float, path_clearance: Optional[float]
    ) -> bool:
        if path_min >= self._LIDAR_BLOCK_DIST:
            return False
        if path_clearance is None:
            return True
        return path_min < path_clearance - self._LIDAR_TARGET_MARGIN

    def _lidar_turn_sign(self) -> float:
        left = self._lidar_sector_min(self._LIDAR_SIDE_GAP, self._LIDAR_SIDE_CONE)
        right = self._lidar_sector_min(-self._LIDAR_SIDE_CONE, -self._LIDAR_SIDE_GAP)
        left_d = left if left is not None else float("inf")
        right_d = right if right is not None else float("inf")
        return 1.0 if left_d >= right_d else -1.0

    def _apply_lidar_avoidance(
        self,
        cmd: CommandsDict,
        *,
        path_clearance: Optional[float] = None,
    ) -> None:
        path_min = self._lidar_path_min()
        if path_min is None:
            self._lidar_blocked_steps = 0
            return

        fwd = float(cmd.get("forward", 0.0))
        blocked = self._lidar_is_blocking(path_min, path_clearance)
        if fwd <= 0.0 or not blocked:
            self._lidar_blocked_steps = 0
            return

        turn = self._lidar_turn_sign()
        self._lidar_blocked_steps += 1
        if (
            self._lidar_blocked_steps >= self._LIDAR_BLOCK_TRIGGER_STEPS
            and not self._lidar_escape_active()
        ):
            self._start_lidar_escape(turn)
            cmd["forward"] = 0.0
            cmd["rotation"] = self._LIDAR_AVOID_ROTATION * turn
            return

        if path_min < self._LIDAR_BLOCK_DIST:
            cmd["forward"] = 0.0
            cmd["rotation"] = self._LIDAR_AVOID_ROTATION * turn
            return

        span = self._LIDAR_SLOW_DIST - self._LIDAR_BLOCK_DIST
        scale = max(0.0, min(1.0, (path_min - self._LIDAR_BLOCK_DIST) / span))
        cmd["forward"] = fwd * scale
        if scale < 0.55:
            cmd["rotation"] = 0.45 * turn

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
            self._escape_back_steps_left -= 1
            cmd["forward"] = self._LIDAR_ESCAPE_BACK_SPEED
            cmd["lateral"] = 0.0
            cmd["rotation"] = 0.0
            return True
        if self._escape_turn_steps_left > 0:
            self._escape_turn_steps_left -= 1
            cmd["forward"] = 0.0
            cmd["lateral"] = 0.0
            cmd["rotation"] = self._LIDAR_AVOID_ROTATION * self._escape_turn_sign
            return True
        if self._escape_strafe_steps_left > 0:
            self._escape_strafe_steps_left -= 1
            cmd["forward"] = 0.25
            cmd["lateral"] = self._LIDAR_ESCAPE_STRAFE_SPEED * self._escape_turn_sign
            cmd["rotation"] = 0.0
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
            "place_bomb": 0,
        }
