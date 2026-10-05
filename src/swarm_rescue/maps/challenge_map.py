"""Shared runtime for the Map06--08 training environments.

Layout data is simulator-side only. It must not be imported by team controllers.
"""
import math
import random

from swarm_rescue.simulation.drone.drone_base import DroneBase
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox
from swarm_rescue.simulation.elements.sensor_disablers import (
    KillZone, NoComZone, NoGpsZone, ZoneType,
)
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.collision_handlers import get_colliding_entities
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.definitions import CollisionTypes
from swarm_rescue.simulation.utils.misc_data import MiscData


class LethalKillZone(KillZone):
    """Pink zone: body contact permanently removes a drone in the same step.

The original KillZone remains unchanged for older maps. Its device-disabling
behavior is retained here; the extra body callback implements actual death.
"""


def _remove_killed_drone(space, key, playground, drone):
    if not drone.removed:
        drone.grasper._release_grasping()
        playground.remove(drone)


def _lethal_body_contact(arbiter, space, data):
    playground = data["playground"]
    zone, base = get_colliding_entities(playground, arbiter)
    if isinstance(zone, LethalKillZone) and isinstance(base, DroneBase):
        drone = base.agent
        if not drone.removed and drone.drone_health > 0:
            drone._drone_health = 0
            # Mutating the physics space during a collision callback is unsafe.
            space.add_post_step_callback(
                _remove_killed_drone, ("kill-zone", id(drone)), playground, drone
            )
    return True


class ChallengeMap(MapAbstract):
    layout = None

    def __init__(self, drone_type, zones_config=(), number_drones=None):
        super().__init__(drone_type, zones_config, number_drones=number_drones)
        layout = self.layout
        self._size_area = tuple(layout["size"])
        self._max_timestep_limit = 9600
        self._max_walltime_limit = 360
        self._start_area_drones = self.to_world(layout["start"])
        self._number_drones = number_drones if number_drones is not None else 10
        self._playground = ClosedPlayground(size=self._size_area)
        self._special_zones = []
        classes = {"NO_COM_ZONE": NoComZone, "NO_GPS_ZONE": NoGpsZone,
                   "KILL_ZONE": LethalKillZone}
        for spec in layout["zones"]:
            if ZoneType[spec["type"]] not in self._zones_config:
                continue
            x0, y0, x1, y1 = spec["rect"]
            zone = classes[spec["type"]](size=(x1 - x0, y1 - y0))
            self._playground.add(zone, (self.to_world(((x0+x1)/2, (y0+y1)/2)), 0))
            self._special_zones.append(zone)
        handler = self._playground.space.add_collision_handler(
            CollisionTypes.DISABLER_ZONE, CollisionTypes.DRONE
        )
        handler.data["playground"] = self._playground
        handler.pre_solve = _lethal_body_contact
        for x0, y0, x1, y1 in layout["walls"]:
            wall = NormalWall(self.to_world((x0, y0)), self.to_world((x1, y1)))
            self._playground.add(wall, wall.wall_coordinates)
        for x0, y0, x1, y1 in layout["boxes"]:
            box = NormalBox(up_left_point=self.to_world((x0, y0)),
                            width=x1-x0, height=y1-y0)
            self._playground.add(box, box.wall_coordinates)
        x0, y0, x1, y1 = layout["disposal"]
        self._disposal_center = DisposalCenter(size=(x1-x0, y1-y0))
        self._disposal_center_pos = (self.to_world(((x0+x1)/2, (y0+y1)/2)), 0)
        self._playground.add(self._disposal_center, self._disposal_center_pos)
        self._explored_map.initialize_walls(self._playground)
        self._bombs_pos = [self.to_world(pt) for pt in layout["bombs"]]
        self._bombs_path = []
        self._bombs = []
        self._number_bombs = 0
        self.spawn_bombs([{"x": x, "y": y} for x, y in self._bombs_pos])
        self._drones_pos = []
        self._drones = []
        side = math.ceil(math.sqrt(self._number_drones))
        sx, sy = self._start_area_drones
        misc = MiscData(size_area=self._size_area, number_drones=self._number_drones,
                        max_timestep_limit=self._max_timestep_limit,
                        max_walltime_limit=self._max_walltime_limit)
        for i in range(self._number_drones):
            pos = ((sx+(i % side-(side-1)/2)*40,
                    sy+(i // side-(side-1)/2)*40), random.uniform(-math.pi, math.pi))
            drone = drone_type(identifier=i, misc_data=misc)
            self._drones.append(drone)
            self._drones_pos.append(pos)
            self._playground.add(drone, pos)

    def to_world(self, point):
        """Editor pixels (top-left origin) to simulator coordinates."""
        return (point[0]-self._size_area[0]/2, self._size_area[1]/2-point[1])

    def clear_bombs(self):
        # Pre-round fixtures should not become ten ghost slots in red recordings.
        for bomb in list(self._bombs):
            self._playground.remove(bomb, definitive=True)
        self._bombs = []
        self._number_bombs = 0
