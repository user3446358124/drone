"""Geometry and real physics/sensor checks for custom challenge maps."""
import importlib
import gc
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label

from swarm_rescue.maps.challenge_layouts import LAYOUTS
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType

ALL_ZONES = tuple(ZoneType)


@pytest.fixture(autouse=True)
def collect_closed_graphics_contexts():
    # Old cyclic Arcade windows must be finalized before a new current window.
    gc.collect()
    yield
    gc.collect()


def navigable_components(layout):
    w,h = layout["size"]
    mask=Image.new("L",(w,h),0)
    d=ImageDraw.Draw(mask)
    d.rectangle((0,0,w-1,h-1),outline=1,width=6)
    for wall in layout["walls"]: d.line(wall,fill=1,width=7)
    for box in layout["boxes"]: d.rectangle(box,fill=1)
    d.rectangle(layout["disposal"],fill=1)
    for z in layout["zones"]:
        if z["type"] == "KILL_ZONE": d.rectangle(z["rect"],fill=1)
    clearance=distance_transform_edt(np.asarray(mask)==0)
    # A conservative 40 px radius for drone + carried target and safety margin.
    labels,_=label(clearance >= 40)
    return clearance,labels


@pytest.mark.parametrize("number",["06","07","08"])
def test_loaded_map_geometry_and_zones(number):
    cls=getattr(importlib.import_module(f"swarm_rescue.maps.map_{number}"),f"Map{number}")
    env=cls(DroneMotionless,zones_config=ALL_ZONES)
    try:
        layout=LAYOUTS[number]
        clearance,components=navigable_components(layout)
        sx,sy=layout["start"]
        component=components[sy,sx]
        assert component>0
        for x,y in layout["bombs"]:
            assert components[y,x]==component, (number,"unreachable bomb",x,y,clearance[y,x])
        for pose in env._drones_pos:
            x=round(pose[0][0]+layout["size"][0]/2)
            y=round(layout["size"][1]/2-pose[0][1])
            assert clearance[y,x]>=20, (number,"unsafe spawn",x,y)
            for zone in layout["zones"]:
                x0,y0,x1,y1=zone["rect"]
                assert not (x0-20<=x<=x1+20 and y0-20<=y<=y1+20)
        x0,y0,x1,y1=layout["disposal"]
        # Accessible approach on the right of the disposal center.
        assert components[(y0+y1)//2,x1+50]==component
        assert len(env.bombs)==10
        assert len(env._special_zones)==6
        assert env.size_area not in {(1250,800),(1200,900),(1113,750),(1660,1122),(1200,1200)}
        env.playground.step()
        assert all(d.drone_health==50 and not d.removed for d in env.drones)
    finally:
        env.playground.window.close()


@pytest.mark.parametrize("number",["06","07","08"])
def test_zones_can_be_disabled(number):
    cls=getattr(importlib.import_module(f"swarm_rescue.maps.map_{number}"),f"Map{number}")
    env=cls(DroneMotionless,zones_config=(),number_drones=1)
    try:
        assert not env._special_zones
    finally:
        env.playground.window.close()


def move_to(env,drone,pixel):
    # Harness-only teleport: never exposed to the competition controller.
    body=drone.base._pm_body
    body.position=env.to_world(pixel)
    body.velocity=(0,0)
    body.angular_velocity=0
    env.playground.space.reindex_shapes_for_body(body)
    env.playground.step()


@pytest.mark.parametrize("number",["06","07","08"])
def test_zone_effects_and_recovery(number):
    cls=getattr(importlib.import_module(f"swarm_rescue.maps.map_{number}"),f"Map{number}")
    env=cls(DroneMotionless,zones_config=ALL_ZONES,number_drones=1)
    drone=env.drones[0]
    try:
        # Choose a collision-free interior pixel in each sensor-disabling zone.
        clearance,_=navigable_components(LAYOUTS[number])
        for kind in ("NO_COM_ZONE","NO_GPS_ZONE"):
            spec=next(z for z in LAYOUTS[number]["zones"] if z["type"]==kind)
            x0,y0,x1,y1=spec["rect"]
            candidates=clearance[y0+35:y1-35,x0+35:x1-35]
            iy,ix=np.unravel_index(np.argmax(candidates),candidates.shape)
            move_to(env,drone,(x0+35+ix,y0+35+iy))
            if kind=="NO_COM_ZONE":
                assert drone.communicator_is_disabled()
                assert drone.communicator.send("probe") is None
                assert not drone.gps_is_disabled()
            else:
                assert drone.gps_is_disabled() and drone.compass_is_disabled()
                assert drone.gps_values() is None and drone.compass_values() is None
                assert not drone.odometer_is_disabled()
                assert drone.odometer_values() is not None
                assert not drone.lidar_is_disabled()
            move_to(env,drone,LAYOUTS[number]["start"])
            assert not drone.communicator_is_disabled()
            assert not drone.gps_is_disabled() and not drone.compass_is_disabled()
        spec=next(z for z in LAYOUTS[number]["zones"] if z["type"]=="KILL_ZONE")
        x0,y0,x1,y1=spec["rect"]
        move_to(env,drone,((x0+x1)/2,(y0+y1)/2))
        assert drone.drone_health==0 and drone.removed
        assert drone not in env.playground.agents
        env.playground.step()
        assert drone.removed and drone.drone_health==0
    finally:
        env.playground.window.close()


def test_death_recording_keeps_survivor_slots():
    from swarm_rescue.maps.map_06 import Map06
    from swarm_rescue.simulation.reporting.state_recorder import StateRecorder
    env=Map06(DroneMotionless,zones_config=ALL_ZONES,number_drones=2)
    try:
        env.clear_bombs()
        recorder=StateRecorder()
        recorder.capture_state(env.playground,0)
        assert not recorder._bomb_frames[-1]
        survivor=env.drones[1]
        x0,y0,x1,y1=next(z['rect'] for z in LAYOUTS['06']['zones'] if z['type']=='KILL_ZONE')
        move_to(env,env.drones[0],((x0+x1)/2,(y0+y1)/2))
        recorder.capture_state(env.playground,1)
        frame=recorder._drone_frames[-1]
        assert len(frame)==2
        assert not frame[0]['active'] and frame[0]['health']==0
        assert frame[1]['active'] and frame[1]['x']==survivor.base.position.x
        assert env.playground.agents==[survivor]
    finally:
        env.playground.window.close()
