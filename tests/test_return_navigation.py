import numpy as np
from swarm_rescue.solutions.return_navigation import observed_return_route, path_length

def fixtures():
    return np.full((75,100),-12,dtype=np.int8), np.zeros((600,800),dtype=np.uint8)

def test_known_open_space_shortens_looping_return_to_observed_home():
    evidence, obstacles = fixtures()
    start=(200,0)
    trail=[(200,180),(-200,180),(-200,-180),(-300,0)]
    route, verified=observed_return_route((800,600),evidence,obstacles,start,trail,(-300,0))
    assert verified and route[-1]==(-300,0)
    assert path_length(start,route)<path_length(start,trail)*0.6

def test_unknown_shortcut_is_not_assumed_free():
    evidence, obstacles=fixtures()
    evidence[:,49:51]=0
    trail=[(200,180),(-200,180),(-300,0)]
    route, verified=observed_return_route((800,600),evidence,obstacles,(200,0),trail,(-300,0))
    assert not verified and route==trail

def test_observed_wall_forces_detour_and_preserves_clearance():
    evidence, obstacles=fixtures()
    obstacles[:400,397:403]=1
    start=(200,0)
    trail=[(200,-230),(-200,-230),(-300,0)]
    route, verified=observed_return_route((800,600),evidence,obstacles,start,trail,(-300,0))
    assert verified
    assert any(y<-125 for _,y in route)
    assert path_length(start,route)>500

def test_goal_relocated_out_of_wall_buffer_is_rejected():
    evidence, obstacles=fixtures()
    obstacles[:,397:403]=1
    trail=[(10,0)]
    route, verified=observed_return_route((800,600),evidence,obstacles,(200,0),trail,(10,0))
    assert not verified and route==trail
