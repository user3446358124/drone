"""Render actual simulator maps and run bounded competition-mode smoke checks.

The 120-step runs check integration only; their scores are NOT benchmarks.
"""
import argparse
import contextlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/challenge_maps'


def smoke(number, mode):
    from swarm_rescue.launcher import Launcher
    from swarm_rescue.maps.challenge_map import ChallengeMap
    original=ChallengeMap.__init__
    def bounded(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self._max_timestep_limit=120
        self._max_walltime_limit=60
    ChallengeMap.__init__=bounded
    # Use the normal YAML and launcher; only shorten this process's round.
    launcher=Launcher(config_path=str(ROOT/f'config/challenge_{mode}_Map{number}.yml'))
    ok=launcher.go(headless=True,competition_mode=True,stop_at_first_crash=True)
    if not ok: raise RuntimeError(f'Map{number} {mode} smoke failed')
    for path in Path(launcher.result_path).glob('recordings/*.npz'):
        with np.load(path,allow_pickle=True) as data:
            meta=data['metadata'].item()
            assert set(meta['zones_config'])=={'NO_COM_ZONE','NO_GPS_ZONE','KILL_ZONE'}
            assert meta['n_timesteps']>0
            assert meta['max_bombs']==10 if mode=='rescue' else meta['max_bombs']>=0
            print('VERIFIED_RECORDING',path,meta)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',nargs=2,metavar=('NUMBER','MODE'))
    args=parser.parse_args()
    if args.smoke:
        smoke(*args.smoke)
        return
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue.simulation.gui_map.top_down_view import TopDownView
    OUT.mkdir(parents=True,exist_ok=True)
    for number in ('06','07','08'):
        cls=getattr(importlib.import_module(f'swarm_rescue.maps.map_{number}'),f'Map{number}')
        env=cls(DroneMotionless,zones_config=tuple(ZoneType))
        try:
            view=TopDownView(env.playground,draw_interactive=False)
            view.update_and_draw_in_framebuffer(force=True)
            Image.fromarray(np.flipud(view.get_np_img())).save(OUT/f'Map{number}_simulator.png')
        finally:
            env.playground.window.close()
    for number in ('06','07','08'):
        for mode in ('place','rescue'):
            path=OUT/f'smoke_Map{number}_{mode}.log'
            with path.open('w',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,__file__,'--smoke',number,mode],
                                      cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print(f'Map{number} {mode}: exit {result.returncode}',flush=True)
            if result.returncode:
                raise RuntimeError(path.read_text(encoding='utf-8')[-5000:])


if __name__=='__main__':
    main()
