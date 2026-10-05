"""Offline fixture generation; geometry is never exposed to controllers."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label

BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=BASE/'eval_fixtures'/'bomb_layouts')
parser.add_argument('--seed-offset',type=int,default=0,help='Change bomb coordinates independently of simulation RNG.')
parser.add_argument('--maps',type=int,nargs='+',default=list(range(1,6)))
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
manifest=[]
for map_index in args.maps:
    map_name=f'Map{map_index:02}'
    data=json.loads((BASE/'map_data'/f'map_{map_index:02}.json').read_text(encoding='utf-8'))
    width,height=data['canvas']['width'],data['canvas']['height']
    obstacles=np.zeros((height,width),dtype=np.uint8)
    eligible=np.ones((height,width),dtype=bool)
    start=None
    for element in data['elements']:
        kind=element['type']
        if kind=='line_wall':
            points=element['points']
            cv2.line(obstacles,tuple(points[0]),tuple(points[1]),1,int(element.get('thickness',6))+2)
        elif kind=='rect_wall':
            cv2.rectangle(obstacles,tuple(element['top_left']),tuple(element['bottom_right']),1,-1)
        elif kind in ('disposal_center','start_area'):
            x0,y0=element['top_left']; x1,y1=element['bottom_right']
            eligible[max(0,y0-30):min(height,y1+31),max(0,x0-30):min(width,x1+31)]=False
            if kind=='start_area': start=((x0+x1)//2,(y0+y1)//2)
    assert start is not None
    clearance=cv2.distanceTransform(1-obstacles,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    components,_=label(clearance>=28)
    connected=components[start[1],start[0]]
    assert connected>0
    eligible &= (clearance>=34)&(components==connected)
    # Downsample eligible centers for deterministic, bounded selection.
    yy,xx=np.mgrid[0:height:5,0:width:5]
    valid=eligible[yy,xx]
    candidates=np.column_stack((xx[valid],yy[valid]))
    assert len(candidates)>=10
    for scenario,seed in [('stratified',11000+map_index+args.seed_offset),('edge_pairs',22000+map_index+args.seed_offset)]:
        rng=np.random.default_rng(seed)
        points=[]
        if scenario=='stratified':
            for row in range(2):
                for col in range(5):
                    region=candidates[(candidates[:,0]>=col*width/5)&(candidates[:,0]<(col+1)*width/5)
                                      &(candidates[:,1]>=row*height/2)&(candidates[:,1]<(row+1)*height/2)]
                    pool=region if len(region) else candidates
                    order=rng.permutation(len(pool))
                    chosen=next((pool[i] for i in order if not points or min(np.linalg.norm(pool[i]-p) for p in points)>=85),None)
                    assert chosen is not None,(map_name,scenario,row,col)
                    points.append(chosen)
        else:
            # Pair close targets near walls or far boundaries; each pair is
            # distinct (38--65 spacing), never intersecting or inside a wall.
            edge=((candidates[:,0]<width*.18)|(candidates[:,0]>width*.82)
                  |(candidates[:,1]<height*.18)|(candidates[:,1]>height*.82)
                  |(clearance[candidates[:,1],candidates[:,0]]<65))
            pool=candidates[edge]
            for i in rng.permutation(len(pool)):
                base=pool[i]
                if points and min(np.linalg.norm(base-p) for p in points)<150: continue
                distance=np.linalg.norm(candidates-base,axis=1)
                partners=candidates[(distance>=38)&(distance<=65)]
                if len(partners)==0: continue
                partner=partners[int(rng.integers(len(partners)))]
                if points and min(np.linalg.norm(partner-p) for p in points)<95: continue
                points.extend([base,partner])
                if len(points)==10: break
            assert len(points)==10,(map_name,scenario)
        assert all(eligible[int(p[1]),int(p[0])] for p in points)
        path=OUT/f'{map_name}_{scenario}.json'
        document={'map_name':map_name,'zones_config':[],
                  'bombs':[{'x':float(p[0]-width/2),'y':float(height/2-p[1]),'theta':0.0} for p in points]}
        path.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest.append(dict(map=map_name,scenario=scenario,layout_seed=seed,path=path.name,
                             minimum_wall_clearance=float(min(clearance[int(p[1]),int(p[0])] for p in points)),
                             reachable_with_radius=28,number_bombs=10))
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
