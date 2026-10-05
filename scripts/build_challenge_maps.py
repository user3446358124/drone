"""Reproducibly build Map06--08 runtime data, editor files and preview diagrams.

Run from the repository: .venv/Scripts/python.exe scripts/build_challenge_maps.py
"""
import json
from pathlib import Path
from pprint import pformat

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["NO_COM_ZONE", "NO_GPS_ZONE", "KILL_ZONE"]
COLORS = {"NO_COM_ZONE": (255, 218, 35, 95),
          "NO_GPS_ZONE": (115, 126, 140, 85),
          "KILL_ZONE": (255, 71, 153, 100)}


def partitions(cols, rows, cw, ch, gap=160):
    walls = []
    for c in range(1, cols):
        for r in range(rows):
            x, y = c*cw, r*ch
            offset = 35 if (c+r) % 2 else -35
            mid = y+ch/2+offset
            walls += [(x, y, x, mid-gap/2), (x, mid+gap/2, x, y+ch)]
    for r in range(1, rows):
        for c in range(cols):
            x, y = c*cw, r*ch
            offset = -35 if (c+r) % 2 else 35
            mid = x+cw/2+offset
            walls += [(x, y, mid-gap/2, y), (mid+gap/2, y, x+cw, y)]
    return [list(map(int, w)) for w in walls]


def make_layouts():
    layouts = {}
    # Interconnected rooms: alternating door offsets, inset L-shaped screens.
    a = dict(title="通信断层城", size=[1680, 1260], start=[245, 1050],
             disposal=[25, 960, 135, 1140], walls=partitions(4, 3, 420, 420), boxes=[])
    for r in range(3):
        for c in range(4):
            if (c, r) == (0, 2):
                continue
            x, y = c*420, r*420
            a["walls"] += [[x+65,y+80,x+145,y+80], [x+145,y+80,x+145,y+155]]
            if (c+r) % 2 == 0:
                a["boxes"].append([x+290, y+290, x+335, y+335])
    a["zones"] = [
        {"type": TYPES[0], "rect": [350, 450, 520, 790]},
        {"type": TYPES[0], "rect": [1170, 70, 1360, 340]},
        {"type": TYPES[1], "rect": [770, 860, 1080, 1180]},
        {"type": TYPES[1], "rect": [470, 50, 710, 360]},
        {"type": TYPES[2], "rect": [555, 550, 625, 625]},
        {"type": TYPES[2], "rect": [1385, 900, 1470, 970]},
    ]
    a["bombs"] = [[225,210],[630,230],[1040,190],[1480,220],[220,630],
                  [660,710],[1040,650],[1480,640],[640,1070],[1480,1100]]
    layouts["06"] = a

    # Warehouse: long double shelves and offset cross-aisles.
    b = dict(title="失导航仓储区", size=[1800, 1440], start=[260,1260],
             disposal=[25,1170,145,1350], walls=partitions(4,3,450,480,180), boxes=[])
    for r in range(3):
        for c in range(4):
            if (c,r) == (0,2):
                continue
            x,y = c*450,r*480
            b["boxes"] += [[x+85,y+105,x+140,y+265], [x+305,y+210,x+355,y+365]]
    b["zones"] = [
        {"type": TYPES[0], "rect": [860, 80, 1040, 850]},
        {"type": TYPES[0], "rect": [480, 1000, 720, 1350]},
        {"type": TYPES[1], "rect": [170, 520, 360, 900]},
        {"type": TYPES[1], "rect": [1410, 510, 1700, 1280]},
        {"type": TYPES[2], "rect": [695, 555, 760, 635]},
        {"type": TYPES[2], "rect": [1130, 1220, 1200, 1290]},
    ]
    b["bombs"] = [[225,205],[685,335],[1130,260],[1580,355],[225,755],
                  [675,780],[1130,690],[1580,775],[675,1220],[1580,1230]]
    layouts["07"] = b

    # Dense diagonal rooms; multiple routes around isolated lethal pockets.
    c = dict(title="斜墙复合迷宫", size=[1600,1600], start=[245,1400],
             disposal=[25,1305,135,1495], walls=partitions(4,4,400,400,160), boxes=[])
    for r in range(4):
        for col in range(4):
            if (col,r) == (0,3):
                continue
            x,y=col*400,r*400
            if (col+r) % 2:
                c["walls"] += [[x+55,y+80,x+135,y+160],[x+265,y+255,x+340,y+330]]
            else:
                c["walls"] += [[x+70,y+70,x+145,y+70],[x+145,y+70,x+145,y+145]]
    c["zones"] = [
        {"type": TYPES[0], "rect": [320, 455, 525, 710]},
        {"type": TYPES[0], "rect": [885, 840, 1180, 1020]},
        {"type": TYPES[1], "rect": [860, 60, 1100, 360]},
        {"type": TYPES[1], "rect": [485, 870, 710, 1170]},
        {"type": TYPES[2], "rect": [1020, 580, 1090, 650]},
        {"type": TYPES[2], "rect": [1390, 1260, 1470, 1330]},
    ]
    c["bombs"] = [[220,230],[610,200],[1000,230],[1410,210],[200,620],
                  [630,620],[1400,640],[600,1030],[1010,1000],[1400,1450]]
    layouts["08"] = c
    return layouts


def editor_data(layout):
    elements=[]
    for x0,y0,x1,y1 in layout["walls"]:
        elements.append(dict(type="line_wall", points=[[x0,y0],[x1,y1]], thickness=6,color="#000000"))
    for rect in layout["boxes"]:
        elements.append(dict(type="rect_wall",top_left=rect[:2],bottom_right=rect[2:],color="#000000"))
    sx,sy=layout["start"]
    elements.append(dict(type="start_area",top_left=[sx-90,sy-90],bottom_right=[sx+90,sy+90],color="#278AFF"))
    elements.append(dict(type="disposal_center",top_left=layout["disposal"][:2],bottom_right=layout["disposal"][2:],color="#E74650"))
    for i,e in enumerate(elements): e["id"]=i+1
    # Separate extension: legacy editor supports walls/start/disposal only.
    return dict(version=1,canvas=dict(width=layout["size"][0],height=layout["size"][1],background="white"),
                elements=elements, special_zones=layout["zones"])


def preview(layout, number):
    w,h=layout["size"]
    img=Image.new("RGBA",(w,h),"white")
    overlay=Image.new("RGBA",img.size)
    d=ImageDraw.Draw(overlay)
    for z in layout["zones"]: d.rectangle(z["rect"],fill=COLORS[z["type"]])
    img=Image.alpha_composite(img,overlay)
    d=ImageDraw.Draw(img)
    d.rectangle((3,3,w-4,h-4),outline="#17273B",width=6)
    for wall in layout["walls"]: d.line(wall,fill="#17273B",width=6)
    for box in layout["boxes"]: d.rectangle(box,fill="#17273B")
    d.rectangle(layout["disposal"],fill="#E74650")
    sx,sy=layout["start"]
    d.rectangle((sx-90,sy-90,sx+90,sy+90),outline="#278AFF",width=4)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    d.text((sx-80,sy-83),"出生区",font=font,fill="#2076D0")
    x0,y0,_,_=layout["disposal"]
    d.text((x0+6,y0+12),"处置中心",font=font,fill="white")
    short={TYPES[0]:"禁通信",TYPES[1]:"无 GPS",TYPES[2]:"杀戮区"}
    for z in layout["zones"]:
        x0,y0,_,_=z["rect"]
        d.text((x0+4,y0+4),short[z["type"]],font=font,fill="#243047")
    for i,(x,y) in enumerate(layout["bombs"]):
        d.ellipse((x-12,y-12,x+12,y+12),fill="#263D54",outline="white",width=2)
        d.text((x+15,y-14),str(i+1),font=font,fill="#263D54")
    img.convert("RGB").save(ROOT/f"map_data/map_{number}.png")
    return img.convert("RGB")


def main():
    layouts=make_layouts()
    out=ROOT/'output/challenge_maps'
    out.mkdir(parents=True,exist_ok=True)
    module=ROOT/'src/swarm_rescue/maps/challenge_layouts.py'
    module.write_text('"""Generated simulator layouts; do not import in team algorithms."""\nLAYOUTS = '+pformat(layouts,width=100,sort_dicts=False)+'\n',encoding='utf-8')
    thumbs=[]
    for num,layout in layouts.items():
        (ROOT/f'map_data/map_{num}.json').write_text(json.dumps(editor_data(layout),ensure_ascii=False,indent=2),encoding='utf-8')
        (ROOT/f'src/swarm_rescue/maps/map_{num}.py').write_text(
            'from swarm_rescue.maps.challenge_map import ChallengeMap\n'
            'from swarm_rescue.maps.challenge_layouts import LAYOUTS\n\n\n'
            f'class Map{num}(ChallengeMap):\n    layout = LAYOUTS["{num}"]\n',encoding='utf-8')
        w,h=layout['size']
        bombs=dict(map_name=f'Map{num}',zones_config=TYPES,
                   bombs=[dict(x=x-w/2,y=h/2-y,theta=0) for x,y in layout['bombs']])
        (ROOT/f'map_data/bombs_Map{num}_challenge.json').write_text(json.dumps(bombs,indent=2),encoding='utf-8')
        im=preview(layout,num)
        im.thumbnail((560,480))
        tile=Image.new('RGB',(600,600),'#F3F6FA')
        tile.paste(im,((600-im.width)//2,60))
        td=ImageDraw.Draw(tile)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
        td.text((22,15),f'Map{num}  {layout["title"]}',font=font,fill='#152A40')
        td.text((22,565),f'{w} × {h}  |  10 个测试目标  |  6 个特殊区',font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18),fill='#405775')
        thumbs.append(tile)
        for mode in ('place','rescue'):
            config=('stat_saving_enabled: false\nstate_recording_enabled: true\nvideo_capture_enabled: false\n\nevaluation_plan:\n'
                    f'  - map_name: Map{num}\n    team_mode: {mode}\n    nb_rounds: 1\n    config_weight: 1\n'
                    f'    number_drones: {5 if mode=="place" else 10}\n    initial_bombs_per_drone: 2\n'
                    '    zones_config: [NO_COM_ZONE, NO_GPS_ZONE, KILL_ZONE]\n')
            (ROOT/f'config/challenge_{mode}_Map{num}.yml').write_text(config,encoding='utf-8')
            cmd=('@echo off\r\ncd /d "%~dp0..\\.."\r\n'
                 f'".venv\\Scripts\\python.exe" -m swarm_rescue.launcher --competition --config "config\\challenge_{mode}_Map{num}.yml"\r\n'
                 'if errorlevel 1 pause\r\n')
            (out/f'run_{mode}_Map{num}.cmd').write_text(cmd,encoding='ascii')
    sheet=Image.new('RGB',(1800,600),'white')
    for i,tile in enumerate(thumbs): sheet.paste(tile,(600*i,0))
    sheet.save(out/'Map06-08_overview.png')
    print('Built Map06--08, editor JSON/PNG, bomb fixtures and six evaluation configs.')


if __name__=='__main__':
    main()
