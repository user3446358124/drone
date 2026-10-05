import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import ast

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eval_layouts', ROOT / 'scripts/evaluate_bomb_layouts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=ROOT)
parser.add_argument('--baseline', type=Path)
parser.add_argument('--output', type=Path)
parser.add_argument('--name', required=True)
parser.add_argument('--cases', nargs='+', default=['Map01:historic','Map01:edge_pairs','Map05:historic'])
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--contacts', action='store_true')
args = parser.parse_args()
if args.contacts:
    raw_run = module.subprocess.run
    instrumentation = '''
import atexit, collections
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
_contacts = collections.defaultdict(lambda: [set(), set()])
def _contact_wrapper(method, kind):
    def tracked(self):
        _contacts[int(self.identifier)][kind].add(int(self.elapsed_timestep))
        return method(self)
    return tracked
DroneAbstract.collide_wall = _contact_wrapper(DroneAbstract.collide_wall, 0)
DroneAbstract.collide_drone = _contact_wrapper(DroneAbstract.collide_drone, 1)
def _save_contacts():
    import json, os
    from pathlib import Path
    values = {str(k): {'wall_frames':len(v[0]), 'peer_frames':len(v[1])} for k,v in _contacts.items()}
    total = {'wall_frames':sum(v['wall_frames'] for v in values.values()),
             'peer_frames':sum(v['peer_frames'] for v in values.values()), 'drones':values}
    (Path(os.environ['SWARM_RESCUE_RESULT_PATH'])/'contacts.json').write_text(json.dumps(total), encoding='utf-8')
atexit.register(_save_contacts)
'''
    def instrumented_run(command, *a, **kw):
        command = list(command)
        if len(command) > 3 and command[2] == '-c':
            command[3] = command[3].replace("runpy.run_module('swarm_rescue.launcher',run_name='__main__')",
                                         instrumentation + "\nrunpy.run_module('swarm_rescue.launcher',run_name='__main__')")
        return raw_run(command, *a, **kw)
    module.subprocess.run = instrumented_run
output = args.output or ROOT / 'doc' / ('evaluation_' + args.name)
layouts = ROOT / 'doc/evaluation_v4_15_rerun_20261004_120037/layouts'
os.environ['SWARM_RESCUE_DIAGNOSTICS'] = '1'
results = []
for case in args.cases:
    map_name, scenario = case.split(':')
    bombs = layouts / f'{map_name}_{scenario}.json'
    if not bombs.exists() and map_name in ('Map06','Map07','Map08') and scenario == 'default':
        # Simulator fixtures only. This data never enters the controller.
        syntax = ast.parse((ROOT/'src/swarm_rescue/maps/challenge_layouts.py').read_text(encoding='utf-8'))
        definitions = ast.literal_eval(next(node.value for node in syntax.body if isinstance(node,ast.Assign)))
        layout = definitions[map_name[-2:]]
        width, height = layout['size']
        document = {'map_name':map_name, 'zones_config':[],
                    'bombs':[{'x':x-width/2,'y':height/2-y,'theta':0.0} for x,y in layout['bombs']]}
        fixture_dir = output/'layouts'
        fixture_dir.mkdir(parents=True, exist_ok=True)
        bombs=fixture_dir/f'{map_name}_{scenario}.json'
        bombs.write_text(json.dumps(document,indent=2),encoding='utf-8')
    versions = [(args.name+'_baseline',args.baseline),(args.name+'_candidate',args.repo)] if args.baseline else [(args.name,args.repo)]
    for version, repo in versions:
        result = module.evaluate(repo.resolve(), version,
                                 {'map':map_name, 'scenario':scenario},
                                 bombs, args.seed,
                                 output, Path(sys.executable))
        results.append(result)
        (output / f'{args.name}_{args.seed}_summary.json').write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result['exit_code'] != 0:
            raise SystemExit(result['exit_code'])
