"""Serial paired evaluation with independent bomb-layout and simulation seeds.

Run with the project virtual environment. Controllers receive only normal
sensor inputs; bomb files are passed to the simulator, never to the solution.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(repo):
    hashes = {str(p.relative_to(repo)): digest(p)
              for p in sorted((repo / 'src').rglob('*.py'))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def evaluate(repo, name, task, bombs, seed, output, python):
    out = output / f"{task['map']}_{name}_{task['scenario']}_seed{seed}"
    identity = {'version': name, 'map': task['map'], 'layout': task['scenario'],
                'seed': seed, 'bombs_sha256': digest(bombs),
                'source_sha256': source_digest(repo), 'repo': str(repo)}
    marker = out / 'metrics.json'
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if all(previous.get(k) == v for k, v in identity.items()):
            return previous
        raise RuntimeError(f'Inputs changed for {out}; choose a fresh output directory.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'bombs_input.json').write_bytes(bombs.read_bytes())
    config = out / 'eval.yml'
    config.write_text(
        'stat_saving_enabled: true\nstate_recording_enabled: true\n'
        'video_capture_enabled: false\nevaluation_plan:\n'
        f"  - map_name: {task['map']}\n    team_mode: rescue\n"
        '    nb_rounds: 1\n    config_weight: 1\n    number_drones: 10\n'
        '    zones_config: []\n', encoding='utf-8')
    environment = dict(os.environ, PYTHONPATH=str(repo / 'src'),
                       PYTHONHASHSEED=str(seed), PYTHONIOENCODING='utf-8',
                       SWARM_RESCUE_RESULT_PATH=str(out))
    program = (
        'import random,numpy as np,runpy,sys; '
        f'random.seed({seed}); np.random.seed({seed}); '
        'import swarm_rescue.solutions.my_drone_rescue_example as s; '
        "print('EVALUATED_SOLUTION',s.__file__,flush=True); "
        f"sys.argv=['swarm_rescue.launcher','--headless','--competition',"
        f"'--stop_at_first_crash','--config',{str(config)!r},"
        f"'--bombs-file',{str(out / 'bombs_input.json')!r}]; "
        "runpy.run_module('swarm_rescue.launcher',run_name='__main__')")
    print(f"START {name} {task['map']} {task['scenario']} seed={seed}", flush=True)
    started = time.monotonic()
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        try:
            code = subprocess.run([str(python), '-u', '-c', program], cwd=repo,
                                  env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=900).returncode
        except subprocess.TimeoutExpired:
            code = -999
    result = dict(identity, exit_code=code, duration_seconds=round(time.monotonic() - started, 2),
                  output=str(out))
    stats = out / 'team006_stats.csv'
    if stats.exists():
        with stats.open(encoding='utf-8-sig', newline='') as stream:
            matches = [row for row in csv.DictReader(stream) if row.get('Map') == task['map']]
        if matches:
            result.update(matches[-1])
    result['full_disposal'] = (code == 0 and float(result.get('Rescued Percent', 0)) >= 100
                               and float(result.get('Crashed', 1)) == 0)
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"DONE full={result['full_disposal']} disposed={result.get('Rescued Percent')} "
          f"health={result.get('Health Return Score')}", flush=True)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=root)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--layouts', type=Path, required=True, help='Generated manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[12345])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    manifest = args.layouts.resolve()
    tasks = json.loads(manifest.read_text(encoding='utf-8'))
    if not tasks or not args.seeds or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('Provide at least one layout and valid NumPy seeds in [0, 2**32).')
    cases = [(task['map'], task['scenario']) for task in tasks]
    if len(cases) != len(set(cases)):
        parser.error('Manifest map/scenario names must be unique.')
    rows = []
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        bomb_path = Path(task['path'])
        if not bomb_path.is_absolute():
            bomb_path = manifest.parent / bomb_path
        for seed in args.seeds:
            for name, repo in [('baseline', args.baseline), ('candidate', args.candidate)]:
                rows.append(evaluate(repo.resolve(), name, task, bomb_path, seed,
                                     output, args.python.resolve()))
                (output / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    pairs = [rows[i:i+2] for i in range(0, len(rows), 2)]
    regressions = [{'map': a['map'], 'layout': a['layout'], 'seed': a['seed']}
                   for a, b in pairs if a['full_disposal'] and not b['full_disposal']]
    gate = {'cases_per_version': len(pairs),
            'baseline_full': sum(a['full_disposal'] for a, _ in pairs),
            'candidate_full': sum(b['full_disposal'] for _, b in pairs),
            'candidate_passes_all_cases': all(b['full_disposal'] for _, b in pairs),
            'full_disposal_regressions': regressions}
    (output / 'gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if gate['candidate_passes_all_cases'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
