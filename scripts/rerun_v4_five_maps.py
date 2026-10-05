"""Replay the released v4 benchmark on five maps and three layouts each."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from evaluate_bomb_layouts import evaluate


def main():
    base = Path(__file__).resolve().parents[1]
    fixtures = base / 'doc/evaluation_v4_15_rerun_20261004_120037/layouts'
    output = base / 'results_swarm_rescue' / ('v4_15_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True)
    os.environ['SWARM_RESCUE_DIAGNOSTICS'] = '1'
    rows = []
    for index in range(1, 6):
        for scenario in ('historic', 'stratified', 'edge_pairs'):
            task = {'map': f'Map{index:02d}', 'scenario': scenario}
            bombs = fixtures / f"{task['map']}_{scenario}.json"
            if not bombs.exists():
                raise FileNotFoundError(bombs)
            rows.append(evaluate(base, 'formal_v4', task, bombs, 12345, output, Path(sys.executable)))
            (output / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"All disposed: {sum(row['full_disposal'] for row in rows)}/{len(rows)}")
    print('Results:', output)


if __name__ == '__main__':
    main()
