"""Select and play one of the 15 recordings bundled with the v4 release."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', nargs='?', choices=[f'Map{i:02d}' for i in range(1, 6)], default='Map01')
    parser.add_argument('--layout', choices=['historic', 'stratified', 'edge_pairs'], default='historic')
    parser.add_argument('--list', action='store_true', help='List all bundled replay paths without opening a window.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / 'doc/evaluation_v4_15_rerun_20261004_120037'
    if args.list:
        for map_name in [f'Map{i:02d}' for i in range(1, 6)]:
            for layout in ('historic', 'stratified', 'edge_pairs'):
                path = data / f'{map_name}_hybrid_v4_{layout}_seed12345' / 'recordings' / f'team006_{map_name}_none_rd1.npz'
                print(map_name, layout, 'OK' if path.is_file() else 'MISSING', path.relative_to(root))
        return 0
    path = data / f'{args.map}_hybrid_v4_{args.layout}_seed12345' / 'recordings' / f'team006_{args.map}_none_rd1.npz'
    if not path.is_file():
        parser.error(f'Recording not found: {path}')
    environment = dict(os.environ, PYTHONPATH=str(root / 'src'))
    return subprocess.run([sys.executable, '-m', 'swarm_rescue.launcher', '--replay', str(path)],
                          cwd=root, env=environment).returncode


if __name__ == '__main__':
    raise SystemExit(main())
