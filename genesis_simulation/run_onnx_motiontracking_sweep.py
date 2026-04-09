import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def load_metrics(path):
    data = np.load(path, allow_pickle=True)

    def fetch_scalar(key, default=np.nan):
        if key not in data:
            return default
        arr = np.asarray(data[key]).reshape(-1)
        if arr.size == 0:
            return default
        return arr[0].item() if hasattr(arr[0], 'item') else arr[0]

    return {
        'steps_executed': int(fetch_scalar('steps_executed', 0)),
        'fell': int(fetch_scalar('fell', 1)),
        'stop_reason': str(fetch_scalar('stop_reason', 'unknown')),
        'min_root_z': float(fetch_scalar('min_root_z', np.nan)),
        'mean_root_z': float(fetch_scalar('mean_root_z', np.nan)),
        'std_root_z': float(fetch_scalar('std_root_z', np.nan)),
        'mean_action_norm': float(fetch_scalar('mean_action_norm', np.nan)),
        'std_action_norm': float(fetch_scalar('std_action_norm', np.nan)),
    }


def score_metrics(m):
    if np.isnan(m['min_root_z']):
        return -1e9
    alive_bonus = 150.0 * (1 - m['fell'])
    score = (
        alive_bonus
        + 1.0 * m['steps_executed']
        + 120.0 * m['min_root_z']
        + 40.0 * m['mean_root_z']
        - 30.0 * m['std_root_z']
    )
    return float(score)


def make_run_env(base_env, args, combo, metrics_path, trace_path, tag):
    env = dict(base_env)
    env['ONNX_PATH'] = str(args.onnx)
    env['NUM_STEPS'] = str(args.num_steps)
    env['NO_RECORD'] = '1' if args.no_record else '0'
    env['STOP_ON_FALL'] = '1' if args.stop_on_fall else '0'
    env['FALL_ROOT_Z'] = f"{args.fall_root_z:.6f}"
    env['STOP_AT_MOTION_END'] = '1' if args.stop_at_motion_end else '0'
    env['USE_IMPLICIT_PD'] = '1' if args.use_implicit_pd else '0'
    env['ACTION_FILTER_ALPHA'] = f"{args.action_filter_alpha:.6f}"
    env['ACTION_SCALE'] = f"{args.action_scale:.6f}"
    env['METRICS_OUT_NPZ'] = str(metrics_path)
    env['TRACE_OUT_NPZ'] = str(trace_path) if args.save_trace else ''
    env['SWEEP_TAG'] = tag

    env['TIME_OFFSET'] = f"{combo['TIME_OFFSET']:.6f}"
    env['KD_SCALE'] = f"{combo['KD_SCALE']:.6f}"
    env['FLOOR_FRICTION'] = f"{combo['FLOOR_FRICTION']:.6f}"
    env['USE_SELF_COLLISION'] = str(combo['USE_SELF_COLLISION'])
    env['USE_MOTION_INIT'] = str(combo['USE_MOTION_INIT'])
    return env


def build_combos(args):
    grid = {
        'TIME_OFFSET': parse_csv_floats(args.time_offsets),
        'KD_SCALE': parse_csv_floats(args.kd_scales),
        'FLOOR_FRICTION': parse_csv_floats(args.floor_frictions),
        'USE_SELF_COLLISION': parse_csv_ints(args.use_self_collision),
        'USE_MOTION_INIT': parse_csv_ints(args.use_motion_init),
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))
        combos.append(combo)
    return combos


def main():
    parser = argparse.ArgumentParser(description='Sweep Genesis motion-tracking params for ONNX policy stability.')
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--num-steps', type=int, default=260)
    parser.add_argument('--time-offsets', type=str, default='0.0,0.5,1.0,1.5,2.0')
    parser.add_argument('--kd-scales', type=str, default='0.8,1.0,1.2')
    parser.add_argument('--floor-frictions', type=str, default='0.8,1.0,1.2')
    parser.add_argument('--use-self-collision', type=str, default='0,1')
    parser.add_argument('--use-motion-init', type=str, default='0,1')
    parser.add_argument('--use-implicit-pd', action='store_true')
    parser.add_argument('--stop-at-motion-end', action='store_true', default=True)
    parser.add_argument('--no-stop-at-motion-end', action='store_true')
    parser.add_argument('--no-record', action='store_true', default=True)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--stop-on-fall', action='store_true', default=True)
    parser.add_argument('--no-stop-on-fall', action='store_true')
    parser.add_argument('--fall-root-z', type=float, default=0.45)
    parser.add_argument('--action-filter-alpha', type=float, default=1.0)
    parser.add_argument('--action-scale', type=float, default=0.25)
    parser.add_argument('--max-runs', type=int, default=0, help='0 means run full grid')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--save-trace', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=Path('/tmp/mt_sweep'))
    parser.add_argument('--runner', type=Path, default=Path('/root/autodl-tmp/ASAP/genesis_simulation/run_onnx_motiontracking.py'))
    args = parser.parse_args()

    if args.record:
        args.no_record = False
    if args.no_stop_on_fall:
        args.stop_on_fall = False
    if args.no_stop_at_motion_end:
        args.stop_at_motion_end = False

    if not args.onnx.is_file():
        raise FileNotFoundError(f'ONNX not found: {args.onnx}')
    if not args.runner.is_file():
        raise FileNotFoundError(f'Runner script not found: {args.runner}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.out_dir / 'metrics'
    traces_dir = args.out_dir / 'traces'
    logs_dir = args.out_dir / 'logs'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if args.save_trace:
        traces_dir.mkdir(parents=True, exist_ok=True)

    combos = build_combos(args)
    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    print(f'[SWEEP] onnx={args.onnx}')
    print(f'[SWEEP] runner={args.runner}')
    print(f'[SWEEP] total_runs={len(combos)} out_dir={args.out_dir}')

    results = []
    for idx, combo in enumerate(combos):
        tag = f'run_{idx:04d}'
        metrics_path = metrics_dir / f'{tag}.npz'
        trace_path = traces_dir / f'{tag}.npz'
        log_path = logs_dir / f'{tag}.log'

        env = make_run_env(os.environ, args, combo, metrics_path, trace_path, tag)
        cmd = [sys.executable, str(args.runner)]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.time() - t0

        log_path.write_text(proc.stdout + '\n\n[STDERR]\n' + proc.stderr)
        metrics = {
            'steps_executed': 0,
            'fell': 1,
            'stop_reason': f'runner_exit_{proc.returncode}',
            'min_root_z': np.nan,
            'mean_root_z': np.nan,
            'std_root_z': np.nan,
            'mean_action_norm': np.nan,
            'std_action_norm': np.nan,
        }
        if proc.returncode == 0 and metrics_path.is_file():
            metrics = load_metrics(metrics_path)

        score = score_metrics(metrics)
        row = {
            'run_id': idx,
            'tag': tag,
            'returncode': proc.returncode,
            'elapsed_s': round(dt, 3),
            'score': score,
            **combo,
            **metrics,
            'log_path': str(log_path),
            'metrics_path': str(metrics_path),
            'trace_path': str(trace_path) if args.save_trace else '',
        }
        results.append(row)

        print(
            f"[SWEEP {idx+1}/{len(combos)}] rc={proc.returncode} score={score:.2f} "
            f"steps={metrics['steps_executed']} fell={metrics['fell']} "
            f"min_root_z={metrics['min_root_z']:.4f} "
            f"params={{TIME_OFFSET={combo['TIME_OFFSET']}, KD_SCALE={combo['KD_SCALE']}, "
            f"FLOOR_FRICTION={combo['FLOOR_FRICTION']}, USE_SELF_COLLISION={combo['USE_SELF_COLLISION']}, "
            f"USE_MOTION_INIT={combo['USE_MOTION_INIT']}}}"
        )

    results.sort(key=lambda x: x['score'], reverse=True)

    csv_path = args.out_dir / 'sweep_results.csv'
    json_path = args.out_dir / 'sweep_results.json'
    summary_path = args.out_dir / 'best_summary.json'

    if results:
        keys = list(results[0].keys())
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        with json_path.open('w') as f:
            json.dump(results, f, indent=2)

        top = results[: args.topk]
        with summary_path.open('w') as f:
            json.dump({'topk': top}, f, indent=2)

        print(f'\n[SWEEP] wrote: {csv_path}')
        print(f'[SWEEP] wrote: {json_path}')
        print(f'[SWEEP] wrote: {summary_path}')
        print(f'[SWEEP] top {min(args.topk, len(results))}:')
        for i, row in enumerate(top, 1):
            print(
                f"  {i:02d}. score={row['score']:.2f} steps={row['steps_executed']} "
                f"fell={row['fell']} min_root_z={row['min_root_z']:.4f} "
                f"TIME_OFFSET={row['TIME_OFFSET']} KD_SCALE={row['KD_SCALE']} "
                f"FLOOR_FRICTION={row['FLOOR_FRICTION']} SELF_COLL={row['USE_SELF_COLLISION']} "
                f"MOTION_INIT={row['USE_MOTION_INIT']}"
            )
    else:
        print('[SWEEP] no runs executed')


if __name__ == '__main__':
    main()
