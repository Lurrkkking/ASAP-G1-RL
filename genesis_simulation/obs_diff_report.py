import argparse
import json
from pathlib import Path

import numpy as np


ACTOR_OBS_BLOCKS = [
    ('actions', 23),
    ('base_ang_vel', 3),
    ('dof_pos', 23),
    ('dof_vel', 23),
    ('history_actor', 304),
    ('projected_gravity', 3),
    ('ref_motion_phase', 1),
]

HISTORY_BLOCKS = [
    ('actions', 4 * 23),
    ('base_ang_vel', 4 * 3),
    ('dof_pos', 4 * 23),
    ('dof_vel', 4 * 23),
    ('projected_gravity', 4 * 3),
    ('ref_motion_phase', 4 * 1),
]


def stats(x, y):
    d = x - y
    abs_d = np.abs(d)
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(abs_d))
    max_abs = float(np.max(abs_d))
    p99 = float(np.percentile(abs_d, 99))

    x1 = x.reshape(-1)
    y1 = y.reshape(-1)
    xnorm = float(np.linalg.norm(x1))
    ynorm = float(np.linalg.norm(y1))
    if xnorm < 1e-12 or ynorm < 1e-12:
        cos = float('nan')
    else:
        cos = float(np.dot(x1, y1) / (xnorm * ynorm))
    return {
        'mae': mae,
        'rmse': rmse,
        'max_abs': max_abs,
        'p99_abs': p99,
        'cosine': cos,
    }


def slice_blocks(arr, blocks):
    out = {}
    st = 0
    for name, dim in blocks:
        ed = st + dim
        out[name] = arr[:, st:ed]
        st = ed
    if st != arr.shape[1]:
        raise ValueError(f'Block dims sum {st} != feature dim {arr.shape[1]}')
    return out


def load_trace(path: Path):
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def align_len(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n], n


def main():
    parser = argparse.ArgumentParser(description='Compare rollout trace npz files and report actor_obs/action mismatch.')
    parser.add_argument('--ref', type=Path, required=True, help='reference trace npz (e.g., IsaacGym export)')
    parser.add_argument('--cand', type=Path, required=True, help='candidate trace npz (e.g., Genesis trace)')
    parser.add_argument('--out-json', type=Path, default=None)
    args = parser.parse_args()

    if not args.ref.is_file():
        raise FileNotFoundError(f'Reference trace not found: {args.ref}')
    if not args.cand.is_file():
        raise FileNotFoundError(f'Candidate trace not found: {args.cand}')

    ref = load_trace(args.ref)
    cand = load_trace(args.cand)

    for key in ('actor_obs',):
        if key not in ref or key not in cand:
            raise KeyError(f'Both traces must contain key: {key}')

    ref_actor, cand_actor, n = align_len(ref['actor_obs'], cand['actor_obs'])
    if n == 0:
        raise ValueError('No overlapping timesteps between traces')

    report = {
        'ref': str(args.ref),
        'cand': str(args.cand),
        'steps_compared': int(n),
        'actor_obs': {
            'overall': stats(ref_actor, cand_actor),
            'blocks': {},
            'history_blocks': {},
        },
    }

    ref_blocks = slice_blocks(ref_actor, ACTOR_OBS_BLOCKS)
    cand_blocks = slice_blocks(cand_actor, ACTOR_OBS_BLOCKS)
    for name, _ in ACTOR_OBS_BLOCKS:
        report['actor_obs']['blocks'][name] = stats(ref_blocks[name], cand_blocks[name])

    ref_hist = slice_blocks(ref_blocks['history_actor'], HISTORY_BLOCKS)
    cand_hist = slice_blocks(cand_blocks['history_actor'], HISTORY_BLOCKS)
    for name, _ in HISTORY_BLOCKS:
        report['actor_obs']['history_blocks'][name] = stats(ref_hist[name], cand_hist[name])

    if 'action' in ref and 'action' in cand:
        ref_action, cand_action, n_act = align_len(ref['action'], cand['action'])
        if n_act > 0:
            report['action'] = {
                'steps_compared': int(n_act),
                'overall': stats(ref_action, cand_action),
            }

    if 'phase' in ref and 'phase' in cand:
        ref_phase, cand_phase, n_phase = align_len(ref['phase'], cand['phase'])
        if n_phase > 0:
            report['phase'] = {
                'steps_compared': int(n_phase),
                'overall': stats(ref_phase.reshape(-1, 1), cand_phase.reshape(-1, 1)),
            }

    if 'root_z' in ref and 'root_z' in cand:
        ref_rz, cand_rz, n_rz = align_len(ref['root_z'], cand['root_z'])
        if n_rz > 0:
            report['root_z'] = {
                'steps_compared': int(n_rz),
                'overall': stats(ref_rz.reshape(-1, 1), cand_rz.reshape(-1, 1)),
            }

    print(f"[REPORT] steps_compared={report['steps_compared']}")
    a = report['actor_obs']['overall']
    print(
        '[REPORT] actor_obs overall '
        f"mae={a['mae']:.6e} rmse={a['rmse']:.6e} max={a['max_abs']:.6e} p99={a['p99_abs']:.6e}"
    )

    ranked = sorted(
        report['actor_obs']['blocks'].items(),
        key=lambda kv: kv[1]['mae'],
        reverse=True,
    )
    print('[REPORT] actor_obs blocks by MAE:')
    for name, st in ranked:
        print(f"  - {name:16s} mae={st['mae']:.6e} rmse={st['rmse']:.6e} max={st['max_abs']:.6e}")

    ranked_hist = sorted(
        report['actor_obs']['history_blocks'].items(),
        key=lambda kv: kv[1]['mae'],
        reverse=True,
    )
    print('[REPORT] history_actor blocks by MAE:')
    for name, st in ranked_hist:
        print(f"  - {name:16s} mae={st['mae']:.6e} rmse={st['rmse']:.6e} max={st['max_abs']:.6e}")

    if 'action' in report:
        st = report['action']['overall']
        print(f"[REPORT] action overall mae={st['mae']:.6e} rmse={st['rmse']:.6e} max={st['max_abs']:.6e}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open('w') as f:
            json.dump(report, f, indent=2)
        print(f'[REPORT] wrote {args.out_json}')


if __name__ == '__main__':
    main()
