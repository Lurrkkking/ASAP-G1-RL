import argparse
from pathlib import Path
import numpy as np


def calc(path: Path):
    d = np.load(path)
    m = d['mask'].astype(bool)
    L = d['episode_lengths'].astype(np.int32)
    ds = np.linalg.norm(d['s_next'] - d['s'], axis=-1)[m]
    a = np.linalg.norm(d['a'], axis=-1)[m]
    z = d['s'][..., 2][m]
    motion_end_len = int(np.max(L)) if len(L) else 0
    fall_rate = float(np.mean(L < motion_end_len)) if len(L) else 0.0
    return {
        'episodes': int(len(L)),
        'motion_end_len': motion_end_len,
        'mean_len': float(L.mean()) if len(L) else 0.0,
        'min_len': int(L.min()) if len(L) else 0,
        'max_len': int(L.max()) if len(L) else 0,
        'fall_rate': fall_rate,
        'steps': int(m.sum()),
        'mean_a': float(a.mean()) if a.size else 0.0,
        'p95_a': float(np.percentile(a, 95)) if a.size else 0.0,
        'mean_ds': float(ds.mean()) if ds.size else 0.0,
        'p95_ds': float(np.percentile(ds, 95)) if ds.size else 0.0,
        'mean_z': float(z.mean()) if z.size else 0.0,
        'min_z': float(z.min()) if z.size else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--require-equal-episodes', action='store_true')
    args = p.parse_args()

    before = Path(args.before)
    after = Path(args.after)
    out = Path(args.out)

    if not before.is_file():
        raise FileNotFoundError(before)
    if not after.is_file():
        raise FileNotFoundError(after)

    b = calc(before)
    a = calc(after)

    if args.require_equal_episodes and b['episodes'] != a['episodes']:
        raise ValueError(f"episodes differ: before={b['episodes']} after={a['episodes']}")

    lines = []
    lines.append(f'before: {before}')
    lines.append(f'after : {after}')
    lines.append('')
    for name, r in [('before', b), ('after', a)]:
        lines.append(f'[{name}]')
        lines.append(f"episodes={r['episodes']} motion_end_len={r['motion_end_len']} mean_len={r['mean_len']:.2f} min/max={r['min_len']}/{r['max_len']}")
        lines.append(f"fall_rate={r['fall_rate']:.2%} valid_steps={r['steps']}")
        lines.append(f"mean||a||={r['mean_a']:.4f} p95||a||={r['p95_a']:.4f}")
        lines.append(f"mean||Δs||={r['mean_ds']:.4f} p95||Δs||={r['p95_ds']:.4f}")
        lines.append(f"mean_root_z={r['mean_z']:.4f} min_root_z={r['min_z']:.4f}")
        lines.append('')

    lines.append('[delta after-before]')
    for k in ['mean_len', 'fall_rate', 'mean_a', 'p95_a', 'mean_ds', 'p95_ds', 'mean_z', 'min_z']:
        lines.append(f"{k}={a[k]-b[k]:+.4f}")

    text = '\n'.join(lines) + '\n'
    out.write_text(text, encoding='utf-8')
    print(text)
    print(f"[DONE] wrote: {out}")


if __name__ == '__main__':
    main()
