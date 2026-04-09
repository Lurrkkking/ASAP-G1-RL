import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def load_actor_from_checkpoint(pt_path: Path):
    ckpt = torch.load(pt_path, map_location='cpu')
    if 'actor_model_state_dict' not in ckpt:
        raise KeyError('Checkpoint missing actor_model_state_dict')
    sd = ckpt['actor_model_state_dict']

    model = torch.nn.Sequential(
        torch.nn.Linear(380, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 23),
    ).eval()

    with torch.no_grad():
        model[0].weight.copy_(sd['actor_module.module.0.weight'])
        model[0].bias.copy_(sd['actor_module.module.0.bias'])
        model[2].weight.copy_(sd['actor_module.module.2.weight'])
        model[2].bias.copy_(sd['actor_module.module.2.bias'])
        model[4].weight.copy_(sd['actor_module.module.4.weight'])
        model[4].bias.copy_(sd['actor_module.module.4.bias'])
        model[6].weight.copy_(sd['actor_module.module.6.weight'])
        model[6].bias.copy_(sd['actor_module.module.6.bias'])

    return model


def abs_stats(diff):
    return {
        'mean': float(np.mean(diff)),
        'max': float(np.max(diff)),
        'p95': float(np.percentile(diff, 95)),
        'p99': float(np.percentile(diff, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare PT actor and ONNX actor outputs on the same trace actor_obs.')
    parser.add_argument('--pt', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, default=None)
    parser.add_argument('--max-steps', type=int, default=0, help='0 means all steps')
    parser.add_argument('--providers', type=str, default='CPUExecutionProvider')
    args = parser.parse_args()

    if not args.pt.is_file():
        raise FileNotFoundError(f'PT checkpoint not found: {args.pt}')
    if not args.onnx.is_file():
        raise FileNotFoundError(f'ONNX file not found: {args.onnx}')
    if not args.trace.is_file():
        raise FileNotFoundError(f'Trace npz not found: {args.trace}')

    trace = np.load(args.trace, allow_pickle=True)
    if 'actor_obs' not in trace:
        raise KeyError('Trace must contain actor_obs')

    actor_obs = np.asarray(trace['actor_obs'], dtype=np.float32)
    if actor_obs.ndim != 2 or actor_obs.shape[1] != 380:
        raise ValueError(f'actor_obs shape must be [T,380], got {actor_obs.shape}')

    steps = actor_obs.shape[0]
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
        actor_obs = actor_obs[:steps]

    model = load_actor_from_checkpoint(args.pt)
    providers = [x.strip() for x in args.providers.split(',') if x.strip()]
    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    pt_out = np.zeros((steps, 23), dtype=np.float32)
    onnx_out = np.zeros((steps, 23), dtype=np.float32)

    for i in range(steps):
        x = actor_obs[i:i + 1]
        with torch.no_grad():
            pt_out[i] = model(torch.from_numpy(x)).cpu().numpy()[0]
        onnx_out[i] = sess.run([out_name], {in_name: x})[0][0].astype(np.float32)

    pt_onnx_abs = np.abs(pt_out - onnx_out)
    step_err = np.mean(pt_onnx_abs, axis=1)
    worst_idx = np.argsort(-step_err)[:10]

    report = {
        'pt': str(args.pt),
        'onnx': str(args.onnx),
        'trace': str(args.trace),
        'steps_compared': int(steps),
        'onnx_input_shape': sess.get_inputs()[0].shape,
        'pt_vs_onnx_abs': abs_stats(pt_onnx_abs),
        'pt_vs_onnx_per_dim_mean': np.mean(pt_onnx_abs, axis=0).tolist(),
        'top_worst_steps': [
            {
                'step': int(i),
                'mean_abs_err': float(step_err[i]),
                'max_abs_err': float(np.max(pt_onnx_abs[i])),
            }
            for i in worst_idx
        ],
    }

    if 'action' in trace:
        trace_action = np.asarray(trace['action'], dtype=np.float32)
        n = min(len(trace_action), steps)
        report['trace_action_vs_onnx_abs'] = abs_stats(np.abs(trace_action[:n] - onnx_out[:n]))
        report['trace_action_vs_pt_abs'] = abs_stats(np.abs(trace_action[:n] - pt_out[:n]))

    s = report['pt_vs_onnx_abs']
    print(f"[COMPARE] steps={steps} mean={s['mean']:.8e} p99={s['p99']:.8e} max={s['max']:.8e}")
    print('[COMPARE] top worst steps:')
    for row in report['top_worst_steps'][:5]:
        print(
            f"  - step={row['step']} mean_abs_err={row['mean_abs_err']:.8e} "
            f"max_abs_err={row['max_abs_err']:.8e}"
        )

    if 'trace_action_vs_onnx_abs' in report:
        t = report['trace_action_vs_onnx_abs']
        print(
            f"[COMPARE] trace_action_vs_onnx mean={t['mean']:.8e} "
            f"p99={t['p99']:.8e} max={t['max']:.8e}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open('w') as f:
            json.dump(report, f, indent=2)
        print(f'[COMPARE] wrote {args.out_json}')


if __name__ == '__main__':
    main()
