#!/usr/bin/env python3
import argparse
import copy
from pathlib import Path

import torch


DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/ASAP/logs/TEST_KickPrimitive/"
    "20260422_145357-MotionTracking_KickPrimitive_RightHitBall-motion_tracking-g1_29dof_anneal_23dof/"
    "model_1200.pt"
)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict from {path}, got {type(ckpt)}")
    for key in ("actor_model_state_dict", "critic_model_state_dict"):
        if key not in ckpt or not isinstance(ckpt[key], dict):
            raise KeyError(f"{path} missing dict key {key}")
    return ckpt


def find_first_linear_weight(state_dict, preferred_prefix):
    candidates = []
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            continue
        if preferred_prefix in name and name.endswith(".weight"):
            candidates.append((name.count("."), name))
    if not candidates:
        for name, tensor in state_dict.items():
            if torch.is_tensor(tensor) and tensor.ndim == 2 and name.endswith(".weight"):
                candidates.append((name.count("."), name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def matching_bias_name(weight_name):
    if not weight_name.endswith(".weight"):
        return None
    return weight_name[: -len(".weight")] + ".bias"


def zero_or_random_(tensor, init_std):
    if init_std == 0.0:
        tensor.zero_()
    else:
        tensor.normal_(mean=0.0, std=init_std)


def resize_first_layer_in_state(state_dict, first_weight_name, input_cols):
    old_weight = state_dict[first_weight_name]
    if old_weight.ndim != 2:
        raise ValueError(f"{first_weight_name} must be a 2D weight, got {tuple(old_weight.shape)}")
    state_dict[first_weight_name] = old_weight.new_zeros((old_weight.shape[0], input_cols))


def adapt_first_layer(
    *,
    src_state,
    tgt_state,
    first_weight_name,
    shared_cols,
    task_cols,
    new_input_init_std,
    label,
):
    logs = []
    if first_weight_name not in src_state or first_weight_name not in tgt_state:
        logs.append(f"[SKIP] {label} first layer {first_weight_name}: missing in source or target")
        return logs

    src_w = src_state[first_weight_name]
    tgt_w = tgt_state[first_weight_name]
    if src_w.ndim != 2 or tgt_w.ndim != 2:
        logs.append(f"[SKIP] {label} first layer {first_weight_name}: expected 2D weights")
        return logs

    if src_w.shape[0] != tgt_w.shape[0]:
        logs.append(
            f"[WARN] {label} first layer out dim mismatch {first_weight_name}: "
            f"source={tuple(src_w.shape)} target={tuple(tgt_w.shape)}, cannot adapt"
        )
        return logs

    copy_cols = min(shared_cols, src_w.shape[1], tgt_w.shape[1])
    tgt_w.copy_(tgt_w)
    tgt_w[:, :copy_cols] = src_w[:, :copy_cols]
    logs.append(
        f"[ADAPT] {label} {first_weight_name}: copied shared prefix cols "
        f"source[0:{copy_cols}] -> target[0:{copy_cols}]"
    )

    if task_cols > 0:
        task_start = max(0, tgt_w.shape[1] - task_cols)
        zero_or_random_(tgt_w[:, task_start:], new_input_init_std)
        logs.append(
            f"[INIT] {label} {first_weight_name}: initialized hitball_task/new cols "
            f"target[{task_start}:{tgt_w.shape[1]}] with "
            f"{'zeros' if new_input_init_std == 0.0 else f'N(0,{new_input_init_std})'}"
        )

    bias_name = matching_bias_name(first_weight_name)
    if bias_name and bias_name in src_state and bias_name in tgt_state:
        if src_state[bias_name].shape == tgt_state[bias_name].shape:
            tgt_state[bias_name].copy_(src_state[bias_name])
            logs.append(f"[LOAD] {label} {bias_name}: copied first-layer bias")
        else:
            logs.append(
                f"[WARN] {label} {bias_name}: bias shape mismatch "
                f"source={tuple(src_state[bias_name].shape)} target={tuple(tgt_state[bias_name].shape)}"
            )

    return logs


def merge_state_dicts(src_state, tgt_state, first_weight_name, label):
    full_loaded = []
    skipped = []
    for name, tgt_tensor in tgt_state.items():
        if name == first_weight_name or name == matching_bias_name(first_weight_name):
            continue
        src_tensor = src_state.get(name)
        if src_tensor is None:
            skipped.append(f"{label} {name}: missing in source")
            continue
        if not torch.is_tensor(src_tensor) or not torch.is_tensor(tgt_tensor):
            if src_tensor == tgt_tensor:
                full_loaded.append(f"{label} {name}")
            else:
                skipped.append(f"{label} {name}: non-tensor mismatch")
            continue
        if src_tensor.shape == tgt_tensor.shape:
            tgt_tensor.copy_(src_tensor)
            full_loaded.append(f"{label} {name}: shape={tuple(tgt_tensor.shape)}")
        else:
            skipped.append(
                f"{label} {name}: shape mismatch source={tuple(src_tensor.shape)} "
                f"target={tuple(tgt_tensor.shape)}"
            )
    return full_loaded, skipped


def print_list(title, values):
    print(title)
    if not values:
        print("  none")
        return
    for value in values:
        print(f"  {value}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a hitball warm-start checkpoint from a motion-tracking checkpoint by "
            "adapting only actor/critic first input layers and loading shape-compatible layers."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Old motion-tracking checkpoint.")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Current hitball initialized checkpoint. If omitted, resize source first layers directly.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-actor-cols", type=int, default=75)
    parser.add_argument("--shared-critic-cols", type=int, default=78)
    parser.add_argument("--hitball-task-cols", type=int, default=24)
    parser.add_argument("--new-input-init-std", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = load_checkpoint(args.source)
    output = copy.deepcopy(load_checkpoint(args.target)) if args.target is not None else copy.deepcopy(source)

    src_actor = source["actor_model_state_dict"]
    src_critic = source["critic_model_state_dict"]
    out_actor = output["actor_model_state_dict"]
    out_critic = output["critic_model_state_dict"]

    actor_first = find_first_linear_weight(src_actor, "actor_module")
    critic_first = find_first_linear_weight(src_critic, "critic_module")
    if actor_first is None:
        raise ValueError("Could not find actor first linear weight")
    if critic_first is None:
        raise ValueError("Could not find critic first linear weight")

    if args.target is None:
        resize_first_layer_in_state(
            out_actor,
            actor_first,
            args.shared_actor_cols + args.hitball_task_cols,
        )
        resize_first_layer_in_state(
            out_critic,
            critic_first,
            args.shared_critic_cols + args.hitball_task_cols,
        )

    print("Assumed hitball actor obs layout:")
    print("  [base_ang_vel(3), projected_gravity(3), dof_pos(23), dof_vel(23), actions(23), hitball_task(24)]")
    print("  shared_actor_prefix_cols=75, hitball_task_cols=24")
    print("Assumed hitball critic obs layout:")
    print("  [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), dof_pos(23), dof_vel(23), actions(23), hitball_task(24)]")
    print("  shared_critic_prefix_cols=78, hitball_task_cols=24")
    print(f"source={args.source}")
    print(f"target={args.target}")
    print(f"output={args.output}")
    print(f"actor_first_layer={actor_first} source_shape={tuple(src_actor.get(actor_first, torch.empty(0)).shape)} target_shape={tuple(out_actor[actor_first].shape)}")
    print(f"critic_first_layer={critic_first} source_shape={tuple(src_critic.get(critic_first, torch.empty(0)).shape)} target_shape={tuple(out_critic[critic_first].shape)}")

    actor_logs = adapt_first_layer(
        src_state=src_actor,
        tgt_state=out_actor,
        first_weight_name=actor_first,
        shared_cols=args.shared_actor_cols,
        task_cols=args.hitball_task_cols,
        new_input_init_std=args.new_input_init_std,
        label="actor",
    )
    critic_logs = adapt_first_layer(
        src_state=src_critic,
        tgt_state=out_critic,
        first_weight_name=critic_first,
        shared_cols=args.shared_critic_cols,
        task_cols=args.hitball_task_cols,
        new_input_init_std=args.new_input_init_std,
        label="critic",
    )

    actor_loaded, actor_skipped = merge_state_dicts(src_actor, out_actor, actor_first, "actor")
    critic_loaded, critic_skipped = merge_state_dicts(src_critic, out_critic, critic_first, "critic")

    # Optimizer states from the target init are not meaningful after parameter surgery.
    output.pop("actor_optimizer_state_dict", None)
    output.pop("critic_optimizer_state_dict", None)
    output["iter"] = int(source.get("iter", 0))
    output["infos"] = {
        "warmstart_source": str(args.source),
        "warmstart_target": str(args.target),
        "shared_actor_cols": args.shared_actor_cols,
        "shared_critic_cols": args.shared_critic_cols,
        "hitball_task_cols": args.hitball_task_cols,
        "new_input_init_std": args.new_input_init_std,
        "note": "Optimizer states removed; train with ++algo.config.load_optimizer=False.",
    }

    print_list("First-layer adaptation:", actor_logs + critic_logs)
    print_list("Fully loaded matching layers:", actor_loaded + critic_loaded)
    print_list("Skipped or mismatched layers:", actor_skipped + critic_skipped)
    print("Optimizer states removed: actor_optimizer_state_dict, critic_optimizer_state_dict")

    if args.dry_run:
        print("dry_run=True, no checkpoint written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"wrote={args.output}")
    print("Use with checkpoint=<output> and ++algo.config.load_optimizer=False")


if __name__ == "__main__":
    main()
